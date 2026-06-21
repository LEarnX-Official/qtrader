"""
Live Inference Engine
Runs phases 1-4 on the latest fetched data, then PPO actor → portfolio weights.
Equivalent to paper_trade.py but designed for live cycling.
Models are loaded ONCE at startup and reused every cycle.
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quantum_trader"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    MODELS_DIR, ALL_TOKENS, TRADE_TOKENS,
    SEQUENCE_LENGTH, FORECAST_HORIZON, NORM_WINDOW,
    RAW_DIR, SUPP_DIR,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QuantumTraderEngine:
    """
    Loads all models once at startup.
    Call .infer(ohlcv, supp) every 4h to get portfolio weights.
    """

    def __init__(self):
        print(f"[Engine] Loading models from {MODELS_DIR} on {DEVICE}")
        self._load_models()
        print("[Engine] All models loaded — ready for inference")

    def _load_models(self):
        from phase2.models.vae                 import MarketVAE
        from phase3.models.transformer         import ObserverAggregatorTransformer
        from phase4.models.pinn                import BornRulePINN
        from phase5.models.ppo_agent           import PPOAgent

        # Phase 2 — VAE
        self.vae = MarketVAE().to(DEVICE)
        ckpt = torch.load(MODELS_DIR / "vae_best.pt",
                          map_location=DEVICE, weights_only=False)
        self.vae.load_state_dict(ckpt["model_state_dict"])
        self.vae.eval()

        # Phase 3 — Transformer
        self.transformer = ObserverAggregatorTransformer().to(DEVICE)
        ckpt = torch.load(MODELS_DIR / "observer_transformer_best.pt",
                          map_location=DEVICE, weights_only=False)
        self.transformer.load_state_dict(ckpt["model_state_dict"])
        self.transformer.eval()

        # Phase 4 — PINN
        self.pinn = BornRulePINN().to(DEVICE)
        ckpt = torch.load(MODELS_DIR / "pinn_collapse_best.pt",
                          map_location=DEVICE, weights_only=False)
        self.pinn.load_state_dict(ckpt["model_state_dict"])
        self.pinn.eval()

        # Phase 5 — PPO Agent
        self.agent = PPOAgent(device=DEVICE)
        self.agent.load(str(MODELS_DIR / "ppo_agent_best.pt"))
        self.agent.actor.eval()
        self.agent.critic.eval()

        from phase3.data.builder import build_multimodal_features
        self._build_mm = build_multimodal_features

    # ── Phase 1: Feature Engineering ─────────────────────────────────────────

    def _phase1(self, ohlcv: Dict, supp: Dict) -> Dict:
        """Build feature matrices and sequences for each token."""
        from phase1.features.engineer import FeatureEngineer

        engineer = FeatureEngineer(ohlcv, supp)
        features = engineer.build_all()

        SEQ_LEN = SEQUENCE_LENGTH
        HORIZON = FORECAST_HORIZON
        NORM_W  = NORM_WINDOW

        seq_data = {}
        for token, df in features.items():
            vals   = df.values.astype(np.float32)
            n_rows = len(vals)
            if n_rows < NORM_W + SEQ_LEN + HORIZON:
                print(f"  [Phase1] {token}: not enough data ({n_rows} rows)")
                continue

            # Only build the LAST sequence (most recent bar)
            i      = n_rows - SEQ_LEN - HORIZON
            seq    = vals[i : i + SEQ_LEN]
            window = vals[i - NORM_W : i]
            mu     = np.nanmean(window, axis=0)
            sigma  = np.nanstd(window,  axis=0)
            sigma  = np.where(sigma < 1e-8, 1.0, sigma)
            seq_n  = (seq - mu) / sigma
            seq_n  = np.where(np.isfinite(seq_n), seq_n, 0.0)

            seq_data[token] = {
                "X":         seq_n[np.newaxis],   # (1, 168, 40)
                "timestamp": df.index[-1],
            }

        return seq_data

    # ── Phase 2: VAE Encoding ─────────────────────────────────────────────────

    def _phase2(self, seq_data: Dict) -> Dict:
        latent_data = {}
        for token, sd in seq_data.items():
            x = torch.from_numpy(sd["X"]).float().to(DEVICE)
            with torch.no_grad():
                z = self.vae.encode(x)     # (1, 32)
            latent_data[token] = {
                "latent":    z.cpu().numpy(),   # (1, 32)
                "timestamp": sd["timestamp"],
            }
        return latent_data

    # ── Phase 3: Transformer ──────────────────────────────────────────────────

    def _phase3(self, latent_data: Dict, seq_data: Dict) -> Dict:
        sigma_data = {}
        for token in ALL_TOKENS:
            if token not in latent_data:
                continue
            latent = latent_data[token]["latent"]   # (1, 32)
            X      = seq_data[token]["X"]           # (1, 168, 40)

            mm = self._build_mm(latent, X, pd.DataFrame(index=[0]))  # (1, 256)
            x  = torch.from_numpy(mm).float().to(DEVICE)
            with torch.no_grad():
                sigma, alpha = self.transformer(x)   # (1,512), (1,1)

            sigma_data[token] = {
                "sigma": sigma.cpu().numpy(),   # (1, 512)
                "alpha": alpha.cpu().numpy(),   # (1, 1)
            }
        return sigma_data

    # ── Phase 4: PINN ─────────────────────────────────────────────────────────

    def _phase4(self, latent_data: Dict, sigma_data: Dict) -> Dict:
        probs_data = {}
        for token in ALL_TOKENS:
            if token not in latent_data or token not in sigma_data:
                continue
            psi0  = torch.from_numpy(latent_data[token]["latent"]).float().to(DEVICE)
            sigma = torch.from_numpy(sigma_data[token]["sigma"]).float().to(DEVICE)
            alpha = torch.from_numpy(sigma_data[token]["alpha"]).float().to(DEVICE)
            with torch.no_grad():
                probs, E, V = self.pinn(psi0, sigma, alpha)   # (1,20), (1,), (1,)
            probs_data[token] = {
                "probs":    probs.cpu().numpy(),          # (1, 20)
                "E":        float(E.cpu().numpy().flat[0]),
                "V":        float(V.cpu().numpy().flat[0]),
            }
        return probs_data

    # ── Phase 5: Build state + PPO inference ──────────────────────────────────

    def _phase5(self, latent_data: Dict, sigma_data: Dict,
                probs_data: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Assemble the 4530-dim state vector and run PPO actor.
        Returns: (weights_7, action_8, cash_weight)
        """
        from phase5.data.environment import CryptoPortfolioEnv
        from phase5.utils.config     import BIN_CENTERS, N_ASSETS, N_STATE_TOKENS, PER_ASSET_DIM

        tok_to_idx = {t: i for i, t in enumerate(ALL_TOKENS)}
        NUM_BINS   = probs_data[ALL_TOKENS[0]]["probs"].shape[1]

        psi0_row  = np.zeros((8, 32),     dtype=np.float32)
        sigma_row = np.zeros((8, 512),    dtype=np.float32)
        alpha_row = np.zeros((8, 1),      dtype=np.float32)
        probs_row = np.full((8, NUM_BINS), 1.0/NUM_BINS, dtype=np.float32)

        for token in ALL_TOKENS:
            k = tok_to_idx[token]
            if token in latent_data:
                psi0_row [k] = latent_data[token]["latent"][0]
            if token in sigma_data:
                sigma_row[k] = sigma_data[token]["sigma"][0]
                alpha_row[k] = sigma_data[token]["alpha"][0]
            if token in probs_data:
                probs_row[k] = probs_data[token]["probs"][0]

        # Build state vector (same as environment._obs but without portfolio context)
        # Use current portfolio weights + metrics from last cycle
        state = np.concatenate([
            psi0_row.flatten(),    # 8×32 = 256
            sigma_row.flatten(),   # 8×512 = 4096
            alpha_row.flatten(),   # 8×1 = 8
            probs_row.flatten(),   # 8×20 = 160
            self._last_weights,    # 7
            [self._pnl, self._vol, self._drawdown],  # 3
        ]).astype(np.float32)      # total: 4530

        # Compute Kelly fractions from probs (trade tokens only, indices 1-7)
        BC     = np.array(BIN_CENTERS, dtype=np.float32)
        p      = probs_row[1:, :]     # (7, 20) — skip BTC
        E      = (p * BC).sum(axis=1)
        V      = (p * (BC - E[:, None])**2).sum(axis=1)
        kelly  = np.where(V > 1e-7, E / V, 0.0)
        kelly  = np.clip(kelly * 0.5, 0.0, 1.0).astype(np.float32)

        with torch.no_grad():
            action, _, _ = self.agent.select_action(
                state, kelly, deterministic=True)

        # action is (8,): 7 asset weights + 1 cash weight
        cash_w   = float(action[7]) if len(action) == 8 else 0.0
        asset_w  = action[:7].copy()

        # ── Eligibility filter ────────────────────────────────────────────────
        # BNB and SOL are NOT on the competition eligible-token list.
        # Zero their weights and redistribute proportionally to eligible tokens.
        from config import INELIGIBLE_TOKENS
        ineligible_idx = [TRADE_TOKENS.index(t) for t in INELIGIBLE_TOKENS
                          if t in TRADE_TOKENS]
        freed = 0.0
        for idx in ineligible_idx:
            freed += asset_w[idx]
            asset_w[idx] = 0.0

        if freed > 0:
            eligible_idx = [i for i in range(len(asset_w))
                            if i not in ineligible_idx]
            elig_sum = asset_w[eligible_idx].sum()
            if elig_sum > 1e-6:
                # Redistribute freed weight proportionally to eligible holdings
                for i in eligible_idx:
                    asset_w[i] += freed * (asset_w[i] / elig_sum)
            else:
                # No eligible holdings — send freed weight to cash
                cash_w += freed

        return asset_w, action, cash_w

    # ── Public API ────────────────────────────────────────────────────────────

    def infer(
        self,
        ohlcv: Dict,
        supp:  Dict,
        current_weights: Optional[np.ndarray] = None,
        current_capital: float = 100_000.0,
        peak_capital:    float = 100_000.0,
        port_returns:    list  = None,
    ) -> Dict:
        """
        Run full inference cycle on latest data.

        Returns dict with:
            weights     : (7,) asset weights
            cash_weight : float
            kelly       : (7,) kelly fractions
            probs       : (8, 20) collapse probabilities
            expected_returns: (8,) expected returns per token
        """
        t0 = time.time()

        # Store portfolio state for state vector
        self._last_weights = (current_weights if current_weights is not None
                              else np.zeros(7, dtype=np.float32))
        pnl = (current_capital - 100_000.0) / 100_000.0
        self._pnl      = float(pnl)
        self._vol      = float(np.std(port_returns[-30:])
                               if port_returns and len(port_returns) >= 30 else 0.0)
        self._drawdown = float((peak_capital - current_capital) /
                               max(peak_capital, 1e-8))

        # Run pipeline
        seq_data    = self._phase1(ohlcv, supp)
        latent_data = self._phase2(seq_data)
        sigma_data  = self._phase3(latent_data, seq_data)
        probs_data  = self._phase4(latent_data, sigma_data)
        asset_w, action, cash_w = self._phase5(
            latent_data, sigma_data, probs_data)

        elapsed = time.time() - t0

        # Compute expected returns for reporting
        from phase5.utils.config import BIN_CENTERS
        BC = np.array(BIN_CENTERS, dtype=np.float32)
        expected_returns = {}
        for token in ALL_TOKENS:
            if token in probs_data:
                p = probs_data[token]["probs"][0]
                expected_returns[token] = float((p * BC).sum())

        result = {
            "weights":          asset_w,
            "cash_weight":      cash_w,
            "action":           action,
            "expected_returns": expected_returns,
            "probs":            {t: probs_data[t]["probs"][0]
                                 for t in probs_data},
            "elapsed_sec":      round(elapsed, 2),
            "timestamp":        pd.Timestamp.now(tz="UTC"),
        }

        print(f"  [Engine] Inference done in {elapsed:.2f}s | "
              f"cash={cash_w:.1%} | "
              f"weights={[f'{w:.2f}' for w in asset_w]}")

        return result


# ── Load supplementary into phase1 format ────────────────────────────────────

def load_live_data() -> Tuple[Dict, Dict]:
    """
    Load latest saved OHLCV and supplementary data for all tokens.
    Returns (ohlcv_dict, supp_dict) ready for FeatureEngineer.
    """
    from phase1.loaders.market        import CryptoDataLoader
    from phase1.loaders.supplementary import SupplementaryLoader

    # Load OHLCV from live CSV files
    ohlcv = {}
    for token in ALL_TOKENS:
        path = RAW_DIR / f"{token}USDT_1h_live.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            # Remove duplicate timestamps — keep last (most recent data)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            # Rename to match CryptoDataLoader output format
            col_map = {
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
                "quote_volume": "QuoteVolume", "trades": "Trades",
                "taker_buy_base": "TakerBuyBase",
                "taker_buy_quote": "TakerBuyQuote",
            }
            df = df.rename(columns=col_map)
            ohlcv[token] = df

    if not ohlcv:
        raise RuntimeError("No OHLCV data found — run fetcher.py first")

    # Build hourly index from BTC
    hourly_idx = ohlcv["BTC"].index

    # Load supplementary
    supp_loader = SupplementaryLoader(
        hourly_index=hourly_idx,
        supp_dir=SUPP_DIR,
    )
    supp = supp_loader.load_all()

    return ohlcv, supp


if __name__ == "__main__":
    print("Testing inference engine...")
    from data.fetcher import run_fetch_cycle
    run_fetch_cycle(full=True)
    ohlcv, supp = load_live_data()
    engine = QuantumTraderEngine()
    result = engine.infer(ohlcv, supp)
    print("\nResult:")
    print(f"  Weights : {result['weights']}")
    print(f"  Cash    : {result['cash_weight']:.1%}")
    print(f"  Time    : {result['elapsed_sec']}s")
    for token, er in result['expected_returns'].items():
        print(f"  E[{token}]  : {er:+.4f}")
