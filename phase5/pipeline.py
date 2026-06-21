"""Phase 5 Pipeline — PPO portfolio optimisation (7 trade tokens, BTC excluded)."""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from phase5.data.builder import load_cached_split
from phase5.data.environment import CryptoPortfolioEnv
from phase5.models.ppo_agent import PPOAgent
from phase5.training.trainer import PPOTrainer
from phase5.utils.config import (
    EPISODE_LEN_TRAIN, LOGS_DIR, MODELS_DIR,
    NUM_EPISODES, PPO_CKPT, PPO_HISTORY, TRADE_TOKENS,
)
from phase5.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class Phase5Result:
    agent:   PPOAgent
    metrics: Dict
    history: Dict = field(default_factory=dict)
    device:  str  = "cpu"


class Phase5Pipeline:
    def __init__(self, num_episodes=NUM_EPISODES, force_retrain=False, device=None):
        self.num_episodes  = num_episodes
        self.force_retrain = force_retrain
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> Phase5Result:
        setup_logging()

        _banner("STEP 1  ─  Load split data  (train 2023–2024 | val 2025)")
        logger.info(f"Trade tokens: {TRADE_TOKENS}  (BTC excluded from portfolio)")

        train_data = load_cached_split("train")
        val_data   = load_cached_split("val")

        _banner("STEP 2  ─  Build environments")
        train_env = CryptoPortfolioEnv(*train_data, episode_len=EPISODE_LEN_TRAIN)
        val_env   = CryptoPortfolioEnv(*val_data,   episode_len=None)
        logger.info(f"Train env: T={train_env.T}  Val env: T={val_env.T}")

        _banner("STEP 3  ─  Train / Load PPO agent")
        agent, history = self._train_or_load(train_env, val_env)

        _banner("STEP 4  ─  Final evaluation on val split (2025)")
        val_metrics = self._evaluate(agent, val_env)

        _banner("PHASE 5 CHECKLIST")
        self._checklist(val_metrics)

        return Phase5Result(agent=agent, metrics=val_metrics,
                            history=history, device=self.device)

    def _train_or_load(self, train_env, val_env):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if PPO_CKPT.exists() and not self.force_retrain:
            logger.info(f"Loading existing checkpoint {PPO_CKPT}")
            agent = PPOAgent(self.device)
            agent.load(PPO_CKPT)
            history = json.load(open(PPO_HISTORY)) if PPO_HISTORY.exists() else {}
            return agent, history
        agent   = PPOAgent(self.device)
        trainer = PPOTrainer(agent, train_env, val_env, self.device)
        history = trainer.train(self.num_episodes, PPO_CKPT)
        return agent, history

    def _evaluate(self, agent: PPOAgent, env: CryptoPortfolioEnv) -> Dict:
        state = env.reset()
        done  = False
        while not done:
            kelly           = env.kelly_fractions()
            action, _, _    = agent.select_action(state, kelly, deterministic=True)
            state, _, done, info = env.step(action)

        ew_sharpe = env.equal_weight_sharpe()
        metrics = {
            "val_sharpe":          info["sharpe"],
            "val_total_return":    info["total_return"],
            "val_max_drawdown":    info["max_drawdown"],
            "ew_benchmark_sharpe": ew_sharpe,
            "sharpe_vs_benchmark": info["sharpe"] - ew_sharpe,
        }
        for k, v in metrics.items():
            logger.info(f"  {k:35s}: {v:.4f}")
        return metrics

    @staticmethod
    def load_result(device="cpu") -> Optional[Phase5Result]:
        if not PPO_CKPT.exists():
            logger.warning("No PPO checkpoint found")
            return None
        agent = PPOAgent(device)
        agent.load(PPO_CKPT)
        history = json.load(open(PPO_HISTORY)) if PPO_HISTORY.exists() else {}
        return Phase5Result(agent=agent, metrics={}, history=history, device=device)

    @staticmethod
    def _checklist(metrics):
        sharpe = metrics.get("val_sharpe", -99)
        ret    = metrics.get("val_total_return", -99)
        dd     = metrics.get("val_max_drawdown", 99)
        checks = {
            f"Val Sharpe > 0  ({sharpe:.3f})":                         sharpe > 0,
            f"Val total return > 0  ({ret:.2%})":                       ret > 0,
            f"Max drawdown < 30%  ({dd:.2%})":                          dd < 0.30,
            f"Beats equal-weight  ({metrics.get('sharpe_vs_benchmark',0):.3f})":
                metrics.get("sharpe_vs_benchmark", -1) > 0,
            "ppo_agent_best.pt saved": PPO_CKPT.exists(),
        }
        all_ok = True
        for lbl, ok in checks.items():
            logger.info(f"  [{'✓' if ok else '✗'}] {lbl}")
            if not ok: all_ok = False
        if all_ok: logger.info("Phase 5 COMPLETE — full pipeline ready")
        else:      logger.warning("Phase 5 finished with failures — review logs")


def _banner(t): logger.info("="*60 + f"\n  {t}\n" + "="*60)
