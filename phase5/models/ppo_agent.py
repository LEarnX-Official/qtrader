"""
PPO Agent — faithful port of crypto/phase5/models/ppo_agent.py.

Key design (from working crypto version):
  - Dirichlet policy with learnable concentration log_conc
  - concentration = softplus(log_conc) * mean_w * N, clamped to [0.1, 50]
  - Actions renormalised before log_prob in update (a_b / a_b.sum())
  - Deterministic eval (use mean_w directly, no sampling)
  - log_conc stored separately in checkpoint
  - Actor and critic updated in separate backward passes
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase5.models.actor_critic import ActorNetwork, CriticNetwork
from phase5.utils.config import (
    BATCH_SIZE, CLIP_EPS, GAMMA, GAE_LAMBDA, GRAD_CLIP,
    LR_ACTOR, LR_CRITIC, N_ASSETS, PPO_EPOCHS, STATE_DIM,
)
from phase5.utils.logging import get_logger

logger = get_logger(__name__)


class PPOAgent:
    def __init__(self, device="cpu"):
        self.device   = device
        self.n_assets = N_ASSETS + 1   # 7 trade assets + 1 cash dim for Dirichlet

        self.actor  = ActorNetwork().to(device)
        self.critic = CriticNetwork().to(device)

        self.actor_opt  = torch.optim.AdamW(
            self.actor.parameters(), lr=LR_ACTOR,  weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=LR_CRITIC, weight_decay=1e-4)

        # Learnable Dirichlet concentration temperature
        self.log_conc = nn.Parameter(torch.zeros(1, device=device))

        self._clear_buffer()
        logger.info(
            f"PPOAgent — actor: {sum(p.numel() for p in self.actor.parameters()):,}  "
            f"critic: {sum(p.numel() for p in self.critic.parameters()):,}"
        )

    def _clear_buffer(self):
        self._states:    List[np.ndarray] = []
        self._actions:   List[np.ndarray] = []
        self._rewards:   List[float]      = []
        self._dones:     List[bool]       = []
        self._values:    List[float]      = []
        self._log_probs: List[float]      = []
        self._kellys:    List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        state:         np.ndarray,
        kelly:         np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        s_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        k_t = torch.from_numpy(kelly).float().unsqueeze(0).to(self.device)

        mean_w = self.actor(s_t, k_t).squeeze(0)   # (N,) on simplex
        value  = self.critic(s_t).item()

        if deterministic:
            return mean_w.cpu().numpy(), 0.0, value

        # Dirichlet sampling
        conc    = F.softplus(self.log_conc).clamp(0.5, 20.0)
        alpha_d = (mean_w * self.n_assets * conc).clamp(0.1, 50.0)
        dist    = torch.distributions.Dirichlet(alpha_d)
        sample  = dist.sample()
        log_prob = dist.log_prob(sample).item()

        return sample.cpu().numpy(), log_prob, value

    def store(self, state, action, reward, done, value, log_prob, kelly):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        self._values.append(value)
        self._log_probs.append(log_prob)
        self._kellys.append(kelly)

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        vals    = np.array(self._values + [next_value], dtype=np.float32)
        rewards = np.array(self._rewards, dtype=np.float32)
        dones   = np.array(self._dones,   dtype=np.float32)
        adv     = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            nt       = 1.0 - dones[t]
            delta    = rewards[t] + GAMMA * vals[t+1] * nt - vals[t]
            adv[t]   = last_gae = delta + GAMMA * GAE_LAMBDA * nt * last_gae
        returns = adv + vals[:-1]
        return adv, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        if not self._states:
            return {}

        adv, returns = self._gae(next_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states   = torch.from_numpy(np.array(self._states,    np.float32)).to(self.device)
        actions  = torch.from_numpy(np.array(self._actions,   np.float32)).to(self.device)
        kellys   = torch.from_numpy(np.array(self._kellys,    np.float32)).to(self.device)
        old_lp   = torch.from_numpy(np.array(self._log_probs, np.float32)).to(self.device)
        adv_t    = torch.from_numpy(adv).to(self.device)
        ret_t    = torch.from_numpy(returns).to(self.device)

        self._clear_buffer()

        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(len(states))
            for start in range(0, len(states), BATCH_SIZE):
                b = idx[start:start + BATCH_SIZE]
                s_b   = states[b];  a_b  = actions[b]
                k_b   = kellys[b];  olp_b = old_lp[b]
                adv_b = adv_t[b];   ret_b = ret_t[b]

                # ── Actor ─────────────────────────────────────────────────
                mean_w  = self.actor(s_b, k_b)
                conc    = F.softplus(self.log_conc).clamp(0.5, 20.0)
                alpha_d = (mean_w * self.n_assets * conc).clamp(0.1, 50.0)
                dist    = torch.distributions.Dirichlet(alpha_d)

                # Renormalise stored actions (numerical safety — same as crypto version)
                a_safe  = a_b.clamp(1e-6, 1.0)
                a_safe  = a_safe / a_safe.sum(dim=1, keepdim=True)
                new_lp  = dist.log_prob(a_safe)
                entropy = dist.entropy().mean()

                ratio  = (new_lp - olp_b).exp()
                surr1  = ratio * adv_b
                surr2  = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
                a_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                self.actor_opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
                self.actor_opt.step()

                # ── Critic ────────────────────────────────────────────────
                val_pred = self.critic(s_b).squeeze(-1)
                c_loss   = F.mse_loss(val_pred, ret_b)

                self.critic_opt.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
                self.critic_opt.step()

                actor_losses.append(a_loss.item())
                critic_losses.append(c_loss.item())
                entropies.append(entropy.item())

        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy":     float(np.mean(entropies)),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path, metadata: dict = None) -> None:
        torch.save({
            "actor":    self.actor.state_dict(),
            "critic":   self.critic.state_dict(),
            "log_conc": self.log_conc.data,
            **(metadata or {}),
        }, path)

    def load(self, path) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "log_conc" in ckpt:
            self.log_conc.data = ckpt["log_conc"].to(self.device)
        return ckpt
