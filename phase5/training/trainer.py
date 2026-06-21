"""Phase 5 Trainer — faithful port of crypto/phase5/models/training/trainer.py."""

import json
from pathlib import Path
from typing import Dict

import numpy as np

from phase5.data.environment import CryptoPortfolioEnv
from phase5.models.ppo_agent import PPOAgent
from phase5.utils.config import (
    EVAL_EVERY, LOGS_DIR, NUM_EPISODES, PPO_CKPT, PPO_HISTORY,
)
from phase5.utils.logging import get_logger

logger = get_logger(__name__)


class PPOTrainer:
    def __init__(
        self,
        agent:     PPOAgent,
        train_env: CryptoPortfolioEnv,
        val_env:   CryptoPortfolioEnv,
        device:    str = "cpu",
    ):
        self.agent     = agent
        self.train_env = train_env
        self.val_env   = val_env
        self.device    = device
        self.history: Dict = {
            "ep_reward": [], "ep_sharpe": [], "ep_return": [], "ep_drawdown": [],
            "val_sharpe": [], "val_return": [], "val_drawdown": [],
            "actor_loss": [], "critic_loss": [], "entropy": [],
        }

    # ------------------------------------------------------------------
    def _run_episode(
        self,
        env: CryptoPortfolioEnv,
        collect: bool = True,
        deterministic: bool = False,
    ) -> Dict:
        state = env.reset()
        done  = False
        total_reward = 0.0

        while not done:
            kelly            = env.kelly_fractions()
            action, lp, val  = self.agent.select_action(
                state, kelly, deterministic=deterministic)
            next_state, reward, done, info = env.step(action)

            if collect:
                self.agent.store(state, action, reward, done, val, lp, kelly)

            total_reward += reward
            state = next_state

        return {
            "reward":   total_reward,
            "sharpe":   info["sharpe"],
            "return":   info["total_return"],
            "drawdown": info["max_drawdown"],
            "capital":  info["capital"],
        }

    def _validate(self) -> Dict:
        result = self._run_episode(
            self.val_env, collect=False, deterministic=True)
        result["ew_sharpe"] = self.val_env.equal_weight_sharpe()
        return result

    # ------------------------------------------------------------------
    def train(
        self,
        num_episodes: int  = NUM_EPISODES,
        ckpt_path:    Path = PPO_CKPT,
    ) -> Dict:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        best_val_sharpe = -np.inf

        logger.info(
            f"PPO training — {num_episodes} episodes | "
            f"train T={self.train_env.T}  val T={self.val_env.T}"
        )

        for ep in range(1, num_episodes + 1):
            ep_info     = self._run_episode(self.train_env, collect=True)
            update_info = self.agent.update(next_value=0.0)

            self.history["ep_reward"].append(ep_info["reward"])
            self.history["ep_sharpe"].append(ep_info["sharpe"])
            self.history["ep_return"].append(ep_info["return"])
            self.history["ep_drawdown"].append(ep_info["drawdown"])
            for k in ["actor_loss", "critic_loss", "entropy"]:
                self.history[k].append(update_info.get(k, 0.0))

            if ep % 10 == 0:
                avg = lambda key, n=10: float(np.mean(self.history[key][-n:]))
                logger.info(
                    f"Ep {ep:4d}/{num_episodes} | "
                    f"reward={avg('ep_reward'):7.3f}  "
                    f"sharpe={avg('ep_sharpe'):6.3f}  "
                    f"return={avg('ep_return'):+.3f}  "
                    f"maxDD={avg('ep_drawdown'):.3f} | "
                    f"a_loss={avg('actor_loss'):.4f}  "
                    f"c_loss={avg('critic_loss'):.4f}  "
                    f"entropy={avg('entropy'):.3f}"
                )

            if ep % EVAL_EVERY == 0:
                val = self._validate()
                self.history["val_sharpe"].append(val["sharpe"])
                self.history["val_return"].append(val["return"])
                self.history["val_drawdown"].append(val["drawdown"])
                logger.info(
                    f"  [VAL] sharpe={val['sharpe']:.4f}  "
                    f"return={val['return']:+.3f}  "
                    f"maxDD={val['drawdown']:.3f}  "
                    f"capital=${val['capital']:,.0f}  "
                    f"EW_sharpe={val['ew_sharpe']:.4f}"
                )

                score = (val["sharpe"]
                         + 2.0 * val["return"]
                         - 5.0 * max(0.0, val["drawdown"] - 0.30))
                if score > best_val_sharpe:
                    best_val_sharpe = score
                    self.agent.save(ckpt_path, metadata={
                        "episode":    ep,
                        "val_sharpe": val["sharpe"],
                        "val_return": val["return"],
                        "history":    self.history,
                    })
                    logger.info(
                        f"  ✓ Best saved (score={score:.4f}  sharpe={val['sharpe']:.4f}  return={val['return']:+.3f}  maxDD={val['drawdown']:.3f})")

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(PPO_HISTORY, "w") as f:
            json.dump(
                {k: [float(x) for x in v] for k, v in self.history.items()},
                f, indent=2)
        logger.info(f"History saved → {PPO_HISTORY}")
        return self.history
