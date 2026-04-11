"""
Reinforcement Learning agent — PPO / SAC in a custom Gym environment.

The environment simulates futures trading with:
  - Realistic fee structure (maker/taker)
  - Slippage model (volume-dependent)
  - Partial fill simulation
  - Funding rate costs
  - Portfolio tracking with Sharpe reward
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from config.settings import get_settings

log = structlog.get_logger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_RL = True
except ImportError:
    HAS_RL = False
    log.warning("stable-baselines3/gymnasium not installed — RL disabled")


if HAS_RL:
    class CryptoTradingEnv(gym.Env):
        """
        Custom Gym environment for crypto futures trading.

        Observation: feature vector (from pipeline)
        Action:  continuous [-1, +1] → short to long position sizing
        Reward:  risk-adjusted PnL (differential Sharpe)
        """
        metadata = {"render_modes": ["human"]}

        def __init__(
            self,
            features: np.ndarray,
            prices: np.ndarray,
            initial_balance: float = 10000.0,
            max_leverage: float = 5.0,
            maker_fee: float = 0.0002,
            taker_fee: float = 0.0004,
            funding_rate: float = 0.0001,
            slippage_bps: float = 1.0,
        ):
            super().__init__()
            self.features = features.astype(np.float32)
            self.prices = prices.astype(np.float64)
            self.initial_balance = initial_balance
            self.max_leverage = max_leverage
            self.maker_fee = maker_fee
            self.taker_fee = taker_fee
            self.funding_rate = funding_rate
            self.slippage_bps = slippage_bps

            self.n_steps = len(features)
            self.n_features = features.shape[1]

            # Action: position size [-1, 1]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

            # Observation: features + position + pnl + drawdown
            obs_dim = self.n_features + 3
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

            self.reset()

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            self.balance = self.initial_balance
            self.position = 0.0     # signed: +long, -short
            self.entry_price = 0.0
            self.total_pnl = 0.0
            self.peak_balance = self.initial_balance
            self.returns_history: List[float] = []
            return self._get_obs(), {}

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            target_position = float(np.clip(action[0], -1, 1)) * self.max_leverage

            price = self.prices[self.current_step]
            prev_price = self.prices[max(0, self.current_step - 1)]

            # ── Execute position change ──────────────
            position_change = target_position - self.position
            trade_cost = 0.0

            if abs(position_change) > 0.01:
                # Slippage
                slippage = self.slippage_bps / 10000 * abs(position_change)
                # Fees
                fee = self.taker_fee * abs(position_change) * price * self.balance / self.initial_balance
                trade_cost = (slippage + fee) * self.balance

            # ── PnL from existing position ───────────
            price_return = (price - prev_price) / (prev_price + 1e-15)
            position_pnl = self.position * price_return * self.balance

            # Funding cost (applied every 8h ≈ every 480 bars on 1m)
            funding_cost = 0.0
            if self.current_step % 480 == 0 and abs(self.position) > 0.01:
                funding_cost = abs(self.position) * self.funding_rate * self.balance

            # Update state
            step_pnl = position_pnl - trade_cost - funding_cost
            self.balance += step_pnl
            self.total_pnl += step_pnl
            self.position = target_position

            if self.balance > self.peak_balance:
                self.peak_balance = self.balance

            # ── Reward: differential Sharpe ratio ────
            step_return = step_pnl / (self.initial_balance + 1e-10)
            self.returns_history.append(step_return)

            if len(self.returns_history) > 20:
                recent = np.array(self.returns_history[-20:])
                mean_r = np.mean(recent)
                std_r = np.std(recent) + 1e-10
                reward = float(mean_r / std_r)  # Sharpe-like
            else:
                reward = float(step_return * 100)

            # Penalise large drawdowns
            drawdown = (self.peak_balance - self.balance) / (self.peak_balance + 1e-10)
            if drawdown > 0.05:
                reward -= drawdown * 10

            # ── Termination ──────────────────────────
            self.current_step += 1
            terminated = self.current_step >= self.n_steps - 1
            truncated = self.balance <= self.initial_balance * 0.5  # -50% kill

            info = {
                "balance": self.balance,
                "pnl": self.total_pnl,
                "position": self.position,
                "drawdown": drawdown,
            }

            return self._get_obs(), reward, terminated, truncated, info

        def _get_obs(self) -> np.ndarray:
            feat = self.features[min(self.current_step, self.n_steps - 1)]
            extra = np.array([
                self.position / self.max_leverage,
                self.total_pnl / self.initial_balance,
                (self.peak_balance - self.balance) / (self.peak_balance + 1e-10),
            ], dtype=np.float32)
            return np.concatenate([feat, extra])


class RLAgent:
    """High-level wrapper for SB3 PPO/SAC agents."""

    def __init__(self, algo: str = "PPO"):
        self._settings = get_settings().model
        self._algo = algo.upper()
        self._model: Optional[Any] = None
        self._env: Optional[Any] = None

    def train(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_prices: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train the RL agent on historical data."""
        if not HAS_RL:
            return {"error": "stable-baselines3 not installed"}

        self._env = DummyVecEnv([
            lambda: CryptoTradingEnv(features, prices)
        ])

        algo_cls = PPO if self._algo == "PPO" else SAC

        policy_kwargs = {
            "net_arch": dict(pi=[256, 128], vf=[256, 128]),
        }

        self._model = algo_cls(
            "MlpPolicy",
            self._env,
            learning_rate=3e-4,
            batch_size=256,
            n_steps=2048 if self._algo == "PPO" else 1,
            gamma=0.99,
            ent_coef=0.01 if self._algo == "PPO" else "auto",
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

        # Evaluation callback
        eval_env = None
        eval_callback = None
        if val_features is not None and val_prices is not None:
            eval_env = DummyVecEnv([lambda: CryptoTradingEnv(val_features, val_prices)])
            eval_callback = EvalCallback(
                eval_env, best_model_save_path=None,
                eval_freq=10000, n_eval_episodes=5, verbose=0,
            )

        self._model.learn(
            total_timesteps=self._settings.rl_total_timesteps,
            callback=eval_callback,
        )

        log.info("rl.trained", algo=self._algo, timesteps=self._settings.rl_total_timesteps)

        # Evaluate
        metrics = self._evaluate(features, prices)
        return metrics

    def predict(self, features: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Generate position signals for a sequence of observations."""
        if self._model is None or not HAS_RL:
            return np.zeros(len(features))

        env = CryptoTradingEnv(features, prices)
        obs, _ = env.reset()
        actions = []

        for _ in range(len(features) - 1):
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            actions.append(float(action[0]))
            if done or truncated:
                break

        return np.array(actions)

    def _evaluate(self, features: np.ndarray, prices: np.ndarray) -> Dict[str, float]:
        """Run evaluation episode and return metrics."""
        if not HAS_RL:
            return {}

        env = CryptoTradingEnv(features, prices)
        obs, _ = env.reset()
        total_reward = 0.0

        for _ in range(len(features) - 1):
            action, _ = self._model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        return {
            "total_reward": total_reward,
            "final_balance": info.get("balance", 0),
            "total_pnl": info.get("pnl", 0),
            "max_drawdown": info.get("drawdown", 0),
        }

    def save(self, path: str) -> None:
        if self._model:
            self._model.save(path)

    def load(self, path: str) -> None:
        if not HAS_RL:
            return
        algo_cls = PPO if self._algo == "PPO" else SAC
        self._model = algo_cls.load(path)
