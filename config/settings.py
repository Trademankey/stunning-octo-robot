"""
Centralised Pydantic-v2 settings — every knob for the entire bot.

Secrets come from .env; nothing is ever hardcoded.

Design note:  ``trading_pairs`` and ``timeframes`` are stored as
plain comma-separated *strings* so that pydantic-settings can read
them straight from an env-var without attempting (and failing) a
``json.loads()`` parse.  The public API exposes them as ``List[str]``
via ``@computed_field`` properties.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import List, Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Every sub-settings class must carry its own env_file so that
# ``default_factory`` construction picks up the same .env file
# the parent Settings reads.
_ENV_FILE = ".env"


class ExchangeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXCHANGE_", env_file=_ENV_FILE, extra="ignore")

    id: str = "binance"
    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")
    sandbox: bool = True
    rate_limit: int = 1200


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=_ENV_FILE, extra="ignore")

    host: str = "localhost"
    port: int = 5432
    db: str = "trading_bot"
    user: str = "trader"
    password: SecretStr = SecretStr("change_me_strong_password_123")

    @property
    def dsn(self) -> str:
        pw = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pw}@{self.host}:{self.port}/{self.db}"

    @property
    def dsn_sync(self) -> str:
        pw = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pw}@{self.host}:{self.port}/{self.db}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=_ENV_FILE, extra="ignore")

    host: str = "localhost"
    port: int = 6379
    password: SecretStr = SecretStr("")
    db: int = 0

    @property
    def url(self) -> str:
        pw = self.password.get_secret_value()
        auth = f":{pw}@" if pw else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class TelegramSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", env_file=_ENV_FILE, extra="ignore")

    bot_token: SecretStr = SecretStr("")
    chat_id: str = ""
    enabled: bool = False


class OnChainSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=_ENV_FILE, extra="ignore")

    dune_api_key: SecretStr = SecretStr("")
    glassnode_api_key: SecretStr = SecretStr("")


class SentimentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=_ENV_FILE, extra="ignore")

    newsapi_key: SecretStr = SecretStr("")
    reddit_client_id: SecretStr = SecretStr("")
    reddit_client_secret: SecretStr = SecretStr("")
    twitter_bearer_token: SecretStr = SecretStr("")


class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=_ENV_FILE, extra="ignore")

    max_risk_per_trade: float = 0.01
    max_daily_drawdown: float = 0.05
    max_correlated_pairs: int = 3
    kelly_fraction: float = 0.25
    max_open_positions: int = 6
    max_leverage: float = 5.0
    trailing_stop_atr_mult: float = 2.5
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=_ENV_FILE, extra="ignore")

    retrain_interval_hours: int = 8
    optuna_trials: int = 500
    walk_forward_splits: int = 5
    monte_carlo_simulations: int = 1000
    sequence_length: int = 128
    forecast_horizon: int = 12
    batch_size: int = 64
    learning_rate: float = 1e-3
    tft_hidden_size: int = 64
    tft_attention_heads: int = 4
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    rl_total_timesteps: int = 500_000
    ensemble_weights_method: Literal["stacking", "bayesian", "equal"] = "stacking"


class MonitoringSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=_ENV_FILE, extra="ignore")

    prometheus_port: int = 9090
    grafana_port: int = 3000
    health_check_interval: int = 30


class Settings(BaseSettings):
    """Master settings — aggregates all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Trading mode ─────────────────────────────────
    trading_mode: Literal["paper", "live", "backtest"] = "paper"
    market_type: Literal["spot", "futures"] = "futures"
    hedge_mode: bool = True

    # ── Pairs & timeframes (raw CSV strings from env) ─
    trading_pairs_csv: str = Field(
        default="BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,TON/USDT",
        alias="TRADING_PAIRS",
        validation_alias="TRADING_PAIRS",
    )
    timeframes_csv: str = Field(
        default="1m,5m,15m,1h,4h,1d",
        alias="TIMEFRAMES",
        validation_alias="TIMEFRAMES",
    )
    primary_timeframe: str = "15m"

    # ── Logging ──────────────────────────────────────
    log_level: str = "INFO"
    log_dir: Path = Path("./logs")

    # ── Sub-settings (loaded from same .env) ─────────
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    onchain: OnChainSettings = Field(default_factory=OnChainSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    # ── Computed list accessors (the public API) ─────
    @computed_field  # type: ignore[prop-decorator]
    @property
    def trading_pairs(self) -> List[str]:
        return [p.strip() for p in self.trading_pairs_csv.split(",") if p.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def timeframes(self) -> List[str]:
        return [t.strip() for t in self.timeframes_csv.split(",") if t.strip()]


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton factory — call from anywhere."""
    return Settings()
