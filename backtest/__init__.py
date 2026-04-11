"""Backtesting framework — walk-forward, Monte Carlo, stress tests, metrics."""


def __getattr__(name: str):
    """Lazy import — heavy deps (models, torch) are only pulled when accessed."""
    if name == "BacktestEngine":
        from backtest.engine import BacktestEngine
        return BacktestEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
