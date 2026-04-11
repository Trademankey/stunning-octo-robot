"""Model zoo — TFT, LSTM+Attention, XGBoost/LightGBM/CatBoost, RL, Ensemble."""


def __getattr__(name: str):
    """Lazy import — avoids pulling torch/sb3 at collection time."""
    if name == "EnsembleModel":
        from models.ensemble import EnsembleModel
        return EnsembleModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
