"""
Gradient Boosting ensemble — XGBoost + LightGBM + CatBoost.

Optuna Bayesian hyper-parameter search with walk-forward validation.
Each model predicts return direction + magnitude; final output is blended.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import structlog
from joblib import dump, load
from sklearn.metrics import accuracy_score, f1_score, log_loss

from config.settings import get_settings

log = structlog.get_logger(__name__)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False


class GradientBoostEnsemble:
    """Triple GBM ensemble with Optuna tuning."""

    def __init__(self):
        self._settings = get_settings().model
        self._xgb_model: Optional[Any] = None
        self._lgb_model: Optional[Any] = None
        self._cb_model: Optional[Any] = None
        self._best_params: Dict[str, Dict] = {}

    # ── Training ─────────────────────────────────────
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tune: bool = True,
    ) -> Dict[str, float]:
        """Train all three GBMs. Returns validation metrics."""
        metrics = {}

        if tune and HAS_OPTUNA:
            self._tune_all(X_train, y_train, X_val, y_val)

        # XGBoost
        if HAS_XGB:
            params = self._best_params.get("xgb", self._default_xgb_params())
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            self._xgb_model = xgb.train(
                params, dtrain, num_boost_round=1000,
                evals=[(dval, "val")],
                early_stopping_rounds=50, verbose_eval=False,
            )
            preds = self._xgb_model.predict(dval)
            metrics["xgb_logloss"] = float(log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7)))
            log.info("xgb.trained", logloss=metrics["xgb_logloss"])

        # LightGBM
        if HAS_LGB:
            params = self._best_params.get("lgb", self._default_lgb_params())
            ltrain = lgb.Dataset(X_train, label=y_train)
            lval = lgb.Dataset(X_val, label=y_val, reference=ltrain)
            self._lgb_model = lgb.train(
                params, ltrain, num_boost_round=1000,
                valid_sets=[lval],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            preds = self._lgb_model.predict(X_val)
            metrics["lgb_logloss"] = float(log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7)))
            log.info("lgb.trained", logloss=metrics["lgb_logloss"])

        # CatBoost
        if HAS_CB:
            params = self._best_params.get("cb", self._default_cb_params())
            pool_train = cb.Pool(X_train, label=y_train)
            pool_val = cb.Pool(X_val, label=y_val)
            self._cb_model = cb.CatBoostClassifier(**params)
            self._cb_model.fit(
                pool_train, eval_set=pool_val,
                early_stopping_rounds=50, verbose=0,
            )
            preds = self._cb_model.predict_proba(X_val)[:, 1]
            metrics["cb_logloss"] = float(log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7)))
            log.info("cb.trained", logloss=metrics["cb_logloss"])

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Blended probability prediction from all available models."""
        preds = []
        weights = []

        if self._xgb_model is not None:
            p = self._xgb_model.predict(xgb.DMatrix(X))
            preds.append(p)
            weights.append(0.35)

        if self._lgb_model is not None:
            p = self._lgb_model.predict(X)
            preds.append(p)
            weights.append(0.35)

        if self._cb_model is not None:
            p = self._cb_model.predict_proba(X)[:, 1]
            preds.append(p)
            weights.append(0.30)

        if not preds:
            return np.full(X.shape[0], 0.5)

        total_w = sum(weights[: len(preds)])
        blended = sum(p * w for p, w in zip(preds, weights)) / total_w
        return blended

    def feature_importance(self) -> Dict[str, np.ndarray]:
        """Return feature importances from each model."""
        importances = {}
        if self._xgb_model:
            scores = self._xgb_model.get_score(importance_type="gain")
            importances["xgb"] = scores
        if self._lgb_model:
            importances["lgb"] = self._lgb_model.feature_importance(importance_type="gain")
        if self._cb_model:
            importances["cb"] = self._cb_model.get_feature_importance()
        return importances

    # ── Optuna Tuning ────────────────────────────────
    def _tune_all(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> None:
        n_trials = min(self._settings.optuna_trials, 200)  # Per model

        if HAS_XGB:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda t: self._xgb_objective(t, X_train, y_train, X_val, y_val),
                n_trials=n_trials, show_progress_bar=False,
            )
            self._best_params["xgb"] = {**self._default_xgb_params(), **study.best_params}
            log.info("optuna.xgb_done", best_value=study.best_value)

        if HAS_LGB:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda t: self._lgb_objective(t, X_train, y_train, X_val, y_val),
                n_trials=n_trials, show_progress_bar=False,
            )
            self._best_params["lgb"] = {**self._default_lgb_params(), **study.best_params}
            log.info("optuna.lgb_done", best_value=study.best_value)

        if HAS_CB:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda t: self._cb_objective(t, X_train, y_train, X_val, y_val),
                n_trials=n_trials, show_progress_bar=False,
            )
            self._best_params["cb"] = {**self._default_cb_params(), **study.best_params}
            log.info("optuna.cb_done", best_value=study.best_value)

    def _xgb_objective(self, trial, X_tr, y_tr, X_val, y_val) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "tree_method": "hist",
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False,
        )
        preds = model.predict(dval)
        return log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7))

    def _lgb_objective(self, trial, X_tr, y_tr, X_val, y_val) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "verbose": -1,
        }
        ltrain = lgb.Dataset(X_tr, label=y_tr)
        lval = lgb.Dataset(X_val, label=y_val, reference=ltrain)
        model = lgb.train(
            params, ltrain, num_boost_round=500,
            valid_sets=[lval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)
        return log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7))

    def _cb_objective(self, trial, X_tr, y_tr, X_val, y_val) -> float:
        params = {
            "iterations": 500,
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "verbose": 0,
            "early_stopping_rounds": 30,
        }
        model = cb.CatBoostClassifier(**params)
        model.fit(cb.Pool(X_tr, label=y_tr), eval_set=cb.Pool(X_val, label=y_val), verbose=0)
        preds = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7))

    # ── Default params ───────────────────────────────
    @staticmethod
    def _default_xgb_params() -> Dict:
        return {
            "objective": "binary:logistic", "eval_metric": "logloss",
            "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "tree_method": "hist", "verbosity": 0,
        }

    @staticmethod
    def _default_lgb_params() -> Dict:
        return {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 63, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "verbose": -1,
        }

    @staticmethod
    def _default_cb_params() -> Dict:
        return {
            "iterations": 1000, "depth": 6, "learning_rate": 0.05,
            "l2_leaf_reg": 3.0, "verbose": 0, "early_stopping_rounds": 50,
        }

    def save(self, base_path: str) -> None:
        if self._xgb_model:
            self._xgb_model.save_model(f"{base_path}/xgb_model.json")
        if self._lgb_model:
            self._lgb_model.save_model(f"{base_path}/lgb_model.txt")
        if self._cb_model:
            self._cb_model.save_model(f"{base_path}/cb_model.cbm")
        dump(self._best_params, f"{base_path}/gbm_params.joblib")

    def load(self, base_path: str) -> None:
        import os
        if HAS_XGB and os.path.exists(f"{base_path}/xgb_model.json"):
            self._xgb_model = xgb.Booster()
            self._xgb_model.load_model(f"{base_path}/xgb_model.json")
        if HAS_LGB and os.path.exists(f"{base_path}/lgb_model.txt"):
            self._lgb_model = lgb.Booster(model_file=f"{base_path}/lgb_model.txt")
        if HAS_CB and os.path.exists(f"{base_path}/cb_model.cbm"):
            self._cb_model = cb.CatBoostClassifier()
            self._cb_model.load_model(f"{base_path}/cb_model.cbm")
        params_path = f"{base_path}/gbm_params.joblib"
        if os.path.exists(params_path):
            self._best_params = load(params_path)
