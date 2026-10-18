"""Gradient-boosted scorer, served by Snap ML.

Two training routes, both scored through Snap ML so inference can use the IBM Z AI
accelerator (Snap ML picks 'zdnn_tensors' when the accelerator is present, else
'compress_trees' on CPU):
- ``snap``: Snap ML's own BoostingMachineClassifier.
- ``xgb``: XGBoost, exported as JSON and imported into Snap ML (the PDF's fallback if Snap ML's
  booster trails the published F1).
"""
import os
import pickle
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from snapml import BoostingMachineClassifier

SNAP_PARAMS = dict(num_round=400, learning_rate=0.1, max_depth=8, early_stopping_rounds=30,
                   class_weight="balanced", colsample_bytree=0.8, subsample=0.8,
                   use_histograms=True)
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.08, max_depth=8, min_child_weight=1,
                  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5.0,
                  tree_method="hist", eval_metric="aucpr", early_stopping_rounds=40)


@dataclass
class Scorer:
    kind: str
    booster: object = field(repr=False)
    feature_count: int = 0
    train_seconds: float = 0.0

    def predict(self, X: np.ndarray, n_jobs: int = 8) -> np.ndarray:
        if getattr(self, "_n_jobs", None) != n_jobs:
            self.booster.set_params(n_jobs=n_jobs)
            self._n_jobs = n_jobs
        return self.booster.predict_proba(np.ascontiguousarray(X, dtype=np.float32))[:, 1]

    @property
    def can_explain(self) -> bool:
        """Reason codes need the portable XGBoost model (and the optional xgboost package)."""
        if self.kind != "xgb" or not getattr(self, "xgb_json", None):
            return False
        try:
            import xgboost  # noqa: F401
        except ImportError:
            return False
        return True

    def contributions(self, X: np.ndarray) -> np.ndarray | None:
        """Per-feature contributions to the raw score (log-odds), bias column dropped.

        Uses XGBoost's approximate (Saabas) attribution: about 2-4 ms for a handful of rows on
        a 600-tree model, versus ~40 ms per row for exact TreeSHAP, so it fits the hot path.
        Only called for held or frozen payments."""
        if not self.can_explain:
            return None
        import xgboost as xgb
        if getattr(self, "_xgb", None) is None:
            b = xgb.Booster()
            b.load_model(bytearray(self.xgb_json.encode()))
            b.set_param({"nthread": 1})
            self._xgb = b
        d = xgb.DMatrix(np.ascontiguousarray(X, dtype=np.float32), nthread=1)
        # all trees, like the Snap ML import that produces the score being explained
        return self._xgb.predict(d, pred_contribs=True, approx_contribs=True)[:, :-1]

    @property
    def tree_format(self) -> str:
        """'zdnn_tensors' when scoring on the IBM Z AI accelerator, 'compress_trees' on CPU."""
        b = getattr(self.booster, "booster_", None)
        return getattr(b, "optimized_tree_format_", "native")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Scorer":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_n_jobs", None)
        state.pop("_xgb", None)
        if self.kind == "xgb":  # keep the portable XGBoost JSON, re-import into Snap ML on load
            state["booster"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.kind == "xgb":
            self.booster = _snap_from_xgb_json(state["xgb_json"])


def _snap_from_xgb_json(text: str):
    """FF_TREE_FORMAT=compress_trees|zdnn_tensors forces CPU or AI-accelerator scoring (default:
    'auto', which picks the accelerator when present) for the accelerator-vs-CPU comparison."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(text)
    snap = BoostingMachineClassifier()
    snap.import_model(f.name, "xgb_json", tree_format=os.environ.get("FF_TREE_FORMAT", "auto"))
    Path(f.name).unlink()
    return snap


def train(X, y, X_val, y_val, kind: str = "xgb", params: dict | None = None,
          n_jobs: int = 8) -> Scorer:
    t0 = time.time()
    X = np.ascontiguousarray(X, dtype=np.float32)
    X_val = np.ascontiguousarray(X_val, dtype=np.float32)
    if kind == "snap":
        m = BoostingMachineClassifier(n_jobs=n_jobs, random_state=0,
                                      **{**SNAP_PARAMS, **(params or {})})
        m.fit(X, y.astype(np.float32), X_val=X_val, y_val=y_val.astype(np.float32))
        s = Scorer("snap", m, X.shape[1])
    elif kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_jobs=n_jobs, random_state=0, **{**XGB_PARAMS, **(params or {})})
        m.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        m.get_booster().save_model(path)
        text = Path(path).read_text()
        Path(path).unlink()
        s = Scorer("xgb", _snap_from_xgb_json(text), X.shape[1])
        s.xgb_json = text
        s.best_iteration = int(getattr(m, "best_iteration", -1) or -1)
    else:
        raise ValueError(kind)
    s.train_seconds = time.time() - t0
    return s
