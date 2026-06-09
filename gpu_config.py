# GPU detection for XGBoost and LightGBM. RandomForest always runs on CPU.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class GPUCapabilities:
    xgboost_cuda: bool
    lightgbm_cuda: bool
    use_gpu_requested: bool
    message: str

    @property
    def any_gpu_tree(self) -> bool:
        return self.xgboost_cuda or self.lightgbm_cuda

    @property
    def ensemble_n_jobs(self) -> int:
        # Avoid parallel base estimators when GPU models are present (VRAM / context contention).
        return 1 if self.any_gpu_tree else -1

    @property
    def tree_cv_n_jobs(self) -> int:
        return 1 if self.any_gpu_tree else -1


def _probe_xgboost_cuda() -> bool:
    try:
        from xgboost import XGBClassifier

        rng = np.random.default_rng(0)
        X = rng.standard_normal((256, 12), dtype=np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
        clf = XGBClassifier(
            n_estimators=8,
            max_depth=3,
            learning_rate=0.3,
            tree_method="hist",
            device="cuda",
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=0,
            n_jobs=1,
        )
        clf.fit(X, y)
        return True
    except Exception:
        return False


def _probe_lightgbm_cuda() -> bool:
    import contextlib, io
    try:
        from lightgbm import LGBMClassifier

        rng = np.random.default_rng(1)
        X = rng.standard_normal((256, 12))
        y = (X[:, 0] > 0).astype(int)
        clf = LGBMClassifier(
            n_estimators=16,
            max_depth=4,
            learning_rate=0.1,
            device="cuda",
            verbose=-1,
            n_jobs=1,
            random_state=0,
        )
        with contextlib.redirect_stderr(io.StringIO()):
            clf.fit(X, y)
        return True
    except BaseException:
        return False


def detect_gpu_capabilities(
    use_gpu: bool | Literal["auto"] = "auto",
) -> GPUCapabilities:
    if use_gpu is False:
        return GPUCapabilities(
            xgboost_cuda=False,
            lightgbm_cuda=False,
            use_gpu_requested=False,
            message="GPU disabled: XGBoost and LightGBM use CPU.",
        )

    x_ok = _probe_xgboost_cuda()
    l_ok = _probe_lightgbm_cuda()
    parts = []
    if x_ok:
        parts.append("XGBoost→CUDA")
    else:
        parts.append("XGBoost→CPU")
    if l_ok:
        parts.append("LightGBM→CUDA")
    else:
        parts.append("LightGBM→CPU")
    msg = "; ".join(parts) + ". RandomForest stays CPU (scikit-learn)."
    if not x_ok and not l_ok:
        msg += " Install CUDA toolkit + GPU builds if you expect GPU acceleration."
    return GPUCapabilities(
        xgboost_cuda=x_ok,
        lightgbm_cuda=l_ok,
        use_gpu_requested=True,
        message=msg,
    )


def xgb_classifier_kwargs(caps: GPUCapabilities) -> dict[str, Any]:
    if caps.xgboost_cuda:
        return {
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": 1,
        }
    return {"tree_method": "hist", "device": "cpu", "n_jobs": -1}


def lgbm_classifier_kwargs(caps: GPUCapabilities) -> dict[str, Any]:
    if caps.lightgbm_cuda:
        return {"device": "cuda", "n_jobs": 1}
    return {"device": "cpu", "n_jobs": -1}


def strip_gpu_kwargs_for_cpu(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    for k in ("device", "device_type", "predictor"):
        out.pop(k, None)
    return out
