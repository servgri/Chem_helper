"""GPU / CPU device helpers for Lightning and boosting libraries."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


def force_cpu() -> bool:
    return os.environ.get("TOXMOL_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}


def cuda_available() -> bool:
    if force_cpu():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_summary() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "force_cpu": force_cpu(),
        "cuda_available": False,
        "device_name": "cpu",
        "cuda_device_count": 0,
        "torch_version": None,
        "cuda_version": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available()) and not force_cpu()
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def lightning_trainer_kwargs(precision: Optional[str] = None) -> Dict[str, Any]:
    """Kwargs for ``pl.Trainer`` — GPU when CUDA is visible, else CPU."""
    kwargs: Dict[str, Any] = {
        "devices": 1,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "logger": False,
        "enable_model_summary": False,
    }
    if cuda_available():
        kwargs["accelerator"] = "gpu"
        # prefer bf16/fp16 only if requested; default full precision for small nets stability
        if precision:
            kwargs["precision"] = precision
        else:
            # env TOXMOL_PRECISION=16-mixed|bf16-mixed|32
            prec = os.environ.get("TOXMOL_PRECISION", "").strip()
            if prec:
                kwargs["precision"] = prec
    else:
        kwargs["accelerator"] = "cpu"
    return kwargs


def flaml_gpu_settings() -> Dict[str, Any]:
    """Extra AutoML.fit kwargs to push XGBoost / LightGBM / CatBoost onto GPU."""
    if not cuda_available() and not _gpu_requested_without_torch():
        # Still try boosters if NVIDIA is present even without torch CUDA
        if not _nvidia_smi_ok():
            return {}

    from flaml import tune

    use_gpu = cuda_available() or _nvidia_smi_ok()
    if not use_gpu or force_cpu():
        return {}

    custom_hp: Dict[str, Any] = {
        "xgboost": {
            "tree_method": {
                "domain": tune.choice(["gpu_hist"]),
                "init_value": "gpu_hist",
            },
        },
        "catboost": {
            "task_type": {
                "domain": tune.choice(["GPU"]),
                "init_value": "GPU",
            },
            "devices": {
                "domain": tune.choice(["0"]),
                "init_value": "0",
            },
        },
    }
    # LightGBM GPU needs a GPU-enabled build; enable only when explicitly requested
    if os.environ.get("TOXMOL_LGBM_GPU", "").strip().lower() in {"1", "true", "yes"}:
        custom_hp["lgbm"] = {
            "device": {
                "domain": tune.choice(["gpu"]),
                "init_value": "gpu",
            },
        }
    return {"custom_hp": custom_hp}


def _gpu_requested_without_torch() -> bool:
    return os.environ.get("TOXMOL_BOOST_GPU", "auto").strip().lower() in {"1", "true", "yes", "auto", "on"}


def _nvidia_smi_ok() -> bool:
    if force_cpu():
        return False
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def prefer_boosting_gpu() -> bool:
    """Whether FLAML should try GPU configs for boosting trees."""
    if force_cpu():
        return False
    flag = os.environ.get("TOXMOL_BOOST_GPU", "auto").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    return cuda_available() or _nvidia_smi_ok()


def resolve_training_device() -> Tuple[str, Dict[str, Any]]:
    """Return (\"gpu\"|\"cpu\", summary dict)."""
    summary = device_summary()
    kind = "gpu" if summary["cuda_available"] else "cpu"
    return kind, summary
