"""ClearML experiment tracking and model registry helpers.

Works with ClearML Cloud or a self-hosted server. If credentials are missing,
falls back to offline mode (artifacts under ``~/.clearml`` / local results)
unless ``CLEARML_ENABLED=0``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

PathLike = Union[str, Path]

DEFAULT_PROJECT = os.environ.get("CLEARML_PROJECT", "ToxMol/Tox21-QSAR")
MODELS_SUBDIR = "models"


def clearml_enabled() -> bool:
    flag = os.environ.get("CLEARML_ENABLED", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _has_credentials() -> bool:
    if os.environ.get("CLEARML_API_ACCESS_KEY") and os.environ.get("CLEARML_API_SECRET_KEY"):
        return True
    conf = Path.home() / "clearml.conf"
    return conf.exists()


def _close_current_task() -> None:
    """ClearML allows only one main ``Task`` per process — close leftovers."""
    try:
        from clearml import Task
    except ImportError:
        return
    try:
        current = Task.current_task()
    except Exception:
        current = None
    if current is None:
        return
    try:
        current.flush(wait_for_uploads=False)
    except Exception:
        pass
    try:
        current.close()
    except Exception:
        pass


def init_task(
    task_name: str,
    project_name: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    reuse_last: bool = False,
) -> Any:
    """Create a ClearML Task or return ``None`` if tracking is disabled."""
    if not clearml_enabled():
        return None
    try:
        from clearml import Task
    except ImportError:
        print("[clearml] package not installed — skipping tracking")
        return None

    offline = os.environ.get("CLEARML_OFFLINE_MODE", "").strip().lower() in {"1", "true", "yes"}
    if not _has_credentials() and not offline:
        # Local-friendly default: keep going without a remote server.
        offline = True
        os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")
        print("[clearml] no credentials — enabling offline mode")

    if offline:
        try:
            Task.set_offline(offline_mode=True)
        except Exception:
            pass

    # Notebooks create many sequential tasks; a leftover main task blocks Task.init.
    _close_current_task()

    task = Task.init(
        project_name=project_name or DEFAULT_PROJECT,
        task_name=task_name,
        tags=list(tags or []),
        reuse_last_task_id=reuse_last,
        auto_connect_frameworks={"pytorch": False, "matplotlib": True, "tensorboard": False},
    )
    if params:
        # ClearML prefers flat JSON-serializable dicts
        flat = _jsonable(params)
        task.connect(flat, name="config")
    return task


def log_metrics(
    task: Any,
    metrics: Dict[str, float],
    series: str = "test",
    iteration: int = 0,
) -> None:
    if task is None:
        return
    logger = task.get_logger()
    for key, value in metrics.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        logger.report_scalar(title=series, series=str(key), value=v, iteration=iteration)


def log_params(task: Any, params: Dict[str, Any], name: str = "params") -> None:
    if task is None:
        return
    try:
        task.connect(_jsonable(params), name=name)
    except Exception as exc:
        print(f"[clearml] connect params failed: {exc}")


def log_artifact_file(task: Any, path: PathLike, name: Optional[str] = None) -> None:
    if task is None:
        return
    path = Path(path)
    if not path.exists():
        return
    try:
        task.upload_artifact(name=name or path.name, artifact_object=str(path))
    except Exception as exc:
        print(f"[clearml] artifact upload failed: {exc}")


def save_and_upload_sklearn_model(
    task: Any,
    model: Any,
    name: str,
    results_dir: PathLike,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Persist a classical/FLAML model with joblib and register in ClearML."""
    import joblib

    out_dir = Path(results_dir) / MODELS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe(name)}.joblib"
    joblib.dump({"model": model, "metadata": metadata or {}}, path)

    if task is not None:
        try:
            from clearml import OutputModel

            output = OutputModel(task=task, framework="scikit-learn", name=name)
            output.update_weights(weights_filename=str(path))
            if metadata:
                try:
                    output.update_labels(_jsonable(metadata))
                except Exception:
                    pass
            try:
                output.publish()
            except Exception:
                # offline / no ACL — weights already attached to the task
                pass
        except Exception as exc:
            print(f"[clearml] sklearn OutputModel failed ({exc}); uploading artifact only")
            log_artifact_file(task, path, name=name)
    return path


def save_and_upload_lightning_model(
    task: Any,
    model: Any,
    name: str,
    results_dir: PathLike,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Save Lightning ``state_dict`` (+ hparams) and register in ClearML."""
    import torch

    out_dir = Path(results_dir) / MODELS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe(name)}.pt"
    payload = {
        "state_dict": model.state_dict(),
        "hparams": dict(getattr(model, "hparams", {}) or {}),
        "metadata": metadata or {},
        "class_name": model.__class__.__name__,
    }
    torch.save(payload, path)

    if task is not None:
        try:
            from clearml import OutputModel

            output = OutputModel(task=task, framework="PyTorch", name=name)
            output.update_weights(weights_filename=str(path))
            try:
                output.publish()
            except Exception:
                pass
        except Exception as exc:
            print(f"[clearml] torch OutputModel failed ({exc}); uploading artifact only")
            log_artifact_file(task, path, name=name)
    return path


def close_task(task: Any) -> None:
    if task is None:
        return
    try:
        task.flush(wait_for_uploads=False)
    except Exception:
        pass
    try:
        task.close()
    except Exception:
        pass
    # Ensure ClearML main-task slot is free for the next init_task call.
    _close_current_task()


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:180]


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)
