"""FLAML AutoML + FLAML tune for classical and Lightning QSAR models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from src.qsar_utils import (
    SplitData,
    best_f1_threshold,
    class_imbalance_weight,
    classification_metrics,
    predict_proba_binary,
)


def _merge_custom_hp(*parts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for part in parts:
        if not part:
            continue
        for est, hps in part.items():
            est_hp = dict(out.get(est) or {})
            est_hp.update(hps or {})
            out[est] = est_hp
    return out


def _custom_hp_has_gpu_knobs(custom_hp: Optional[Dict[str, Any]]) -> bool:
    for hps in (custom_hp or {}).values():
        if not isinstance(hps, dict):
            continue
        if "task_type" in hps or "devices" in hps or "device" in hps:
            return True
        tm = hps.get("tree_method")
        if isinstance(tm, dict):
            init = str(tm.get("init_value", "")).lower()
            if "gpu" in init:
                return True
    return False


def _is_alloc_failure(exc: BaseException) -> bool:
    msg = str(exc)
    if "Unable to allocate" in msg or "Cannot allocate memory" in msg:
        return True
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 12:
        return True
    return False


def _looks_gpu_related(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in ("cuda", "gpu", "cupy", "device ordinal", "out of memory"))


def _strip_gpu_knobs(custom_hp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for est, hps in dict(custom_hp or {}).items():
        est_hp = dict(hps or {})
        for k in ("tree_method", "task_type", "devices", "device"):
            est_hp.pop(k, None)
        if est_hp:
            cleaned[est] = est_hp
    return cleaned


def _memory_safe_estimator_hp() -> Dict[str, Any]:
    """Cap CatBoost threads only — do not pin ``n_jobs`` in custom_hp.

    FLAML already forwards AutoML ``n_jobs`` into estimators; setting it again
    via ``custom_hp`` raises ``got multiple values for keyword argument 'n_jobs'``.
    """
    from flaml import tune

    return {
        "catboost": {
            "thread_count": {"domain": tune.choice([1]), "init_value": 1},
        },
    }


def _release_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Classical FLAML AutoML
# ---------------------------------------------------------------------------

DEFAULT_CLASSICAL_ESTIMATORS: List[str] = [
    # Skip lrl1/lrl2 on wide FP matrices: sklearn casts folds to float64 and
    # FLAML then floods logs with _ArrayMemoryError on (fold, n_features).
    "rf",
    "xgboost",
    "lgbm",
    "catboost",
    "extra_tree",
]


def _available_classical_estimators(
    preferred: Optional[Sequence[str]] = None,
) -> List[str]:
    """Filter preferred FLAML estimator names by optional dependency availability."""
    wanted = list(preferred) if preferred is not None else list(DEFAULT_CLASSICAL_ESTIMATORS)
    dep_check = {
        "xgboost": "xgboost",
        "lgbm": "lightgbm",
        "catboost": "catboost",
    }
    available: List[str] = []
    for name in wanted:
        mod = dep_check.get(name)
        if mod is not None:
            try:
                __import__(mod)
            except ImportError:
                continue
        available.append(name)
    return available or ["rf", "extra_tree"]


def _drop_memory_heavy_linear_estimators(
    estimators: Sequence[str],
    n_features: int,
    *,
    feature_threshold: int = 512,
) -> List[str]:
    """Drop lrl1/lrl2 when the feature matrix is wide enough that float64 CV copies hurt."""
    if int(n_features) < feature_threshold:
        return list(estimators)
    return [e for e in estimators if e not in {"lrl1", "lrl2"}]


def default_nn_config(model_type: str) -> Dict[str, Any]:
    """Concrete (non-search-space) defaults used when refitting after a failed tune."""
    cfg: Dict[str, Any] = {
        "hidden_dim": 128,
        "n_layers": 3,
        "dropout": 0.3,
        "lr": 1e-3,
        "batch_size": 128,
        "max_epochs": 20,
        "weight_decay": 1e-5,
    }
    if model_type == "resnet":
        cfg["n_blocks"] = 3
    elif model_type == "transformer":
        cfg["n_heads"] = 4
        cfg["ff_dim"] = 256
    elif model_type == "gnn":
        cfg["n_message_passes"] = 3
    return cfg


def run_classical_flaml(
    split: SplitData,
    time_budget: int = 60,
    metric: str = "ap",
    estimator_list: Optional[List[str]] = None,
    seed: int = 42,
    n_splits: int = 3,
    verbose: int = 0,
    target: Optional[str] = None,
    results_dir: Optional[str] = None,
    use_clearml: bool = True,
    clearml_project: Optional[str] = None,
    use_sample_weight: bool = True,
) -> Dict[str, Any]:
    """Fit FLAML AutoML on a classical tabular SplitData and score on the hold-out test set.

    Defaults to average precision (``ap``) because Tox21 is heavily imbalanced.
    Positive-class upweighting via ``sample_weight`` and an F1-optimal threshold
    (calibrated on the train set) improve F1 / balanced accuracy without hurting ranking metrics.
    """
    from flaml import AutoML
    from src.clearml_tracking import (
        close_task,
        init_task,
        log_metrics,
        log_params,
        save_and_upload_sklearn_model,
    )
    from src.device import configure_training_runtime

    configure_training_runtime()
    estimators = _available_classical_estimators(estimator_list)
    estimators = _drop_memory_heavy_linear_estimators(
        estimators, int(np.shape(split.X_train)[1])
    )
    task_name = f"FLAML/{target}" if target else "FLAML/classical"
    task = None
    if use_clearml:
        task = init_task(
            task_name=task_name,
            project_name=clearml_project,
            tags=["flaml", "classical", target or "multi"],
            params={
                "time_budget": time_budget,
                "metric": metric,
                "seed": seed,
                "n_splits": n_splits,
                "estimator_list": estimators,
                "n_train": int(len(split.y_train)),
                "n_test": int(len(split.y_test)),
                "pos_rate_train": float(np.mean(split.y_train)),
                "n_physchem": int(getattr(split, "n_physchem", 0)),
                "use_sample_weight": use_sample_weight,
            },
        )

    try:
        automl = AutoML()
        settings: Dict[str, Any] = {
            "time_budget": time_budget,
            "metric": metric,
            "task": "classification",
            "log_file_name": "",
            "seed": seed,
            "eval_method": "cv",
            "n_splits": n_splits,
            "verbose": verbose,
            "estimator_list": estimators,
            # Avoid nested joblib forks (Errno 12 despite free host RAM).
            "n_jobs": 1,
        }
        from src.device import flaml_gpu_settings, prefer_boosting_gpu, resolve_training_device

        kind, dev_info = resolve_training_device()
        gpu_settings = flaml_gpu_settings() if prefer_boosting_gpu() else {}
        if gpu_settings:
            settings["custom_hp"] = _merge_custom_hp(
                settings.get("custom_hp"),
                gpu_settings.get("custom_hp"),
            )
        settings["custom_hp"] = _merge_custom_hp(
            settings.get("custom_hp"),
            _memory_safe_estimator_hp(),
        )
        if task is not None:
            log_params(task, {"device": kind, **{f"dev_{k}": v for k, v in dev_info.items() if k != "error"}}, name="device")

        fit_kwargs: Dict[str, Any] = dict(settings)
        spw = class_imbalance_weight(split.y_train) if use_sample_weight else 1.0
        if use_sample_weight:
            sample_weight = np.where(split.y_train == 1, spw, 1.0).astype(np.float32)
            fit_kwargs["sample_weight"] = sample_weight
            # Also pin scale_pos_weight for boosters (same ratio as sample_weight).
            from flaml import tune as flaml_tune

            booster_spw = {
                "scale_pos_weight": {
                    "domain": flaml_tune.choice([float(spw)]),
                    "init_value": float(spw),
                }
            }
            custom_hp = dict(fit_kwargs.get("custom_hp") or {})
            for est in ("xgboost", "lgbm", "catboost"):
                est_hp = dict(custom_hp.get(est) or {})
                est_hp.update(booster_spw)
                custom_hp[est] = est_hp
            fit_kwargs["custom_hp"] = custom_hp

        try:
            automl.fit(split.X_train, split.y_train, **fit_kwargs)
        except Exception as fit_exc:
            had_gpu = _custom_hp_has_gpu_knobs(fit_kwargs.get("custom_hp"))
            should_retry = had_gpu and (_is_alloc_failure(fit_exc) or _looks_gpu_related(fit_exc))
            if not should_retry:
                raise

            print(f"[flaml] GPU/alloc failure ({fit_exc}); stripping GPU knobs and retrying on CPU")
            cleaned = _strip_gpu_knobs(fit_kwargs.get("custom_hp"))
            if cleaned:
                fit_kwargs["custom_hp"] = cleaned
            else:
                fit_kwargs.pop("custom_hp", None)
            fit_kwargs["n_jobs"] = 1
            _release_memory()
            automl = AutoML()
            automl.fit(split.X_train, split.y_train, **fit_kwargs)

        y_prob_train = predict_proba_binary(automl, split.X_train)
        threshold, _ = best_f1_threshold(split.y_train, y_prob_train)
        y_prob = predict_proba_binary(automl, split.X_test)
        y_pred = (y_prob >= threshold).astype(int)
        metrics = classification_metrics(split.y_test, y_prob, y_pred, threshold=threshold)

        best_estimator = getattr(automl, "best_estimator", None)
        best_config = getattr(automl, "best_config", None)
        out_dir = results_dir or "results"
        model_name = f"flaml_{target or 'model'}_{best_estimator or 'auto'}"
        if task is not None:
            log_params(
                task,
                {"best_estimator": best_estimator, "best_config": best_config, "threshold": threshold},
                name="best",
            )
            log_metrics(task, metrics, series="test")
        # Always persist locally; ClearML upload is optional inside the helper.
        model_path = save_and_upload_sklearn_model(
            task,
            automl,
            name=model_name,
            results_dir=out_dir,
            metadata={
                "target": target,
                "best_estimator": best_estimator,
                "best_config": best_config,
                "metrics": metrics,
                "threshold": threshold,
            },
        )

        return {
            "model_name": f"FLAML:{best_estimator}" if best_estimator else "FLAML",
            "best_estimator": best_estimator,
            "best_config": best_config,
            "best_loss": float(getattr(automl, "best_loss", np.nan)),
            "metrics": metrics,
            "threshold": threshold,
            "estimator": automl,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "estimator_list": estimators,
            "clearml_model_path": str(model_path) if model_path else None,
        }
    finally:
        _release_memory()
        close_task(task)


# ---------------------------------------------------------------------------
# Lightning model / data factories (match src.nn_models / src.nn_data)
# ---------------------------------------------------------------------------

NN_MODEL_TYPES = ("mlp", "resnet", "transformer", "gnn")


def _import_nn_backends():
    """Lazy-import Lightning NN modules written by the parallel agent."""
    from src.nn_models import (  # type: ignore
        AtomTransformer,
        FingerprintMLP,
        FingerprintResNet,
        MolGNN,
    )
    from src.nn_data import FingerprintDataModule, GraphDataModule  # type: ignore

    return {
        "mlp": FingerprintMLP,
        "resnet": FingerprintResNet,
        "transformer": AtomTransformer,
        "gnn": MolGNN,
        "fp_dm": FingerprintDataModule,
        "graph_dm": GraphDataModule,
    }


def default_nn_search_space(model_type: str) -> Dict[str, Any]:
    """FLAML ``tune`` search space shared across architectures (plus type-specific knobs)."""
    from flaml import tune

    base: Dict[str, Any] = {
        "hidden_dim": tune.choice([64, 128, 256]),
        "n_layers": tune.choice([2, 3, 4]),
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-4, 5e-3),
        "batch_size": tune.choice([64, 128, 256]),
        "max_epochs": tune.choice([10, 20, 30]),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
    }
    if model_type == "resnet":
        base["n_blocks"] = tune.choice([2, 3, 4])
    elif model_type == "transformer":
        base["n_heads"] = tune.choice([2, 4, 8])
        base["ff_dim"] = tune.choice([128, 256])
    elif model_type == "gnn":
        base["n_message_passes"] = tune.choice([2, 3, 4])
    return base


def default_nn_low_cost_partial_config(model_type: str) -> Dict[str, Any]:
    """Cheap starting point for FLAML cost-frugal search (small net, few epochs)."""
    cfg: Dict[str, Any] = {
        "hidden_dim": 64,
        "n_layers": 2,
        "batch_size": 64,
        "max_epochs": 10,
    }
    if model_type == "resnet":
        cfg["n_blocks"] = 2
    elif model_type == "transformer":
        cfg["n_heads"] = 2
        cfg["ff_dim"] = 128
    elif model_type == "gnn":
        cfg["n_message_passes"] = 2
    return cfg


def _build_datamodule(
    backends: Dict[str, Any],
    model_type: str,
    smiles_train: np.ndarray,
    y_train: np.ndarray,
    smiles_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    n_bits: int = 2048,
    smiles_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Any:
    common = dict(
        smiles_train=smiles_train,
        y_train=y_train,
        smiles_val=smiles_val,
        y_val=y_val,
        batch_size=batch_size,
    )
    if smiles_test is not None:
        common["smiles_test"] = smiles_test
        common["y_test"] = y_test

    if model_type in ("mlp", "resnet"):
        return backends["fp_dm"](**common, n_bits=n_bits)
    return backends["graph_dm"](**common)


def _infer_input_dim(datamodule: Any, n_bits: int = 2048) -> int:
    for attr in ("input_dim", "n_features", "n_bits"):
        if hasattr(datamodule, attr):
            val = getattr(datamodule, attr)
            if callable(val):
                val = val()
            if val is not None:
                return int(val)
    return int(n_bits)


def _pos_weight(y: np.ndarray) -> float:
    return class_imbalance_weight(y)


def _build_model(
    backends: Dict[str, Any],
    model_type: str,
    config: Dict[str, Any],
    input_dim: int,
    node_dim: int = 35,
    pos_weight: float = 1.0,
) -> Any:
    from src.nn_models import build_model

    hparams = {
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "n_layers": int(config.get("n_layers", 3)),
        "dropout": float(config.get("dropout", 0.3)),
        "lr": float(config.get("lr", 1e-3)),
        "weight_decay": float(config.get("weight_decay", 1e-5)),
        "pos_weight": float(pos_weight),
    }
    if model_type in ("mlp", "resnet"):
        hparams["input_dim"] = int(input_dim)
    else:
        hparams["node_dim"] = int(node_dim)
    if model_type == "resnet" and "n_blocks" in config:
        hparams["n_blocks"] = int(config["n_blocks"])
    if model_type == "transformer":
        if "n_heads" in config:
            hparams["n_heads"] = int(config["n_heads"])
        if "ff_dim" in config:
            hparams["ff_dim"] = int(config["ff_dim"])
    if model_type == "gnn" and "n_message_passes" in config:
        hparams["n_message_passes"] = int(config["n_message_passes"])
    return build_model(model_type, **hparams)


def _extract_val_auc(trainer: Any) -> float:
    """Prefer validation ROC-AUC; fall back to 1 - val_loss."""
    metrics = getattr(trainer, "callback_metrics", {}) or {}
    # Convert tensors / numpy to float.
    flat: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            flat[str(k)] = float(v.detach().cpu().item() if hasattr(v, "detach") else v)
        except Exception:
            continue

    for key in (
        "val_auc",
        "val_roc_auc",
        "val_auroc",
        "auc",
        "roc_auc",
        "val/auc",
        "val/roc_auc",
    ):
        if key in flat and np.isfinite(flat[key]):
            return flat[key]

    for key in ("val_loss", "loss", "val/loss"):
        if key in flat and np.isfinite(flat[key]):
            return 1.0 - flat[key]

    return float("nan")


def _predict_proba_lightning(
    model: Any,
    datamodule: Any,
    trainer: Any,
    *,
    split: str = "test",
) -> np.ndarray:
    """Collect positive-class probabilities on ``test`` or ``val`` loader.

    Raises on failure instead of inventing 0.5 probabilities.
    """
    import torch

    errors: List[str] = []
    stage = "test" if split == "test" else "fit"
    loader_name = "test_dataloader" if split == "test" else "val_dataloader"

    try:
        if hasattr(datamodule, "setup"):
            datamodule.setup(stage)
        loader_fn = getattr(datamodule, loader_name, None)
        if loader_fn is None:
            raise RuntimeError(f"no {loader_name}")
        loader = loader_fn()
        if loader is None:
            raise RuntimeError(f"empty {loader_name}")
        probs: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if hasattr(model, "predict_step"):
                    out = model.predict_step(batch, 0)
                elif hasattr(model, "_forward_batch"):
                    logits, _ = model._forward_batch(batch)
                    out = torch.sigmoid(logits.view(-1))
                else:
                    out = model(batch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                arr = out.detach().cpu().numpy().ravel() if hasattr(out, "detach") else np.asarray(out).ravel()
                if arr.min() < 0.0 or arr.max() > 1.0:
                    arr = 1.0 / (1.0 + np.exp(-arr))
                probs.append(arr.astype(float))
        if probs:
            return np.concatenate(probs)
        errors.append(f"{loader_name} yielded no batches")
    except Exception as exc:
        errors.append(f"dataloader path: {exc}")

    if split == "test":
        try:
            preds = trainer.predict(model, datamodule=datamodule)
            chunks = []
            for p in preds:
                if isinstance(p, (tuple, list)):
                    p = p[0]
                if hasattr(p, "detach"):
                    p = p.detach().cpu().numpy()
                chunks.append(np.asarray(p).ravel())
            if not chunks:
                raise RuntimeError("trainer.predict returned no batches")
            arr = np.concatenate(chunks)
            if arr.min() < 0.0 or arr.max() > 1.0:
                arr = 1.0 / (1.0 + np.exp(-arr))
            return arr.astype(float)
        except Exception as exc:
            errors.append(f"trainer.predict path: {exc}")

    raise RuntimeError(f"Lightning predict ({split}) failed: " + " | ".join(errors))


def tune_lightning_model(
    model_type: str,
    smiles_train: np.ndarray,
    y_train: np.ndarray,
    search_space: Optional[Dict[str, Any]] = None,
    num_samples: int = 8,
    time_budget_s: Optional[float] = None,
    val_size: float = 0.2,
    random_state: int = 42,
    n_bits: int = 2048,
    metric: str = "val_auc",
    mode: str = "max",
) -> Dict[str, Any]:
    """Search hyperparameters for a Lightning architecture with FLAML ``tune``.

    The trainable builds a model from ``config``, trains with ``pl.Trainer``
    (CPU, progress bar off), and reports validation ROC-AUC (or ``1 - val_loss``).
    """
    if model_type not in NN_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {NN_MODEL_TYPES}, got {model_type!r}")

    try:
        import lightning.pytorch as pl
    except ImportError:  # pragma: no cover
        import pytorch_lightning as pl  # type: ignore
    from flaml import tune
    from src.device import configure_training_runtime

    configure_training_runtime()
    backends = _import_nn_backends()
    space = search_space or default_nn_search_space(model_type)

    smiles_tr, smiles_va, y_tr, y_va = train_test_split(
        smiles_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train if len(np.unique(y_train)) > 1 else None,
    )
    pw = _pos_weight(y_tr)

    def trainable(config: Dict[str, Any]):
        batch_size = int(config.get("batch_size", 128))
        max_epochs = int(config.get("max_epochs", 20))
        dm = _build_datamodule(
            backends,
            model_type,
            smiles_tr,
            y_tr,
            smiles_va,
            y_va,
            batch_size=batch_size,
            n_bits=n_bits,
        )
        if hasattr(dm, "setup"):
            dm.setup("fit")
        input_dim = _infer_input_dim(dm, n_bits=n_bits)
        node_dim = int(getattr(dm, "atom_feat_dim", 35))
        model = _build_model(
            backends, model_type, config, input_dim=input_dim, node_dim=node_dim, pos_weight=pw
        )
        from src.device import lightning_trainer_kwargs

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            **lightning_trainer_kwargs(),
        )
        trainer.fit(model, datamodule=dm)
        score = _extract_val_auc(trainer)
        if not np.isfinite(score):
            try:
                import torch
                from sklearn.metrics import roc_auc_score

                model.eval()
                ys, ps = [], []
                loader = dm.val_dataloader()
                with torch.no_grad():
                    for batch in loader:
                        if isinstance(batch, dict):
                            yb = batch["y"]
                        elif isinstance(batch, (tuple, list)):
                            yb = batch[-1]
                        else:
                            continue
                        logits, _ = model._forward_batch(batch)
                        arr = torch.sigmoid(logits.view(-1)).detach().cpu().numpy().ravel()
                        ys.append(np.asarray(yb.detach().cpu().numpy() if hasattr(yb, "detach") else yb).ravel())
                        ps.append(arr)
                if ys:
                    y_true = np.concatenate(ys)
                    y_prob = np.concatenate(ps)
                    if len(np.unique(y_true)) > 1:
                        score = float(roc_auc_score(y_true, y_prob))
            except Exception:
                score = 0.0
        tune.report(**{metric: float(score if np.isfinite(score) else 0.0)})

    run_kwargs: Dict[str, Any] = {
        "num_samples": num_samples,
        "metric": metric,
        "mode": mode,
        "verbose": 0,
        "low_cost_partial_config": default_nn_low_cost_partial_config(model_type),
    }
    if time_budget_s is not None:
        run_kwargs["time_budget_s"] = time_budget_s

    analysis = tune.run(trainable, config=space, **run_kwargs)
    best_config = analysis.best_config
    best_score = analysis.best_result.get(metric, float("nan")) if analysis.best_result else float("nan")
    return {
        "model_type": model_type,
        "best_config": best_config,
        "best_score": float(best_score) if best_score is not None else float("nan"),
        "analysis": analysis,
        "metric": metric,
    }


def _refit_and_eval(
    model_type: str,
    best_config: Dict[str, Any],
    smiles_train: np.ndarray,
    y_train: np.ndarray,
    smiles_test: np.ndarray,
    y_test: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 42,
    n_bits: int = 2048,
    target: Optional[str] = None,
    results_dir: Optional[str] = None,
    use_clearml: bool = True,
    clearml_project: Optional[str] = None,
    best_val_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Retrain with the best config and evaluate on the external test set."""
    try:
        import lightning.pytorch as pl
    except ImportError:  # pragma: no cover
        import pytorch_lightning as pl  # type: ignore

    from src.clearml_tracking import (
        close_task,
        init_task,
        log_metrics,
        log_params,
        save_and_upload_lightning_model,
    )

    backends = _import_nn_backends()
    smiles_tr, smiles_va, y_tr, y_va = train_test_split(
        smiles_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train if len(np.unique(y_train)) > 1 else None,
    )
    batch_size = int(best_config.get("batch_size", 128))
    max_epochs = int(best_config.get("max_epochs", 20))

    task = None
    if use_clearml:
        task = init_task(
            task_name=f"Lightning/{model_type}/{target or 'endpoint'}",
            project_name=clearml_project,
            tags=["lightning", "flaml_tune", model_type, target or "multi"],
            params={
                "model_type": model_type,
                "best_config": best_config,
                "n_bits": n_bits,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "best_val_score": best_val_score,
            },
        )

    try:
        dm = _build_datamodule(
            backends,
            model_type,
            smiles_tr,
            y_tr,
            smiles_va,
            y_va,
            batch_size=batch_size,
            n_bits=n_bits,
            smiles_test=smiles_test,
            y_test=y_test,
        )
        if hasattr(dm, "setup"):
            dm.setup("fit")
        input_dim = _infer_input_dim(dm, n_bits=n_bits)
        node_dim = int(getattr(dm, "atom_feat_dim", 35))
        model = _build_model(
            backends,
            model_type,
            best_config,
            input_dim=input_dim,
            node_dim=node_dim,
            pos_weight=_pos_weight(y_tr),
        )
        from src.device import lightning_trainer_kwargs

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            **lightning_trainer_kwargs(),
        )
        trainer.fit(model, datamodule=dm)
        if hasattr(dm, "setup"):
            try:
                dm.setup("test")
            except Exception:
                pass
        y_prob = _predict_proba_lightning(model, dm, trainer, split="test")
        # Align length if needed (graph/FP filtering can drop a few mols)
        y_test_arr = np.asarray(y_test).astype(int).ravel()
        if len(y_prob) != len(y_test_arr):
            n = min(len(y_prob), len(y_test_arr))
            print(
                f"[warn] predict/label length mismatch for {model_type}/{target}: "
                f"probs={len(y_prob)} labels={len(y_test_arr)}; truncating to {n}"
            )
            y_prob = y_prob[:n]
            y_test_arr = y_test_arr[:n]

        threshold = 0.5
        try:
            y_prob_val = _predict_proba_lightning(model, dm, trainer, split="val")
            y_val_arr = np.asarray(y_va).astype(int).ravel()
            if len(y_prob_val) != len(y_val_arr):
                n_val = min(len(y_prob_val), len(y_val_arr))
                y_prob_val = y_prob_val[:n_val]
                y_val_arr = y_val_arr[:n_val]
            threshold, _ = best_f1_threshold(y_val_arr, y_prob_val)
        except Exception as thr_exc:
            print(f"[warn] val threshold failed for {model_type}/{target}: {thr_exc}; using 0.5")

        y_pred = (y_prob >= threshold).astype(int)
        metrics = classification_metrics(y_test_arr, y_prob, y_pred, threshold=threshold)

        if task is not None:
            log_params(task, {"best_config": best_config, "threshold": threshold}, name="best")
            log_metrics(task, metrics, series="test")
            if best_val_score is not None and np.isfinite(best_val_score):
                log_metrics(task, {"val_auc": float(best_val_score)}, series="val")
        model_path = save_and_upload_lightning_model(
            task,
            model,
            name=f"lightning_{model_type}_{target or 'model'}",
            results_dir=results_dir or "results",
            metadata={
                "target": target,
                "model_type": model_type,
                "best_config": best_config,
                "metrics": metrics,
                "best_val_score": best_val_score,
                "threshold": threshold,
            },
        )

        return {
            "model_name": f"Lightning_{model_type}",
            "model_type": model_type,
            "best_config": best_config,
            "metrics": metrics,
            "threshold": threshold,
            "estimator": model,
            "trainer": trainer,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "clearml_model_path": str(model_path) if model_path else None,
        }
    finally:
        close_task(task)


def run_all_nn_flaml_tune(
    smiles_train: np.ndarray,
    y_train: np.ndarray,
    smiles_test: np.ndarray,
    y_test: np.ndarray,
    model_types: Sequence[str] = NN_MODEL_TYPES,
    num_samples: int = 8,
    time_budget_s: Optional[float] = None,
    val_size: float = 0.2,
    random_state: int = 42,
    n_bits: int = 2048,
    search_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
    target: Optional[str] = None,
    results_dir: Optional[str] = None,
    use_clearml: bool = True,
    clearml_project: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Tune each Lightning architecture with FLAML and evaluate on the test set."""
    rows: List[Dict[str, Any]] = []
    for mt in model_types:
        space = None
        if search_spaces and mt in search_spaces:
            space = search_spaces[mt]
        tune_result = tune_lightning_model(
            model_type=mt,
            smiles_train=smiles_train,
            y_train=y_train,
            search_space=space,
            num_samples=num_samples,
            time_budget_s=time_budget_s,
            val_size=val_size,
            random_state=random_state,
            n_bits=n_bits,
        )
        best_cfg = tune_result.get("best_config") or default_nn_config(mt)
        eval_result = _refit_and_eval(
            model_type=mt,
            best_config=best_cfg,
            smiles_train=smiles_train,
            y_train=y_train,
            smiles_test=smiles_test,
            y_test=y_test,
            val_size=val_size,
            random_state=random_state,
            n_bits=n_bits,
            target=target,
            results_dir=results_dir,
            use_clearml=use_clearml,
            clearml_project=clearml_project,
            best_val_score=tune_result.get("best_score"),
        )
        eval_result["best_val_score"] = tune_result["best_score"]
        eval_result["tune_analysis"] = tune_result["analysis"]
        eval_result["source"] = "flaml_tune"
        rows.append(eval_result)
    return rows
