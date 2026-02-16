"""Load and run Tox21 QSAR models."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from django.conf import settings

from molecules.services import lipinski_veber, parse_smiles
from predictions.model_registry import MODEL_SPECS, ModelSpec, SPECS_BY_TARGET
from src.qsar_utils import (
    PHYSCHEM_COLS,
    TOX21_TARGETS,
    build_fp_physchem_matrix,
    compute_descriptors,
    load_tox21,
    morgan_fingerprint,
    predict_proba_binary,
    prepare_endpoint,
    smiles_to_mol,
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_LOADED: Dict[str, "LoadedModel"] = {}
_SCALERS: Dict[str, Any] = {}
_STATUS: Dict[str, Any] = {"ready": False, "entries": {}}


@dataclass
class LoadedModel:
    target: str
    family: str
    model_name: str
    kind: str  # classical | lightning
    artifact: str
    roc_auc: float
    fallback: bool
    model: Any
    threshold: float
    lightning_type: Optional[str] = None  # mlp|resnet|transformer|gnn


def models_dir() -> Path:
    return Path(settings.TOXMOL_MODELS_DIR)


def _fit_scaler_for_target(target: str) -> Optional[Any]:
    """Reconstruct physchem StandardScaler with the same split seed as training."""
    if target in _SCALERS:
        return _SCALERS[target]
    try:
        from sklearn.preprocessing import StandardScaler

        df = load_tox21()
        X_all, valid_mask, n_phys = build_fp_physchem_matrix(
            df, n_bits=settings.TOXMOL_N_BITS, radius=settings.TOXMOL_FP_RADIUS, include_physchem=True
        )
        split = prepare_endpoint(
            df, target, X_all, valid_mask, n_physchem=n_phys, scale_physchem=False
        )
        if split is None:
            return None
        fp_dim = split.X_train.shape[1] - n_phys
        scaler = StandardScaler()
        scaler.fit(split.X_train[:, fp_dim:])
        _SCALERS[target] = scaler
        return scaler
    except Exception as exc:
        logger.warning("Scaler fit failed for %s: %s", target, exc)
        return None


def _model_n_features(model: Any) -> Optional[int]:
    for attr in ("n_features_in_", "n_features_"):
        if hasattr(model, attr):
            try:
                return int(getattr(model, attr))
            except Exception:
                pass
    # FLAML AutoML wrapper
    est = getattr(model, "model", None) or getattr(model, "best_model", None)
    if est is not None and est is not model:
        return _model_n_features(est)
    # nested estimator
    for name in ("estimator", "_estimator", "learner"):
        inner = getattr(model, name, None)
        if inner is not None and inner is not model:
            n = _model_n_features(inner)
            if n:
                return n
    return None


def _featurize_classical(smiles: str, target: str, model: Any) -> np.ndarray:
    fp = morgan_fingerprint(smiles, n_bits=settings.TOXMOL_N_BITS, radius=settings.TOXMOL_FP_RADIUS)
    if fp is None:
        raise ValueError("Invalid SMILES")
    n_feat = _model_n_features(model)
    # Default: research classical artifacts are Morgan-2048; physchem only if model expects it.
    if n_feat is None or n_feat <= settings.TOXMOL_N_BITS:
        return fp.reshape(1, -1).astype(np.float32)

    desc = compute_descriptors(smiles)
    phys = np.array([[float(desc[c]) for c in PHYSCHEM_COLS]], dtype=np.float32)
    phys = np.nan_to_num(phys, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = _fit_scaler_for_target(target)
    if scaler is not None:
        phys = scaler.transform(phys).astype(np.float32)
    X = np.hstack([fp.reshape(1, -1), phys]).astype(np.float32)
    if X.shape[1] != n_feat:
        if X.shape[1] > n_feat:
            X = X[:, :n_feat]
        else:
            pad = np.zeros((1, n_feat - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
    return X


def _load_classical(path: Path) -> Tuple[Any, float]:
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        meta = obj.get("metadata") or {}
        thr = float(meta.get("threshold", 0.5))
        return model, thr
    return obj, 0.5


def _load_lightning(path: Path, model_type: str) -> Tuple[Any, float]:
    import torch
    from src.nn_models import build_model

    payload = torch.load(path, map_location="cpu")
    hparams = dict(payload.get("hparams") or {})
    meta = payload.get("metadata") or {}
    thr = float(meta.get("threshold", 0.5))
    model = build_model(model_type, **hparams)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, thr


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_artifact(spec: ModelSpec) -> Tuple[Path, str, str, float, bool, Optional[str]]:
    """Return path, kind, model_name, roc_auc, is_fallback, lightning_type."""
    mdir = models_dir()
    preferred = mdir / spec.artifact
    can_use_preferred = preferred.exists()
    if can_use_preferred and spec.preferred_kind == "lightning" and not _torch_available():
        logger.info("Skipping %s: torch is not installed, using classical fallback", spec.artifact)
        can_use_preferred = False
    if can_use_preferred:
        lt = None
        if spec.preferred_kind == "lightning":
            lt = spec.artifact.split("_")[1]
        return preferred, spec.preferred_kind, spec.preferred_name, spec.roc_auc, False, lt

    if spec.fallback_artifact:
        fb = mdir / spec.fallback_artifact
        if fb.exists():
            name = f"FLAML:{spec.fallback_estimator}"
            return (
                fb,
                "classical",
                name,
                float(spec.fallback_roc_auc or 0.0),
                True,
                None,
            )

    for p in sorted(mdir.glob(f"flaml_{spec.target}_*.joblib")):
        est = p.stem.split("_")[-1]
        return p, "classical", f"FLAML:{est}", float(spec.fallback_roc_auc or spec.roc_auc), True, None

    raise FileNotFoundError(f"No model artifact for {spec.target}")


def ensure_loaded(target: str) -> LoadedModel:
    with _lock:
        if target in _LOADED:
            return _LOADED[target]
        spec = SPECS_BY_TARGET[target]
        path, kind, name, roc, fb, lt = _resolve_artifact(spec)
        if kind == "classical":
            model, thr = _load_classical(path)
        else:
            model, thr = _load_lightning(path, lt or "gnn")
        loaded = LoadedModel(
            target=target,
            family=spec.family,
            model_name=name,
            kind=kind,
            artifact=path.name,
            roc_auc=roc,
            fallback=fb,
            model=model,
            threshold=thr,
            lightning_type=lt,
        )
        _LOADED[target] = loaded
        _STATUS["entries"][target] = {
            "model_name": name,
            "artifact": path.name,
            "kind": kind,
            "fallback": fb,
            "roc_auc": roc,
        }
        _STATUS["ready"] = len(_STATUS["entries"]) > 0
        return loaded


def model_status() -> Dict[str, Any]:
    mdir = models_dir()
    entries = {}
    for spec in MODEL_SPECS:
        preferred = (mdir / spec.artifact).exists()
        fallback = bool(spec.fallback_artifact and (mdir / spec.fallback_artifact).exists())
        any_classical = list(mdir.glob(f"flaml_{spec.target}_*.joblib"))
        entries[spec.target] = {
            "preferred": preferred,
            "fallback_available": fallback or bool(any_classical),
            "preferred_artifact": spec.artifact,
            "loaded": spec.target in _LOADED,
        }
    return {"models_dir": str(mdir), "targets": entries, **_STATUS}


def _predict_lightning(loaded: LoadedModel, smiles: str) -> float:
    import torch
    from src.nn_data import (
        DEFAULT_MAX_NODES,
        collate_graphs,
        morgan_fingerprint as nn_morgan,
        smiles_to_graph,
    )

    model = loaded.model
    lt = loaded.lightning_type or "gnn"
    with torch.no_grad():
        if lt in ("mlp", "resnet"):
            fp = nn_morgan(
                smiles, n_bits=settings.TOXMOL_N_BITS, radius=settings.TOXMOL_FP_RADIUS
            )
            if fp is None:
                raise ValueError("Invalid SMILES")
            x = torch.from_numpy(fp).unsqueeze(0)
            logits = model(x)
            return float(torch.sigmoid(logits).view(-1)[0].item())
        graph = smiles_to_graph(smiles, max_nodes=DEFAULT_MAX_NODES)
        if graph is None:
            raise ValueError("Invalid SMILES")
        feats, adj, mask = graph
        batch = collate_graphs(
            [
                {
                    "node_feats": torch.from_numpy(feats),
                    "adj": torch.from_numpy(adj),
                    "mask": torch.from_numpy(mask.astype(bool)),
                    "y": torch.tensor(0.0),
                }
            ],
            max_nodes=DEFAULT_MAX_NODES,
        )
        logits, _ = model._forward_batch(batch)
        return float(torch.sigmoid(logits).view(-1)[0].item())


def predict_target(smiles: str, target: str) -> Dict[str, Any]:
    loaded = ensure_loaded(target)
    if loaded.kind == "classical":
        X = _featurize_classical(smiles, target, loaded.model)
        prob = float(predict_proba_binary(loaded.model, X)[0])
    else:
        prob = _predict_lightning(loaded, smiles)
    label = int(prob >= loaded.threshold)
    return {
        "target": target,
        "family": loaded.family,
        "probability": prob,
        "label": label,
        "threshold": loaded.threshold,
        "model_name": loaded.model_name,
        "roc_auc": loaded.roc_auc,
        "artifact": loaded.artifact,
        "fallback": loaded.fallback,
    }


def predict_all(smiles: str) -> Dict[str, Any]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    parsed = parse_smiles(smiles)
    results: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    for target in TOX21_TARGETS:
        try:
            results.append(predict_target(parsed["canonical_smiles"], target))
        except Exception as exc:
            logger.exception("Predict failed for %s", target)
            errors[target] = str(exc)

    nr = [r for r in results if r["family"] == "NR"]
    sr = [r for r in results if r["family"] == "SR"]
    probs = [r["probability"] for r in results]
    mean_p = float(np.mean(probs)) if probs else 0.0
    max_p = float(np.max(probs)) if probs else 0.0
    active = [r for r in results if r["label"] == 1]

    physchem = parsed["physchem"]
    lip = lipinski_veber(physchem)

    return {
        "smiles": parsed["canonical_smiles"],
        "svg": parsed["svg"],
        "qsar": {
            "mean_probability": mean_p,
            "max_probability": max_p,
            "n_active_endpoints": len(active),
            "n_endpoints": len(results),
            "risk_level": "high" if max_p >= 0.7 else ("medium" if max_p >= 0.4 else "low"),
            "active_targets": [r["target"] for r in active],
        },
        "admet": {
            "physchem": physchem,
            "lipinski": lip,
            "toxicity_profile": {
                "mean_probability": mean_p,
                "max_probability": max_p,
                "n_active": len(active),
            },
        },
        "nr": nr,
        "sr": sr,
        "endpoints": {r["target"]: r for r in results},
        "errors": errors,
        "meta": model_status(),
    }
