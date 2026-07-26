"""Tanimoto similarity against Tox21 fingerprint index."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from django.conf import settings

from molecules.services import mol_to_svg, molecule_meta
from src.qsar_utils import morgan_fingerprint, smiles_to_mol

_lock = threading.Lock()
_INDEX: Optional[Dict[str, Any]] = None


def _load_index() -> Dict[str, Any]:
    global _INDEX
    with _lock:
        if _INDEX is not None:
            return _INDEX
        path = Path(settings.TOXMOL_FP_INDEX_PATH)
        if not path.exists():
            from predictions.management.commands.prepare_models import build_fp_index

            build_fp_index()
        data = np.load(path, allow_pickle=True)
        _INDEX = {
            "fps": data["fps"].astype(np.float32),
            "smiles": data["smiles"],
            "mol_ids": data["mol_ids"] if "mol_ids" in data.files else None,
        }
        return _INDEX


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    inter = float(np.minimum(a, b).sum())
    union = float(np.maximum(a, b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def find_similar(smiles: str, top_n: Optional[int] = None) -> Dict[str, Any]:
    top_n = int(top_n or settings.TOXMOL_SIMILAR_TOP_N)
    query = morgan_fingerprint(
        smiles, n_bits=settings.TOXMOL_N_BITS, radius=settings.TOXMOL_FP_RADIUS
    )
    if query is None:
        raise ValueError("Invalid SMILES")

    idx = _load_index()
    fps = idx["fps"]
    # Vectorized Tanimoto for binary-like floats
    inter = fps @ query
    union = fps.sum(axis=1) + float(query.sum()) - inter
    sims = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    # Exclude exact self if present
    order = np.argsort(-sims)
    results: List[Dict[str, Any]] = []
    for i in order:
        smi = str(idx["smiles"][i])
        if smi == smiles:
            continue
        mol = smiles_to_mol(smi)
        svg = mol_to_svg(mol) if mol is not None else ""
        mol_id = None
        if idx["mol_ids"] is not None:
            mol_id = str(idx["mol_ids"][i])
        meta = molecule_meta(smi, lookup=True)
        results.append(
            {
                "smiles": smi,
                "mol_id": mol_id,
                "name": meta["name"],
                "formula": meta["formula"],
                "tanimoto": float(sims[i]),
                "svg": svg,
            }
        )
        if len(results) >= top_n:
            break

    return {"smiles": smiles, "similar": results, "top_n": top_n}
