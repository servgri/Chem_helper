"""QSAR / Tox21 helpers for the research notebook."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# MoleculeNet / DeepChem canonical dump (~7.8k compounds, 12 binary assays)
TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
DEFAULT_TOX21_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "tox21.csv.gz"

TOX21_TARGETS: List[str] = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

NR_TARGETS = [t for t in TOX21_TARGETS if t.startswith("NR-")]
SR_TARGETS = [t for t in TOX21_TARGETS if t.startswith("SR-")]

PHYSCHEM_COLS: List[str] = [
    "MW",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
    "Rings",
    "HeavyAtoms",
]


@dataclass
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    smiles_train: np.ndarray
    smiles_test: np.ndarray
    n_physchem: int = 0


def download_tox21(
    dest: Optional[PathLike] = None,
    *,
    force: bool = False,
    url: str = TOX21_URL,
) -> Path:
    """Download MoleculeNet Tox21 CSV (gzipped) into ``data/raw/`` if missing.

    Returns the local path to ``tox21.csv.gz``.
    """
    path = Path(dest) if dest is not None else DEFAULT_TOX21_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path

    tmp = path.with_suffix(path.suffix + ".part")
    logger.info("Downloading Tox21 from %s -> %s", url, path)
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    logger.info("Tox21 saved (%.2f MB)", path.stat().st_size / 1e6)
    return path


def load_tox21(
    path: Optional[PathLike] = None,
    *,
    download: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load Tox21 as a DataFrame; optionally download MoleculeNet dump first."""
    csv_path = Path(path) if path is not None else DEFAULT_TOX21_PATH
    if force_download or not csv_path.exists():
        if not download and not csv_path.exists():
            raise FileNotFoundError(
                f"Tox21 file not found: {csv_path}. "
                "Pass download=True or call download_tox21()."
            )
        download_tox21(csv_path, force=force_download)

    df = pd.read_csv(csv_path)
    missing = [c for c in TOX21_TARGETS + ["smiles"] if c not in df.columns]
    if missing:
        raise ValueError(f"Tox21 file missing columns: {missing}")
    return df


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def compute_descriptors(smiles: str) -> Dict[str, float]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {
            "MW": np.nan,
            "LogP": np.nan,
            "TPSA": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "RotBonds": np.nan,
            "Rings": np.nan,
            "HeavyAtoms": np.nan,
            "valid_mol": 0,
        }
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtoms": Lipinski.HeavyAtomCount(mol),
        "valid_mol": 1,
    }


def build_descriptor_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [compute_descriptors(s) for s in df["smiles"].tolist()]
    desc = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), desc], axis=1)


def morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> Optional[np.ndarray]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_smiles(
    smiles_list: List[str],
    n_bits: int = 2048,
    radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    vectors: List[np.ndarray] = []
    valid_mask: List[bool] = []
    for smi in smiles_list:
        fp = morgan_fingerprint(smi, n_bits=n_bits, radius=radius)
        if fp is None:
            valid_mask.append(False)
            vectors.append(np.zeros((n_bits,), dtype=np.float32))
        else:
            valid_mask.append(True)
            vectors.append(fp)
    return np.vstack(vectors), np.asarray(valid_mask, dtype=bool)


def build_fp_physchem_matrix(
    df: pd.DataFrame,
    *,
    n_bits: int = 2048,
    radius: int = 2,
    include_physchem: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Morgan FP (+ optional physchem). Returns ``(X, valid_mask, n_physchem)``."""
    X_fp, valid_mask = featurize_smiles(df["smiles"].tolist(), n_bits=n_bits, radius=radius)
    if not include_physchem:
        return X_fp, valid_mask, 0

    desc = build_descriptor_frame(df)
    phys = desc[PHYSCHEM_COLS].to_numpy(dtype=np.float32)
    # Invalid mols already have NaN descriptors; zero-fill (masked out later).
    phys = np.nan_to_num(phys, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.hstack([X_fp, phys]).astype(np.float32)
    return X, valid_mask, len(PHYSCHEM_COLS)


def prepare_endpoint(
    df: pd.DataFrame,
    target: str,
    X_all: np.ndarray,
    valid_mask: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_physchem: int = 0,
    scale_physchem: bool = True,
) -> Optional[SplitData]:
    y_raw = df[target].to_numpy()
    mask = valid_mask & ~pd.isna(y_raw)
    if mask.sum() < 50:
        return None
    X = X_all[mask]
    y = y_raw[mask].astype(int)
    smiles = df.loc[mask, "smiles"].to_numpy()
    if len(np.unique(y)) < 2:
        return None
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X,
        y,
        smiles,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    if n_physchem > 0 and scale_physchem:
        scaler = StandardScaler()
        fp_dim = X_train.shape[1] - n_physchem
        X_train = np.hstack(
            [
                X_train[:, :fp_dim],
                scaler.fit_transform(X_train[:, fp_dim:]).astype(np.float32),
            ]
        ).astype(np.float32)
        X_test = np.hstack(
            [
                X_test[:, :fp_dim],
                scaler.transform(X_test[:, fp_dim:]).astype(np.float32),
            ]
        ).astype(np.float32)
    return SplitData(
        X_train,
        X_test,
        y_train,
        y_test,
        smiles_train,
        smiles_test,
        n_physchem=n_physchem,
    )


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Pick decision threshold that maximises F1 on a calibration set."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, 0.0
    denom = np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    f1 = 2.0 * precision[:-1] * recall[:-1] / denom
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def class_imbalance_weight(y: np.ndarray) -> float:
    """``n_neg / n_pos`` for sample_weight / scale_pos_weight / BCE pos_weight."""
    y = np.asarray(y).astype(int).ravel()
    n_pos = max(float((y == 1).sum()), 1.0)
    n_neg = max(float((y == 0).sum()), 1.0)
    return float(n_neg / n_pos)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if threshold is not None:
        out["threshold"] = float(threshold)
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return out


def predict_proba_binary(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def run_flaml(
    split: SplitData,
    time_budget: int = 60,
    metric: str = "ap",
    estimator_list: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    from src.flaml_train import run_classical_flaml

    return run_classical_flaml(
        split,
        time_budget=time_budget,
        metric=metric,
        estimator_list=estimator_list,
        **kwargs,
    )


def results_to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        item = {
            "target": r["target"],
            "family": r.get("family", ""),
            "model_name": r["model_name"],
            "source": r.get("source", "flaml"),
            "best_cv_score": r.get("best_cv_score", r.get("best_val_score", np.nan)),
        }
        item.update({f"test_{k}": v for k, v in r["metrics"].items()})
        flat.append(item)
    return pd.DataFrame(flat)
