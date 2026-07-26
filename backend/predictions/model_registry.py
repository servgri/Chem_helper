"""Top Tox21 models from research metrics (test ROC-AUC winners)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    target: str
    family: str  # NR | SR
    preferred_kind: str  # classical | lightning
    preferred_name: str  # e.g. FLAML:lgbm or Lightning_gnn
    artifact: str
    roc_auc: float
    # Classical fallback when Lightning artifact is missing
    fallback_estimator: Optional[str] = None
    fallback_artifact: Optional[str] = None
    fallback_roc_auc: Optional[float] = None


# Winners by test_roc_auc from results/tox21_model_metrics.csv
MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("NR-AR", "NR", "classical", "FLAML:xgboost", "flaml_NR-AR_xgboost.joblib", 0.754),
    ModelSpec(
        "NR-AR-LBD",
        "NR",
        "lightning",
        "Lightning_gnn",
        "lightning_gnn_NR-AR-LBD.pt",
        0.831,
        fallback_estimator="xgboost",
        fallback_artifact="flaml_NR-AR-LBD_xgboost.joblib",
        fallback_roc_auc=0.806,
    ),
    ModelSpec("NR-AhR", "NR", "classical", "FLAML:lgbm", "flaml_NR-AhR_lgbm.joblib", 0.888),
    ModelSpec(
        "NR-Aromatase", "NR", "classical", "FLAML:lgbm", "flaml_NR-Aromatase_lgbm.joblib", 0.859
    ),
    ModelSpec("NR-ER", "NR", "classical", "FLAML:lgbm", "flaml_NR-ER_lgbm.joblib", 0.716),
    ModelSpec(
        "NR-ER-LBD",
        "NR",
        "lightning",
        "Lightning_gnn",
        "lightning_gnn_NR-ER-LBD.pt",
        0.771,
        fallback_estimator="lgbm",
        fallback_artifact="flaml_NR-ER-LBD_lgbm.joblib",
        fallback_roc_auc=0.759,
    ),
    ModelSpec(
        "NR-PPAR-gamma", "NR", "classical", "FLAML:lrl1", "flaml_NR-PPAR-gamma_lrl1.joblib", 0.785
    ),
    ModelSpec("SR-ARE", "SR", "classical", "FLAML:lgbm", "flaml_SR-ARE_lgbm.joblib", 0.819),
    ModelSpec(
        "SR-ATAD5",
        "SR",
        "lightning",
        "Lightning_resnet",
        "lightning_resnet_SR-ATAD5.pt",
        0.855,
        fallback_estimator="lrl1",
        fallback_artifact="flaml_SR-ATAD5_lrl1.joblib",
        fallback_roc_auc=0.854,
    ),
    ModelSpec("SR-HSE", "SR", "classical", "FLAML:lgbm", "flaml_SR-HSE_lgbm.joblib", 0.755),
    ModelSpec(
        "SR-MMP",
        "SR",
        "lightning",
        "Lightning_gnn",
        "lightning_gnn_SR-MMP.pt",
        0.870,
        fallback_estimator="lrl1",
        fallback_artifact="flaml_SR-MMP_lrl1.joblib",
        fallback_roc_auc=0.857,
    ),
    ModelSpec("SR-p53", "SR", "classical", "FLAML:lgbm", "flaml_SR-p53_lgbm.joblib", 0.835),
]

SPECS_BY_TARGET: Dict[str, ModelSpec] = {s.target: s for s in MODEL_SPECS}
