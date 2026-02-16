"""Ensure Tox21 fingerprint index and classical winner models exist."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from predictions.model_registry import MODEL_SPECS, ModelSpec
from src.qsar_utils import (
    build_fp_physchem_matrix,
    download_tox21,
    load_tox21,
    morgan_fingerprint,
    prepare_endpoint,
)

logger = logging.getLogger(__name__)


def build_fp_index(force: bool = False) -> Path:
    out = Path(settings.TOXMOL_FP_INDEX_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return out

    download_tox21()
    df = load_tox21()
    smiles_list: List[str] = []
    fps: List[np.ndarray] = []
    mol_ids: List[str] = []
    for _, row in df.iterrows():
        smi = str(row["smiles"])
        fp = morgan_fingerprint(smi, n_bits=settings.TOXMOL_N_BITS, radius=settings.TOXMOL_FP_RADIUS)
        if fp is None:
            continue
        smiles_list.append(smi)
        fps.append(fp)
        mol_ids.append(str(row.get("mol_id", "")))
    X = np.vstack(fps).astype(np.float32)
    np.savez_compressed(
        out,
        fps=X,
        smiles=np.array(smiles_list, dtype=object),
        mol_ids=np.array(mol_ids, dtype=object),
    )
    return out


def _estimator_from_spec(spec: ModelSpec) -> Optional[str]:
    if spec.preferred_kind == "classical" and ":" in spec.preferred_name:
        return spec.preferred_name.split(":", 1)[1]
    return spec.fallback_estimator


def _needed_classical_artifacts(spec: ModelSpec, models_dir: Path) -> List[tuple[str, str]]:
    """Return list of (estimator, filename) to train if missing."""
    needed: List[tuple[str, str]] = []
    if spec.preferred_kind == "classical":
        path = models_dir / spec.artifact
        if not path.exists():
            est = _estimator_from_spec(spec)
            if est:
                needed.append((est, spec.artifact))
    else:
        # Lightning preferred: train fallback classical if both preferred and fallback missing
        preferred = models_dir / spec.artifact
        if preferred.exists():
            return []
        fb_name = spec.fallback_artifact
        fb_est = spec.fallback_estimator
        if fb_name and fb_est and not (models_dir / fb_name).exists():
            needed.append((fb_est, fb_name))
        elif not fb_name:
            # generic lgbm fallback
            generic = f"flaml_{spec.target}_lgbm.joblib"
            if not (models_dir / generic).exists():
                needed.append(("lgbm", generic))
    return needed


def _build_estimator(name: str):
    name = (name or "lgbm").lower()
    if name in ("lgbm", "lightgbm"):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    if name in ("xgboost", "xgb"):
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    if name in ("lrl1", "lr", "logistic"):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.5,
            max_iter=400,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )


def train_classical(
    target: str,
    estimator: str,
    filename: str,
    time_budget: int = 60,
    fast: bool = True,
) -> Path:
    import joblib
    import shutil

    from src.qsar_utils import best_f1_threshold, predict_proba_binary

    models_dir = Path(settings.TOXMOL_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / filename

    download_tox21()
    df = load_tox21()
    X_all, valid_mask, n_phys = build_fp_physchem_matrix(
        df,
        n_bits=settings.TOXMOL_N_BITS,
        radius=settings.TOXMOL_FP_RADIUS,
        include_physchem=True,
    )
    split = prepare_endpoint(df, target, X_all, valid_mask, n_physchem=n_phys)
    if split is None:
        raise RuntimeError(f"Cannot prepare split for {target}")

    if fast:
        est_name = estimator if estimator != "lrl1" else "lrl1"
        model = _build_estimator(est_name)
        try:
            model.fit(split.X_train, split.y_train)
        except Exception:
            model = _build_estimator("lgbm")
            est_name = "lgbm"
            model.fit(split.X_train, split.y_train)
        y_prob_train = predict_proba_binary(model, split.X_train)
        thr, _ = best_f1_threshold(split.y_train, y_prob_train)
        joblib.dump(
            {
                "model": model,
                "metadata": {
                    "target": target,
                    "best_estimator": est_name,
                    "threshold": thr,
                    "fast": True,
                },
            },
            out_path,
        )
        return out_path

    from src.flaml_train import run_classical_flaml

    estimators = [estimator]
    if estimator == "lrl1":
        estimators = ["lrl1", "lgbm", "xgboost"]

    result = run_classical_flaml(
        split,
        time_budget=time_budget,
        metric="ap",
        estimator_list=estimators,
        target=target,
        results_dir=str(Path(settings.TOXMOL_REPO_ROOT) / "results"),
        use_clearml=False,
        verbose=0,
    )

    produced = result.get("clearml_model_path")
    if produced and Path(produced).exists():
        src = Path(produced)
        if src.resolve() != out_path.resolve():
            shutil.copy2(src, out_path)
        return out_path

    automl = result["estimator"]
    thr = result.get("threshold", 0.5)
    joblib.dump(
        {
            "model": automl,
            "metadata": {
                "target": target,
                "best_estimator": result.get("best_estimator"),
                "threshold": thr,
                "metrics": result.get("metrics"),
            },
        },
        out_path,
    )
    return out_path


class Command(BaseCommand):
    help = "Build Tox21 FP index and train missing classical winner/fallback models."

    def add_arguments(self, parser):
        parser.add_argument("--force-index", action="store_true")
        parser.add_argument("--time-budget", type=int, default=60)
        parser.add_argument("--skip-train", action="store_true")
        parser.add_argument(
            "--fast",
            action="store_true",
            default=True,
            help="Train lightweight sklearn/LGBM models (default, fast bootstrap).",
        )
        parser.add_argument(
            "--flaml",
            action="store_true",
            help="Use FLAML AutoML instead of --fast trainers.",
        )

    def handle(self, *args, **options):
        self.stdout.write("Building fingerprint index…")
        idx = build_fp_index(force=options["force_index"])
        self.stdout.write(self.style.SUCCESS(f"FP index: {idx}"))

        if options["skip_train"]:
            self.stdout.write("Skipping model training (--skip-train).")
            return

        models_dir = Path(settings.TOXMOL_MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        time_budget = int(options["time_budget"])
        fast = not bool(options.get("flaml"))

        for spec in MODEL_SPECS:
            needed = _needed_classical_artifacts(spec, models_dir)
            if not needed:
                preferred = models_dir / spec.artifact
                if preferred.exists():
                    self.stdout.write(f"[ok] {spec.target}: {spec.artifact}")
                elif spec.fallback_artifact and (models_dir / spec.fallback_artifact).exists():
                    self.stdout.write(
                        self.style.WARNING(
                            f"[fallback] {spec.target}: using {spec.fallback_artifact}"
                        )
                    )
                else:
                    alts = list(models_dir.glob(f"flaml_{spec.target}_*.joblib"))
                    if alts:
                        self.stdout.write(
                            self.style.WARNING(f"[alt] {spec.target}: {alts[0].name}")
                        )
                    else:
                        self.stdout.write(self.style.ERROR(f"[missing] {spec.target}"))
                continue

            for est, filename in needed:
                mode = "fast" if fast else "flaml"
                self.stdout.write(f"Training [{mode}] {spec.target} ({est}) → {filename} …")
                try:
                    path = train_classical(
                        spec.target,
                        est,
                        filename,
                        time_budget=time_budget,
                        fast=fast,
                    )
                    self.stdout.write(self.style.SUCCESS(f"Saved {path}"))
                except Exception as exc:
                    self.stdout.write(self.style.ERROR(f"Failed {spec.target}/{est}: {exc}"))
                    logger.exception("train failed")
                    if est != "lgbm":
                        alt = f"flaml_{spec.target}_lgbm.joblib"
                        try:
                            path = train_classical(
                                spec.target,
                                "lgbm",
                                alt,
                                time_budget=time_budget,
                                fast=fast,
                            )
                            self.stdout.write(self.style.WARNING(f"Saved fallback {path}"))
                        except Exception as exc2:
                            self.stdout.write(self.style.ERROR(f"lgbm fallback failed: {exc2}"))
