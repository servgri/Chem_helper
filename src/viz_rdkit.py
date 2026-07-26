"""RDKit molecule depiction helpers for Tox21 notebooks."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

PathLike = Union[str, Path]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _ensure_results_dir(path: Optional[PathLike] = None) -> Path:
    out = Path(path) if path is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_pil_image(img: Any) -> PILImage.Image:
    """Normalize RDKit draw output to a PIL Image.

    In Jupyter, ``rdkit.Chem.Draw.IPythonConsole`` patches ``MolsToGridImage`` to
    ``ShowMols``, which returns ``IPython.core.display.Image`` (no ``.save``).
    """
    if hasattr(img, "save") and callable(img.save):
        return img  # type: ignore[return-value]
    data = getattr(img, "data", None)
    if data is not None:
        return PILImage.open(io.BytesIO(data))
    raise TypeError(f"Unsupported image type from RDKit draw: {type(img)!r}")


def smiles_to_mols(
    smiles: Sequence[str],
    legends: Optional[Sequence[str]] = None,
) -> tuple[List[Chem.Mol], List[str], List[int]]:
    """Parse SMILES; return (valid mols, legends for those mols, original indices)."""
    mols: List[Chem.Mol] = []
    kept_legends: List[str] = []
    indices: List[int] = []
    for i, smi in enumerate(smiles):
        if not isinstance(smi, str) or not smi.strip():
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        if legends is not None and i < len(legends):
            kept_legends.append(str(legends[i]))
        else:
            kept_legends.append(smi if len(smi) <= 28 else smi[:25] + "...")
        indices.append(i)
    return mols, kept_legends, indices


def draw_smiles_grid(
    smiles: Sequence[str],
    legends: Optional[Sequence[str]] = None,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (200, 200),
    max_mols: int = 16,
):
    """Draw a grid image from SMILES strings (always returns a PIL Image)."""
    mols, kept_legends, _ = smiles_to_mols(smiles[:max_mols], legends[:max_mols] if legends else None)
    if not mols:
        raise ValueError("No valid molecules to draw from the provided SMILES.")
    # returnPNG=False: avoid IPython.display.Image when IPythonConsole patched Draw
    raw = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=kept_legends,
        useSVG=False,
        returnPNG=False,
    )
    return _to_pil_image(raw)



def save_smiles_grid_png(
    smiles: Sequence[str],
    filename: str,
    legends: Optional[Sequence[str]] = None,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (200, 200),
    max_mols: int = 16,
    results_dir: Optional[PathLike] = None,
) -> Path:
    """Render a SMILES grid and save PNG under ``results/`` (or ``results_dir``)."""
    out_dir = _ensure_results_dir(results_dir)
    img = draw_smiles_grid(
        smiles,
        legends=legends,
        mols_per_row=mols_per_row,
        sub_img_size=sub_img_size,
        max_mols=max_mols,
    )
    path = out_dir / filename
    if not str(path).lower().endswith(".png"):
        path = path.with_suffix(".png")
    img.save(str(path))
    return path


def draw_active_vs_inactive(
    df: pd.DataFrame,
    target: str,
    smiles_col: str = "smiles",
    n_each: int = 6,
    mols_per_row: int = 3,
    sub_img_size: tuple[int, int] = (200, 200),
    random_state: int = 42,
):
    """Sample active and inactive molecules for an endpoint and draw a labeled grid.

    Layout: actives first (legends prefixed with ``A``), then inactives (``I``).
    """
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not in dataframe.")
    if smiles_col not in df.columns:
        raise KeyError(f"SMILES column {smiles_col!r} not in dataframe.")

    sub = df[[smiles_col, target]].dropna()
    act = sub[sub[target] == 1][smiles_col].tolist()
    ina = sub[sub[target] == 0][smiles_col].tolist()

    rng = np.random.default_rng(random_state)

    def _sample(pool: List[str], k: int) -> List[str]:
        if not pool:
            return []
        k = min(k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in idx]

    act_s = _sample(act, n_each)
    ina_s = _sample(ina, n_each)
    smiles = act_s + ina_s
    legends = [f"A:{target}" for _ in act_s] + [f"I:{target}" for _ in ina_s]
    if not smiles:
        raise ValueError(f"No labeled molecules available for endpoint {target!r}.")
    return draw_smiles_grid(
        smiles,
        legends=legends,
        mols_per_row=mols_per_row,
        sub_img_size=sub_img_size,
        max_mols=len(smiles),
    )


def save_active_vs_inactive_png(
    df: pd.DataFrame,
    target: str,
    filename: Optional[str] = None,
    smiles_col: str = "smiles",
    n_each: int = 6,
    mols_per_row: int = 3,
    sub_img_size: tuple[int, int] = (200, 200),
    random_state: int = 42,
    results_dir: Optional[PathLike] = None,
) -> Path:
    """Save active-vs-inactive grid PNG for ``target`` into ``results/``."""
    out_dir = _ensure_results_dir(results_dir)
    safe = target.replace("/", "_").replace(" ", "_")
    path = out_dir / (filename or f"{safe}_active_vs_inactive.png")
    img = draw_active_vs_inactive(
        df,
        target=target,
        smiles_col=smiles_col,
        n_each=n_each,
        mols_per_row=mols_per_row,
        sub_img_size=sub_img_size,
        random_state=random_state,
    )
    img.save(str(path))
    return path


def draw_single_mol_png(
    smiles: str,
    filename: str,
    size: tuple[int, int] = (300, 300),
    results_dir: Optional[PathLike] = None,
) -> Path:
    """Draw one molecule to a PNG under ``results/``."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    out_dir = _ensure_results_dir(results_dir)
    path = out_dir / filename
    if not str(path).lower().endswith(".png"):
        path = path.with_suffix(".png")
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    with open(path, "wb") as f:
        f.write(drawer.GetDrawingText())
    return path


def draw_pipeline_banner(
    smiles: Optional[Sequence[str]] = None,
    results_dir: Optional[PathLike] = None,
    filename: str = "rdkit_pipeline_banner.png",
    max_mols: int = 8,
):
    """Draw a small SMILES strip used as a visual banner for the RDKit pipeline section."""
    out_dir = _ensure_results_dir(results_dir)
    if smiles is None:
        smiles = [
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
            "CCO",
            "c1ccccc1",
            "CC(=O)Nc1ccc(O)cc1",  # acetaminophen
            "CN1CCC[C@H]1c2cccnc2",  # nicotine
            "CC(C)NCC(O)c1ccc(O)c(CO)c1",  # salbutamol-ish
        ]
    img = draw_smiles_grid(list(smiles)[:max_mols], mols_per_row=min(4, max_mols), max_mols=max_mols)
    path = out_dir / filename
    img.save(str(path))
    return img, path


def draw_endpoint_examples(
    df: pd.DataFrame,
    targets: Sequence[str] = ("NR-AhR", "SR-MMP"),
    n_each: int = 4,
    results_dir: Optional[PathLike] = None,
) -> dict:
    """Draw and save active-vs-inactive grids for selected Tox21 endpoints."""
    out_dir = _ensure_results_dir(results_dir)
    artifacts: dict = {}
    for target in targets:
        try:
            img = draw_active_vs_inactive(df, target=target, n_each=n_each)
            safe = target.replace("/", "_").replace(" ", "_")
            path = out_dir / f"{safe}_examples.png"
            img.save(str(path))
            artifacts[target] = {"image": img, "path": path}
        except Exception as exc:  # noqa: BLE001 — notebook-friendly soft fail
            artifacts[target] = {"error": str(exc)}
    return artifacts


def show_rdkit_pipeline_visuals(
    df: pd.DataFrame,
    results_dir: Optional[PathLike] = None,
    targets: Sequence[str] = ("NR-AhR", "SR-MMP"),
    n_each: int = 4,
) -> dict:
    """End-to-end RDKit visual line: banner + endpoint example grids.

    Returns a dict with banner/examples artifacts (images + paths) for notebook display.
    """
    out_dir = _ensure_results_dir(results_dir)
    sample = df["smiles"].dropna().astype(str).head(8).tolist() if "smiles" in df.columns else None
    banner_img, banner_path = draw_pipeline_banner(smiles=sample, results_dir=out_dir)
    examples = draw_endpoint_examples(df, targets=targets, n_each=n_each, results_dir=out_dir)
    return {
        "banner": {"image": banner_img, "path": banner_path},
        "examples": examples,
        "results_dir": out_dir,
    }
