"""Molecule parse / draw / physchem helpers."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from src.qsar_utils import PHYSCHEM_COLS, compute_descriptors, smiles_to_mol

logger = logging.getLogger(__name__)

_NAME_CACHE: Dict[str, str] = {}


def mol_to_svg(mol: Chem.Mol, width: int = 320, height: int = 240) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_molblock(mol: Chem.Mol) -> str:
    return Chem.MolToMolBlock(mol)


def molecular_formula(smiles: str) -> str:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
    try:
        return str(rdMolDescriptors.CalcMolFormula(mol))
    except Exception:
        return ""


def molecule_name(smiles: str, *, lookup: bool = True) -> str:
    """Prefer PubChem Title / IUPAC; fall back to molecular formula."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return smiles.strip() or "—"
    canonical = Chem.MolToSmiles(mol)
    if canonical in _NAME_CACHE:
        return _NAME_CACHE[canonical]

    formula = molecular_formula(canonical) or canonical
    name = formula
    if lookup:
        try:
            q = urllib.parse.quote(canonical, safe="")
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                f"{q}/property/Title,IUPACName/JSON"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "ToxMolAI/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            props = (data.get("PropertyTable") or {}).get("Properties") or []
            if props:
                row = props[0]
                name = (
                    (row.get("Title") or "").strip()
                    or (row.get("IUPACName") or "").strip()
                    or formula
                )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError) as exc:
            logger.debug("PubChem name lookup failed for %s: %s", canonical, exc)
            name = formula
        except Exception as exc:  # noqa: BLE001
            logger.debug("PubChem name lookup error: %s", exc)
            name = formula

    _NAME_CACHE[canonical] = name
    return name


def molecule_meta(smiles: str, *, lookup: bool = True) -> Dict[str, str]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {"name": smiles or "—", "formula": "", "smiles": smiles or ""}
    canonical = Chem.MolToSmiles(mol)
    formula = molecular_formula(canonical)
    return {
        "name": molecule_name(canonical, lookup=lookup),
        "formula": formula,
        "smiles": canonical,
    }


def lipinski_veber(physchem: Dict[str, float]) -> Dict[str, Any]:
    mw = float(physchem.get("MW", 0))
    logp = float(physchem.get("LogP", 0))
    hbd = float(physchem.get("HBD", 0))
    hba = float(physchem.get("HBA", 0))
    tpsa = float(physchem.get("TPSA", 0))
    rot = float(physchem.get("RotBonds", 0))

    rules = {
        "mw_le_500": mw <= 500,
        "logp_le_5": logp <= 5,
        "hbd_le_5": hbd <= 5,
        "hba_le_10": hba <= 10,
        "tpsa_le_140": tpsa <= 140,
        "rotbonds_le_10": rot <= 10,
    }
    lipinski_ok = all(
        [rules["mw_le_500"], rules["logp_le_5"], rules["hbd_le_5"], rules["hba_le_10"]]
    )
    veber_ok = rules["tpsa_le_140"] and rules["rotbonds_le_10"]
    return {
        "rules": rules,
        "lipinski_pass": lipinski_ok,
        "veber_pass": veber_ok,
        "lipinski_violations": sum(
            1
            for k in ("mw_le_500", "logp_le_5", "hbd_le_5", "hba_le_10")
            if not rules[k]
        ),
    }


def parse_smiles(smiles: str) -> Dict[str, Any]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    canonical = Chem.MolToSmiles(mol)
    desc = compute_descriptors(canonical)
    physchem = {k: float(desc[k]) for k in PHYSCHEM_COLS}
    admet = {
        "physchem": physchem,
        "lipinski": lipinski_veber(physchem),
    }
    meta = molecule_meta(canonical, lookup=True)
    return {
        "smiles": smiles.strip(),
        "canonical_smiles": canonical,
        "name": meta["name"],
        "formula": meta["formula"],
        "molblock": mol_to_molblock(mol),
        "svg": mol_to_svg(mol),
        "physchem": physchem,
        "admet": admet,
        "valid": True,
    }


def from_molfile(molfile: str) -> Dict[str, Any]:
    mol = Chem.MolFromMolBlock(molfile)
    if mol is None:
        raise ValueError("Invalid molfile")
    smiles = Chem.MolToSmiles(mol)
    return parse_smiles(smiles)
