"""Rule-based retrosynthesis (1–2 steps) via RDKit reaction SMARTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from django.conf import settings
from rdkit import Chem
from rdkit.Chem import AllChem

from molecules.services import mol_to_svg, molecule_meta
from src.qsar_utils import smiles_to_mol


@dataclass(frozen=True)
class RetroRule:
    name: str
    smarts: str
    description: str


# Product-side transforms: apply RunReactants on the target molecule
RETRO_RULES: List[RetroRule] = [
    RetroRule(
        "amide_hydrolysis",
        "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OH].[N:3]",
        "Amide bond disconnection (hydrolysis)",
    ),
    RetroRule(
        "ester_hydrolysis",
        "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[O:3][C:4]",
        "Ester bond disconnection (hydrolysis)",
    ),
    RetroRule(
        "ether_cleavage",
        "[C:1][O:2][C:3]>>[C:1][OH].[C:3][OH]",
        "Ether cleavage",
    ),
    RetroRule(
        "nitro_reduction",
        "[c:1][N+:2](=O)[O-]>>[c:1][NH2:2]",
        "Aromatic nitro → aniline (forward reduction; retro shows nitro precursor)",
    ),
    # For nitro: we want retro so product aniline ← nitro. Invert:
    RetroRule(
        "aniline_from_nitro",
        "[c:1][NH2:2]>>[c:1][N+:2](=O)[O-]",
        "Aniline from nitroarene (retro)",
    ),
    RetroRule(
        "boc_deprotection",
        "[N:1][C:2](=O)OC(C)(C)C>>[N:1]",
        "Boc deprotection (retro adds Boc)",
    ),
    RetroRule(
        "amine_boc_protection_retro",
        "[N:1;H1,H2]>>[N:1]C(=O)OC(C)(C)C",
        "Install Boc on amine (as precursor of free amine)",
    ),
    RetroRule(
        "suzuki_biaryl",
        "[c:1][c:2]>>[c:1]B(O)O.[c:2]Br",
        "Biaryl Suzuki disconnection",
    ),
    RetroRule(
        "reductive_amination",
        "[C:1][N:2]>>[C:1]=O.[N:2]",
        "Reductive amination disconnection",
    ),
    RetroRule(
        "n_alkylation",
        "[N:1][C:2]>>[N:1].[C:2]Br",
        "N-alkylation disconnection",
    ),
    RetroRule(
        "williamson_ether",
        "[C:1][O:2][C:3]>>[C:1][OH:2].[C:3]Br",
        "Williamson ether disconnection",
    ),
    RetroRule(
        "amide_from_acid_chloride",
        "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])Cl.[N:3]",
        "Amide from acid chloride + amine",
    ),
]


def _compile_rules() -> List[Tuple[RetroRule, AllChem.ChemicalReaction]]:
    out = []
    for rule in RETRO_RULES:
        try:
            rxn = AllChem.ReactionFromSmarts(rule.smarts)
            if rxn is not None:
                out.append((rule, rxn))
        except Exception:
            continue
    return out


_COMPILED = _compile_rules()


def _canonical(smi: str) -> Optional[str]:
    mol = smiles_to_mol(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _precursor_set_key(precursors: List[str]) -> str:
    """Order-independent key: A+B == B+A (canonical SMILES, sorted)."""
    cans: List[str] = []
    for p in precursors:
        cans.append(_canonical(p) or str(p).strip())
    return "|".join(sorted(cans))


def _apply_one_step(smiles: str) -> List[Dict[str, Any]]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return []
    hits: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for rule, rxn in _COMPILED:
        try:
            outcomes = rxn.RunReactants((mol,))
        except Exception:
            continue
        for products in outcomes:
            precursors: List[str] = []
            ok = True
            for p in products:
                try:
                    Chem.SanitizeMol(p)
                    smi = Chem.MolToSmiles(p)
                    # Re-canonicalize without atom maps / tautomer noise
                    smi = _canonical(smi) or smi
                except Exception:
                    ok = False
                    break
                if not smi:
                    ok = False
                    break
                precursors.append(smi)
            if not ok or not precursors:
                continue
            # Deduplicate A+B vs B+A (and identical precursor multisets)
            key = _precursor_set_key(precursors)
            if key in seen:
                continue
            seen.add(key)
            # Stable display order: sorted by canonical SMILES
            order = sorted(range(len(precursors)), key=lambda i: precursors[i])
            precursors = [precursors[i] for i in order]
            svgs = []
            labels = []
            for smi in precursors:
                m = smiles_to_mol(smi)
                svgs.append(mol_to_svg(m) if m is not None else "")
                meta = molecule_meta(smi, lookup=True)
                labels.append(meta["name"] or meta["formula"] or smi)
            hits.append(
                {
                    "reaction": rule.name,
                    "description": rule.description,
                    "precursors": precursors,
                    "precursor_labels": labels,
                    "precursor_svgs": svgs,
                    "depth": 1,
                }
            )
            if len(hits) >= int(settings.TOXMOL_RETRO_BRANCH_LIMIT):
                return hits
    return hits


def plan_retrosynthesis(
    smiles: str,
    max_depth: Optional[int] = None,
) -> Dict[str, Any]:
    max_depth = int(max_depth or settings.TOXMOL_RETRO_MAX_DEPTH)
    max_depth = max(1, min(max_depth, 2))
    canonical = _canonical(smiles)
    if canonical is None:
        raise ValueError("Invalid SMILES")

    product_mol = smiles_to_mol(canonical)
    product_svg = mol_to_svg(product_mol) if product_mol else ""

    routes: List[Dict[str, Any]] = []
    seen_route_sets: Set[str] = set()
    step1 = _apply_one_step(canonical)
    for route in step1:
        set_key = _precursor_set_key(route["precursors"])
        if set_key in seen_route_sets:
            continue
        seen_route_sets.add(set_key)
        product_meta = molecule_meta(canonical, lookup=True)
        routes.append(
            {
                **route,
                "product": canonical,
                "product_label": product_meta["name"] or product_meta["formula"] or "Продукт",
                "steps": [
                    {
                        "depth": 1,
                        "reaction": route["reaction"],
                        "description": route["description"],
                        "from_smiles": canonical,
                        "precursors": route["precursors"],
                        "precursor_svgs": route["precursor_svgs"],
                    }
                ],
            }
        )

    if max_depth >= 2:
        extended: List[Dict[str, Any]] = []
        for route in routes[: int(settings.TOXMOL_RETRO_BRANCH_LIMIT)]:
            for prec in route["precursors"]:
                nested = _apply_one_step(prec)
                for n in nested[:3]:
                    # Final leaf precursors define the route identity (order-invariant)
                    set_key = f"2|{_precursor_set_key(n['precursors'])}|via|{_canonical(prec) or prec}"
                    if set_key in seen_route_sets:
                        continue
                    seen_route_sets.add(set_key)
                    inter_meta = molecule_meta(prec, lookup=True)
                    product_meta = molecule_meta(canonical, lookup=True)
                    extended.append(
                        {
                            "reaction": f"{route['reaction']} → {n['reaction']}",
                            "description": f"{route['description']}; then {n['description']}",
                            "precursors": n["precursors"],
                            "precursor_labels": n.get("precursor_labels")
                            or [
                                molecule_meta(s, lookup=True)["name"] for s in n["precursors"]
                            ],
                            "precursor_svgs": n["precursor_svgs"],
                            "depth": 2,
                            "product": canonical,
                            "product_label": product_meta["name"]
                            or product_meta["formula"]
                            or "Продукт",
                            "intermediate": prec,
                            "intermediate_label": inter_meta["name"]
                            or inter_meta["formula"]
                            or "Интермедиат",
                            "steps": [
                                route["steps"][0],
                                {
                                    "depth": 2,
                                    "reaction": n["reaction"],
                                    "description": n["description"],
                                    "from_smiles": prec,
                                    "precursors": n["precursors"],
                                    "precursor_svgs": n["precursor_svgs"],
                                },
                            ],
                        }
                    )
                    if len(extended) >= int(settings.TOXMOL_RETRO_BRANCH_LIMIT):
                        break
                if len(extended) >= int(settings.TOXMOL_RETRO_BRANCH_LIMIT):
                    break
            if len(extended) >= int(settings.TOXMOL_RETRO_BRANCH_LIMIT):
                break
        routes.extend(extended)

    return {
        "smiles": canonical,
        "svg": product_svg,
        "max_depth": max_depth,
        "routes": routes[: int(settings.TOXMOL_RETRO_BRANCH_LIMIT) * 2],
        "n_routes": len(routes),
    }
