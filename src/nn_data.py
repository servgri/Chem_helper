"""Datasets and LightningDataModules for fingerprint / graph Tox21 models.

SMILES → Morgan fingerprints (MLP / ResNet) or padded atom graphs
(Transformer / MolGNN). No torch_geometric.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Common organic atoms + catch-all bucket for rare elements
COMMON_ATOMIC_NUMS: Tuple[int, ...] = (
    1,   # H
    5,   # B
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    14,  # Si
    15,  # P
    16,  # S
    17,  # Cl
    35,  # Br
    53,  # I
)
# one-hot over COMMON + unknown → 13 dims
# + degree (one-hot 0..5 → 6) + aromatic (1) + formal_charge clipped [-2,2] one-hot (5)
# + hybridization one-hot (5: SP, SP2, SP3, SP3D, OTHER) + in_ring (1) + num_hs clipped 0..3 (4)
# Total ≈ 13 + 6 + 1 + 5 + 5 + 1 + 4 = 35
DEFAULT_ATOM_FEAT_DIM = 35
DEFAULT_MAX_NODES = 64


# ---------------------------------------------------------------------------
# Low-level featurization
# ---------------------------------------------------------------------------


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def morgan_fingerprint(
    smiles: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> Optional[np.ndarray]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _one_hot(index: int, size: int) -> List[float]:
    v = [0.0] * size
    if 0 <= index < size:
        v[index] = 1.0
    else:
        v[-1] = 1.0  # unknown / overflow bucket when index out of range
    return v


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Compact atom feature vector (~35 dims)."""
    z = atom.GetAtomicNum()
    try:
        z_idx = COMMON_ATOMIC_NUMS.index(z)
    except ValueError:
        z_idx = len(COMMON_ATOMIC_NUMS)  # unknown bucket
    feats: List[float] = []
    feats.extend(_one_hot(z_idx, len(COMMON_ATOMIC_NUMS) + 1))  # 13

    deg = min(atom.GetDegree(), 5)
    feats.extend(_one_hot(deg, 6))  # 6

    feats.append(1.0 if atom.GetIsAromatic() else 0.0)  # 1

    # formal charge clipped to [-2, 2] → indices 0..4
    fc = int(max(-2, min(2, atom.GetFormalCharge())))
    feats.extend(_one_hot(fc + 2, 5))  # 5

    hybr = atom.GetHybridization()
    hybr_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 3,
    }
    feats.extend(_one_hot(hybr_map.get(hybr, 4), 5))  # 5

    feats.append(1.0 if atom.IsInRing() else 0.0)  # 1

    nhs = min(atom.GetTotalNumHs(), 3)
    feats.extend(_one_hot(nhs, 4))  # 4

    return np.asarray(feats, dtype=np.float32)


def smiles_to_graph(
    smiles: str,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """SMILES → (node_feats [N,F], adj [N,N], mask [N]).

    Molecules with more than ``max_nodes`` heavy+H atoms are truncated to the
    first ``max_nodes`` atoms (and their induced subgraph). Invalid SMILES
    return ``None``.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    n = min(n_atoms, max_nodes)
    feats = np.stack([atom_features(mol.GetAtomWithIdx(i)) for i in range(n)], axis=0)
    adj = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < n and j < n:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    mask = np.ones((n,), dtype=np.bool_)
    return feats, adj, mask


def smiles_to_fingerprint_tensor(
    smiles: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> Optional[torch.Tensor]:
    fp = morgan_fingerprint(smiles, n_bits=n_bits, radius=radius)
    if fp is None:
        return None
    return torch.from_numpy(fp)


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------


def collate_fingerprints(
    batch: Sequence[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    xs = torch.stack([b["x"] for b in batch], dim=0)
    ys = torch.stack([b["y"] for b in batch], dim=0)
    return {"x": xs, "y": ys}


def collate_graphs(
    batch: Sequence[Dict[str, Any]],
    max_nodes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pad node features / adjacency to a common N in the batch (or max_nodes)."""
    ns = [b["node_feats"].shape[0] for b in batch]
    n_pad = max(ns) if max_nodes is None else max_nodes
    feat_dim = batch[0]["node_feats"].shape[1]
    bsz = len(batch)

    node_feats = torch.zeros(bsz, n_pad, feat_dim, dtype=torch.float32)
    adj = torch.zeros(bsz, n_pad, n_pad, dtype=torch.float32)
    mask = torch.zeros(bsz, n_pad, dtype=torch.bool)
    ys = torch.stack([b["y"] for b in batch], dim=0)

    for i, item in enumerate(batch):
        n = item["node_feats"].shape[0]
        n_use = min(n, n_pad)
        node_feats[i, :n_use] = item["node_feats"][:n_use]
        adj[i, :n_use, :n_use] = item["adj"][:n_use, :n_use]
        mask[i, :n_use] = True

    return {"node_feats": node_feats, "adj": adj, "mask": mask, "y": ys}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class FingerprintDataset(Dataset):
    """Precomputed or on-the-fly Morgan fingerprints + binary labels."""

    def __init__(
        self,
        smiles: Sequence[str],
        y: Union[np.ndarray, Sequence[float]],
        n_bits: int = 2048,
        radius: int = 2,
        fingerprints: Optional[np.ndarray] = None,
        filter_invalid: bool = True,
    ) -> None:
        self.n_bits = n_bits
        self.radius = radius
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        kept_smiles: List[str] = []

        for i, smi in enumerate(smiles):
            if fingerprints is not None:
                fp = fingerprints[i]
                if fp is None or (isinstance(fp, float) and np.isnan(fp)):
                    if filter_invalid:
                        continue
                    t = torch.zeros(n_bits, dtype=torch.float32)
                else:
                    t = torch.as_tensor(fp, dtype=torch.float32)
            else:
                t_opt = smiles_to_fingerprint_tensor(smi, n_bits=n_bits, radius=radius)
                if t_opt is None:
                    if filter_invalid:
                        continue
                    t = torch.zeros(n_bits, dtype=torch.float32)
                else:
                    t = t_opt

            if not np.isfinite(y_arr[i]):
                if filter_invalid:
                    continue
            xs.append(t)
            ys.append(torch.tensor(float(y_arr[i]), dtype=torch.float32))
            kept_smiles.append(smi)

        self.xs = xs
        self.ys = ys
        self.smiles = kept_smiles

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": self.xs[idx], "y": self.ys[idx]}


class GraphDataset(Dataset):
    """Atom-feature graphs + binary labels for Transformer / MolGNN."""

    def __init__(
        self,
        smiles: Sequence[str],
        y: Union[np.ndarray, Sequence[float]],
        max_nodes: int = DEFAULT_MAX_NODES,
        filter_invalid: bool = True,
        drop_oversized: bool = False,
    ) -> None:
        self.max_nodes = max_nodes
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

        node_list: List[torch.Tensor] = []
        adj_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        kept_smiles: List[str] = []

        for i, smi in enumerate(smiles):
            if not np.isfinite(y_arr[i]):
                if filter_invalid:
                    continue
            mol = smiles_to_mol(smi)
            if mol is None:
                if filter_invalid:
                    continue
                continue
            if drop_oversized and mol.GetNumAtoms() > max_nodes:
                continue

            graph = smiles_to_graph(smi, max_nodes=max_nodes)
            if graph is None:
                if filter_invalid:
                    continue
                continue
            feats, adj, mask = graph
            node_list.append(torch.from_numpy(feats))
            adj_list.append(torch.from_numpy(adj))
            mask_list.append(torch.from_numpy(mask))
            ys.append(torch.tensor(float(y_arr[i]), dtype=torch.float32))
            kept_smiles.append(smi)

        self.node_feats = node_list
        self.adjs = adj_list
        self.masks = mask_list
        self.ys = ys
        self.smiles = kept_smiles
        self.atom_feat_dim = (
            int(node_list[0].shape[1]) if node_list else DEFAULT_ATOM_FEAT_DIM
        )

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "node_feats": self.node_feats[idx],
            "adj": self.adjs[idx],
            "mask": self.masks[idx],
            "y": self.ys[idx],
        }


# ---------------------------------------------------------------------------
# Lightning DataModules
# ---------------------------------------------------------------------------


def _as_str_array(smiles: Union[np.ndarray, Sequence[str]]) -> np.ndarray:
    return np.asarray(smiles, dtype=object)


class FingerprintDataModule(pl.LightningDataModule):
    """Train / val / test loaders of Morgan fingerprint tensors."""

    def __init__(
        self,
        smiles_train: Union[np.ndarray, Sequence[str]],
        y_train: np.ndarray,
        smiles_val: Union[np.ndarray, Sequence[str]],
        y_val: np.ndarray,
        smiles_test: Optional[Union[np.ndarray, Sequence[str]]] = None,
        y_test: Optional[np.ndarray] = None,
        n_bits: int = 2048,
        radius: int = 2,
        batch_size: int = 64,
        num_workers: int = 0,
        fingerprints_train: Optional[np.ndarray] = None,
        fingerprints_val: Optional[np.ndarray] = None,
        fingerprints_test: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.smiles_train = _as_str_array(smiles_train)
        self.smiles_val = _as_str_array(smiles_val)
        self.smiles_test = None if smiles_test is None else _as_str_array(smiles_test)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.y_val = np.asarray(y_val, dtype=np.float32)
        self.y_test = None if y_test is None else np.asarray(y_test, dtype=np.float32)
        self.n_bits = n_bits
        self.radius = radius
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fingerprints_train = fingerprints_train
        self.fingerprints_val = fingerprints_val
        self.fingerprints_test = fingerprints_test
        self.train_ds: Optional[FingerprintDataset] = None
        self.val_ds: Optional[FingerprintDataset] = None
        self.test_ds: Optional[FingerprintDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = FingerprintDataset(
                self.smiles_train,
                self.y_train,
                n_bits=self.n_bits,
                radius=self.radius,
                fingerprints=self.fingerprints_train,
            )
            self.val_ds = FingerprintDataset(
                self.smiles_val,
                self.y_val,
                n_bits=self.n_bits,
                radius=self.radius,
                fingerprints=self.fingerprints_val,
            )
        if stage in (None, "test") and self.smiles_test is not None and self.y_test is not None:
            self.test_ds = FingerprintDataset(
                self.smiles_test,
                self.y_test,
                n_bits=self.n_bits,
                radius=self.radius,
                fingerprints=self.fingerprints_test,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fingerprints,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fingerprints,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fingerprints,
        )


class GraphDataModule(pl.LightningDataModule):
    """Train / val / test loaders of padded molecular graphs."""

    def __init__(
        self,
        smiles_train: Union[np.ndarray, Sequence[str]],
        y_train: np.ndarray,
        smiles_val: Union[np.ndarray, Sequence[str]],
        y_val: np.ndarray,
        smiles_test: Optional[Union[np.ndarray, Sequence[str]]] = None,
        y_test: Optional[np.ndarray] = None,
        max_nodes: int = DEFAULT_MAX_NODES,
        batch_size: int = 32,
        num_workers: int = 0,
        drop_oversized: bool = False,
        pad_to_max_nodes: bool = True,
    ) -> None:
        super().__init__()
        self.smiles_train = _as_str_array(smiles_train)
        self.smiles_val = _as_str_array(smiles_val)
        self.smiles_test = None if smiles_test is None else _as_str_array(smiles_test)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.y_val = np.asarray(y_val, dtype=np.float32)
        self.y_test = None if y_test is None else np.asarray(y_test, dtype=np.float32)
        self.max_nodes = max_nodes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_oversized = drop_oversized
        self.pad_to_max_nodes = pad_to_max_nodes
        self.train_ds: Optional[GraphDataset] = None
        self.val_ds: Optional[GraphDataset] = None
        self.test_ds: Optional[GraphDataset] = None
        self.atom_feat_dim: int = DEFAULT_ATOM_FEAT_DIM

    def _make_ds(
        self,
        smiles: np.ndarray,
        y: np.ndarray,
    ) -> GraphDataset:
        return GraphDataset(
            smiles,
            y,
            max_nodes=self.max_nodes,
            drop_oversized=self.drop_oversized,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = self._make_ds(self.smiles_train, self.y_train)
            self.val_ds = self._make_ds(self.smiles_val, self.y_val)
            self.atom_feat_dim = self.train_ds.atom_feat_dim
        if stage in (None, "test") and self.smiles_test is not None and self.y_test is not None:
            self.test_ds = self._make_ds(self.smiles_test, self.y_test)

    def _collate(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_n = self.max_nodes if self.pad_to_max_nodes else None
        return collate_graphs(batch, max_nodes=max_n)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )


__all__ = [
    "COMMON_ATOMIC_NUMS",
    "DEFAULT_ATOM_FEAT_DIM",
    "DEFAULT_MAX_NODES",
    "atom_features",
    "morgan_fingerprint",
    "smiles_to_fingerprint_tensor",
    "smiles_to_graph",
    "smiles_to_mol",
    "collate_fingerprints",
    "collate_graphs",
    "FingerprintDataset",
    "GraphDataset",
    "FingerprintDataModule",
    "GraphDataModule",
]
