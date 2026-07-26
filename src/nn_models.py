"""PyTorch Lightning molecular models for Tox21 (no torch_geometric).

Compatible with ``src.nn_data`` dict batches and ``src.flaml_train`` hparam names.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

try:
    from torchmetrics.classification import BinaryAUROC
except Exception:  # pragma: no cover
    BinaryAUROC = None

# Must match src.nn_data.DEFAULT_ATOM_FEAT_DIM
NODE_FEAT_DIM = 35

Batch = Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]


def _normalize_adj(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-normalize A' = A + I on valid nodes."""
    n = adj.size(-1)
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand_as(adj)
    pair = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    a = (adj + eye) * pair.float()
    deg = a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return a / deg


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class _AUROCHelper:
    def __init__(self) -> None:
        self.metric = BinaryAUROC() if BinaryAUROC is not None else None

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        if self.metric is None:
            return
        self.metric.update(torch.sigmoid(logits.detach().view(-1)), y.view(-1).int())

    def compute(self) -> Optional[torch.Tensor]:
        if self.metric is None:
            return None
        try:
            return self.metric.compute()
        except Exception:
            return None

    def reset(self) -> None:
        if self.metric is not None:
            self.metric.reset()


class BaseBinaryModule(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-5, pos_weight: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.register_buffer("pos_weight_buf", torch.tensor([float(pos_weight)], dtype=torch.float32))
        self._val_auroc = _AUROCHelper()
        self._test_auroc = _AUROCHelper()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1),
            y.view(-1).float(),
            pos_weight=self.pos_weight_buf,
        )

    def training_step(self, batch: Batch, batch_idx: int):
        logits, y = self._forward_batch(batch)
        loss = self._loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=y.size(0))
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        logits, y = self._forward_batch(batch)
        loss = self._loss(logits, y)
        self._val_auroc.update(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=y.size(0))
        return loss

    def on_validation_epoch_end(self) -> None:
        auc = self._val_auroc.compute()
        if auc is not None:
            self.log("val_auroc", auc, prog_bar=True)
            self.log("val_auc", auc, prog_bar=False)
        self._val_auroc.reset()

    def test_step(self, batch: Batch, batch_idx: int):
        logits, y = self._forward_batch(batch)
        loss = self._loss(logits, y)
        self._test_auroc.update(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=y.size(0))
        return loss

    def on_test_epoch_end(self) -> None:
        auc = self._test_auroc.compute()
        if auc is not None:
            self.log("test_auroc", auc)
        self._test_auroc.reset()

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        logits, _ = self._forward_batch(batch)
        return torch.sigmoid(logits.view(-1))

    def _forward_batch(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class FingerprintMLP(BaseBinaryModule):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        **_: Any,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, pos_weight=pos_weight)
        self.save_hyperparameters()
        layers = []
        prev = int(input_dim)
        h = int(hidden_dim)
        for _i in range(int(n_layers)):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(float(dropout))]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def _forward_batch(self, batch: Batch):
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            x, y = batch[0], batch[1]
        return self.forward(x), y


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FingerprintResNet(BaseBinaryModule):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        n_layers: int = 3,  # alias used by shared search space
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        **_: Any,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, pos_weight=pos_weight)
        self.save_hyperparameters()
        blocks = int(n_blocks) if n_blocks is not None else int(n_layers)
        h = int(hidden_dim)
        self.stem = nn.Sequential(nn.Linear(int(input_dim), h), nn.ReLU(), nn.BatchNorm1d(h))
        self.blocks = nn.Sequential(*[_ResBlock(h, float(dropout)) for _ in range(blocks)])
        self.head = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x))).squeeze(-1)

    def _forward_batch(self, batch: Batch):
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            x, y = batch[0], batch[1]
        return self.forward(x), y


class AtomTransformer(BaseBinaryModule):
    def __init__(
        self,
        node_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 64,
        d_model: Optional[int] = None,
        n_layers: int = 2,
        nhead: int = 4,
        n_heads: Optional[int] = None,
        dim_feedforward: int = 128,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        **_: Any,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, pos_weight=pos_weight)
        self.save_hyperparameters()
        d_model = int(d_model or hidden_dim)
        nhead = int(n_heads or nhead)
        # nhead must divide d_model
        while d_model % nhead != 0 and nhead > 1:
            nhead //= 2
        ff = int(ff_dim or dim_feedforward)
        self.input_proj = nn.Linear(int(node_dim), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=float(dropout),
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 1),
        )

    def forward(self, node_feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(node_feats)
        h = self.encoder(h, src_key_padding_mask=~mask.bool())
        return self.head(_masked_mean(h, mask)).squeeze(-1)

    def _forward_batch(self, batch: Batch):
        if isinstance(batch, dict):
            return self.forward(batch["node_feats"], batch["mask"]), batch["y"]
        node_feats, _adj, mask, y = batch
        return self.forward(node_feats, mask), y


class _GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.matmul(adj, self.lin(x)))


class MolGNN(BaseBinaryModule):
    def __init__(
        self,
        node_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_message_passes: Optional[int] = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        **_: Any,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, pos_weight=pos_weight)
        self.save_hyperparameters()
        n_layers = int(n_message_passes or n_layers)
        h = int(hidden_dim)
        dims = [int(node_dim)] + [h] * n_layers
        self.convs = nn.ModuleList([_GraphConv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.drop = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(h, 1),
        )

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        adj_n = _normalize_adj(adj, mask)
        h = node_feats
        for conv in self.convs:
            h = self.drop(conv(h, adj_n))
            h = h * mask.unsqueeze(-1).float()
        return self.head(_masked_mean(h, mask)).squeeze(-1)

    def _forward_batch(self, batch: Batch):
        if isinstance(batch, dict):
            return self.forward(batch["node_feats"], batch["adj"], batch["mask"]), batch["y"]
        node_feats, adj, mask, y = batch
        return self.forward(node_feats, adj, mask), y


MODEL_REGISTRY: Dict[str, Any] = {
    "mlp": FingerprintMLP,
    "resnet": FingerprintResNet,
    "transformer": AtomTransformer,
    "gnn": MolGNN,
}


def build_model(model_type: str, pos_weight: float = 1.0, **hparams) -> BaseBinaryModule:
    key = model_type.lower().replace("lightning_", "").replace("lightning", "")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type={model_type}. Choose from {list(MODEL_REGISTRY)}")
    import inspect

    cls = MODEL_REGISTRY[key]
    sig = inspect.signature(cls.__init__)
    allowed = {k: v for k, v in hparams.items() if k in sig.parameters}
    allowed["pos_weight"] = pos_weight
    return cls(**allowed)
