"""Research helpers for ToxMol / Tox21 experiments."""

from .qsar_utils import (  # noqa: F401
    TOX21_TARGETS,
    NR_TARGETS,
    SR_TARGETS,
    TOX21_URL,
    DEFAULT_TOX21_PATH,
    SplitData,
    download_tox21,
    load_tox21,
    morgan_fingerprint,
    featurize_smiles,
    prepare_endpoint,
)

try:
    from .nn_models import (  # noqa: F401
        FingerprintMLP,
        FingerprintResNet,
        AtomTransformer,
        MolGNN,
    )
    from .nn_data import (  # noqa: F401
        DEFAULT_ATOM_FEAT_DIM,
        DEFAULT_MAX_NODES,
        FingerprintDataModule,
        GraphDataModule,
        FingerprintDataset,
        GraphDataset,
        smiles_to_graph,
        collate_graphs,
        collate_fingerprints,
    )

    _NN_AVAILABLE = True
except ImportError:  # torch / lightning optional until installed
    _NN_AVAILABLE = False

__all__ = [
    "TOX21_TARGETS",
    "NR_TARGETS",
    "SR_TARGETS",
    "TOX21_URL",
    "DEFAULT_TOX21_PATH",
    "SplitData",
    "download_tox21",
    "load_tox21",
    "morgan_fingerprint",
    "featurize_smiles",
    "prepare_endpoint",
]

if _NN_AVAILABLE:
    __all__ += [
        "FingerprintMLP",
        "FingerprintResNet",
        "AtomTransformer",
        "MolGNN",
        "DEFAULT_ATOM_FEAT_DIM",
        "DEFAULT_MAX_NODES",
        "FingerprintDataModule",
        "GraphDataModule",
        "FingerprintDataset",
        "GraphDataset",
        "smiles_to_graph",
        "collate_graphs",
        "collate_fingerprints",
    ]
