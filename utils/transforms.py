import torch
from torch_geometric.data import Data


class NormalizeTarget:
    """
    Extracts a single QM9 target property and normalizes it (zero mean, unit std).
    Applied as a transform so normalization happens on-the-fly.
    """
    def __init__(self, target_idx: int):
        self.target_idx = target_idx
        self.mean = None
        self.std = None

    def __call__(self, data: Data) -> Data:
        # Extract the single target column
        data.y = data.y[:, self.target_idx]
        return data
