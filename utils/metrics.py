import torch


def compute_mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds - targets)).item()


def compute_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def compute_r2(preds: torch.Tensor, targets: torch.Tensor) -> float:
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    return (1 - ss_res / ss_tot).item()
