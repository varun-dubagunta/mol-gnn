"""
Training script for Molecular GNN on QM9 dataset.

QM9 contains ~130k small organic molecules with 19 quantum chemical properties.
We predict a single target property (default: HOMO-LUMO gap, target index 4).

Usage:
    python train.py --target 4 --epochs 100 --hidden_dim 64 --num_layers 3
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.gnn import MolecularGNN
from utils.metrics import compute_mae, compute_rmse
from utils.transforms import NormalizeTarget


# QM9 property names for reference
QM9_TARGETS = [
    'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
    'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom',
    'H_atom', 'G_atom', 'A', 'B', 'C'
]


def get_args():
    parser = argparse.ArgumentParser(description='Train Molecular GNN on QM9')
    parser.add_argument('--target', type=int, default=4,
                        help='QM9 target property index (0-18). Default: 4 (HOMO-LUMO gap)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'sum'])
    parser.add_argument('--data_dir', type=str, default='./data/qm9')
    parser.add_argument('--save_path', type=str, default='./checkpoints/best_model.pt')
    return parser.parse_args()


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return dataset[:train_end], dataset[train_end:val_end], dataset[val_end:]


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).squeeze(-1)
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).squeeze(-1)
        preds.append(pred.cpu())
        targets.append(batch.y.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return compute_mae(preds, targets), compute_rmse(preds, targets)


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Predicting property: {QM9_TARGETS[args.target]} (index {args.target})")

    print("Loading QM9 dataset...")
    import torch as _torch
    from torch_geometric.data import Data

    raw = _torch.load(f"{args.data_dir}/qm9_v3.pt", weights_only=False)
    dataset = []
    for item in raw[:10000]:
        d = Data(
            x=item['x'],
            edge_index=item['edge_index'],
            edge_attr=item['edge_attr'],
            y=item['y'][:, args.target],
            pos=item['pos']
        )
        dataset.append(d)

    train_set, val_set, test_set = split_dataset(dataset)
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size)

    # Build model
    sample = dataset[0]
    model = MolecularGNN(
        node_dim=sample.x.size(-1),
        edge_dim=sample.edge_attr.size(-1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_targets=1,
        pooling=args.pooling
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_mae = float('inf')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mae, val_rmse = evaluate(model, val_loader, device)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), args.save_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Final test evaluation
    model.load_state_dict(torch.load(args.save_path))
    test_mae, test_rmse = evaluate(model, test_loader, device)
    print(f"\nTest MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}")


if __name__ == '__main__':
    main()
