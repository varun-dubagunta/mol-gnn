"""
Visualization for Molecular GNN results — with SHAP feature importance.
Run after training: python visualize.py

Outputs:
  results.png      — predicted vs actual, error distribution, error vs actual
  shap.png         — SHAP feature importance bar chart
  mol_shap.png     — atom-level SHAP values drawn on example molecules
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import shap
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from models.gnn import MolecularGNN

TARGET_IDX = 4
DATA_PATH  = './data/qm9/qm9_v3.pt'
QM9_NAMES  = ['mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']

# Node feature names matching QM9's 11-dim atom feature vector
NODE_FEATURE_NAMES = [
    'C (one-hot)', 'H (one-hot)', 'O (one-hot)', 'N (one-hot)', 'F (one-hot)',
    'Atomic number', 'Acceptor', 'Donor', 'Aromatic', 'Num Hs', 'Valence'
]

ATOM_SYMBOLS = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# ── Load data ──────────────────────────────────────────────────────
print("Loading dataset...")
raw = torch.load(DATA_PATH, weights_only=False)
dataset = []
for item in raw[:10000]:
    d = Data(
        x=item['x'],
        edge_index=item['edge_index'],
        edge_attr=item['edge_attr'],
        y=item['y'][:, TARGET_IDX],
        pos=item['pos'],
        z=item['z']
    )
    dataset.append(d)

test_set    = dataset[9000:]
test_loader = DataLoader(test_set, batch_size=32)

# ── Load model ─────────────────────────────────────────────────────
sample = dataset[0]
model = MolecularGNN(
    node_dim=sample.x.shape[1],
    edge_dim=sample.edge_attr.shape[1],
    hidden_dim=64,
    num_layers=3
).to(device)
model.load_state_dict(torch.load('./checkpoints/best_model.pt', map_location=device))
model.eval()

# ── Collect predictions ────────────────────────────────────────────
preds, targets = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds.append(model(batch).squeeze().cpu())
        targets.append(batch.y.cpu())

preds   = torch.cat(preds).numpy()
targets = torch.cat(targets).numpy()
errors  = preds - targets
mae     = np.mean(np.abs(errors))
rmse    = np.sqrt(np.mean(errors**2))
r2      = 1 - np.sum((targets - preds)**2) / np.sum((targets - targets.mean())**2)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# ══════════════════════════════════════════════════════════════════
# Plot 1: Prediction quality (3 panels)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Molecular GNN — {QM9_NAMES[TARGET_IDX].upper()} Prediction', fontsize=14)

ax = axes[0]
ax.scatter(targets, preds, alpha=0.4, s=10, color='steelblue')
lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_title(f'Predicted vs Actual\nR² = {r2:.3f}')
ax.legend()

ax = axes[1]
sns.histplot(errors, bins=50, kde=True, ax=ax, color='steelblue')
ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Prediction Error (eV)')
ax.set_title(f'Error Distribution\nMAE={mae:.4f}  RMSE={rmse:.4f}')

ax = axes[2]
ax.scatter(targets, np.abs(errors), alpha=0.4, s=10, color='steelblue')
ax.axhline(mae, color='red', linestyle='--', linewidth=1.5, label=f'MAE={mae:.4f}')
ax.set_xlabel('Actual Value (eV)')
ax.set_ylabel('Absolute Error (eV)')
ax.set_title('Error vs Actual Value')
ax.legend()

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("Saved → results.png")

# ══════════════════════════════════════════════════════════════════
# Plot 2: SHAP global feature importance
# ══════════════════════════════════════════════════════════════════
print("\nComputing SHAP values (this takes ~2-3 min)...")

# Run SHAP per-molecule: each molecule gets its own KernelExplainer
# Input is the (n_atoms, 11) node feature matrix for that molecule
# This avoids edge_index mismatch from stacking molecules together
cpu = torch.device('cpu')
model_cpu = model.to(cpu)
shap_molecules = test_set[:50]  # 50 molecules is enough for global importance

all_shap_values = []  # will collect (n_atoms, 11) shap arrays per molecule

for mol_idx, mol in enumerate(shap_molecules):
    mol_feats = mol.x.numpy()           # (n_atoms, 11)
    edge_index = mol.edge_index.cpu()
    edge_attr  = mol.edge_attr.cpu()
    n = mol_feats.shape[0]

    def predict_this_mol(x: np.ndarray) -> np.ndarray:
        """x shape: (n_samples, 11) — SHAP perturbs one atom at a time."""
        out = np.zeros(x.shape[0])
        with torch.no_grad():
            for j in range(x.shape[0]):
                # Replace all atoms' features with this perturbed version
                x_t = torch.tensor(
                    np.tile(x[j], (n, 1)), dtype=torch.float
                )
                d = Data(
                    x=x_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=torch.zeros(n, dtype=torch.long)
                )
                out[j] = model_cpu(d).item()
        return out

    # Background = mean atom features for this molecule
    background_mol = mol_feats.mean(axis=0, keepdims=True)
    explainer_mol  = shap.KernelExplainer(predict_this_mol, background_mol)
    sv = explainer_mol.shap_values(mol_feats, nsamples=30)  # (n_atoms, 11)
    all_shap_values.append(sv)

    if (mol_idx + 1) % 10 == 0:
        print(f"  {mol_idx+1}/{len(shap_molecules)} molecules done")

# Stack all atom SHAP values → (total_atoms, 11)
all_shap_stacked = np.vstack(all_shap_values)
mean_shap = np.abs(all_shap_stacked).mean(axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['steelblue' if v < mean_shap.max() else 'tomato' for v in mean_shap]
bars = ax.barh(NODE_FEATURE_NAMES, mean_shap, color=colors)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Global Feature Importance (SHAP)\nWhich atom features drive HOMO-LUMO gap prediction?')
ax.invert_yaxis()
for bar, val in zip(bars, mean_shap):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('shap.png', dpi=150, bbox_inches='tight')
print("Saved → shap.png")

# ══════════════════════════════════════════════════════════════════
# Plot 3: Per-atom SHAP on 4 example molecules
# ══════════════════════════════════════════════════════════════════
print("Drawing per-atom SHAP on example molecules...")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Per-Atom SHAP Values — color = contribution to HOMO-LUMO gap prediction',
             fontsize=12)

for mol_idx in range(4):
    mol  = shap_molecules[mol_idx]
    n    = mol.x.shape[0]
    sv   = all_shap_values[mol_idx]            # (n_atoms, 11) — already per-molecule
    atom_shap = sv.sum(axis=1)                 # sum across features → per-atom scalar
    z_list = mol.z.numpy() if hasattr(mol, 'z') else [6]*n
    pos2d  = mol.pos.numpy()[:, :2]            # use x,y coords from QM9

    norm   = mcolors.TwoSlopeNorm(
        vmin=atom_shap.min(), vcenter=0, vmax=max(atom_shap.max(), 1e-6)
    )
    cmap   = cm.RdBu_r

    ax = axes[mol_idx]
    # Draw bonds
    ei = mol.edge_index.numpy()
    for src, dst in zip(ei[0], ei[1]):
        if src < dst:
            ax.plot(
                [pos2d[src,0], pos2d[dst,0]],
                [pos2d[src,1], pos2d[dst,1]],
                'k-', linewidth=1.2, zorder=1
            )
    # Draw atoms
    sc = ax.scatter(
        pos2d[:,0], pos2d[:,1],
        c=atom_shap, cmap=cmap, norm=norm,
        s=300, zorder=2, edgecolors='black', linewidths=0.5
    )
    # Atom labels
    for j, (x, y) in enumerate(pos2d):
        sym = ATOM_SYMBOLS.get(int(z_list[j]), '?')
        ax.text(x, y, sym, ha='center', va='center', fontsize=7, fontweight='bold')

    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='SHAP')
    actual = mol.y.item() if mol.y.dim() == 0 else mol.y[0].item()
    ax.set_title(f'Molecule {mol_idx+1}\nActual gap: {actual:.3f} eV', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('mol_shap.png', dpi=150, bbox_inches='tight')
print("Saved → mol_shap.png")
print("\nDone! Open results.png, shap.png, mol_shap.png in Windows Explorer.")