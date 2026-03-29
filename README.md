# Molecular Property Prediction with Graph Neural Networks

A message-passing neural network (MPNN) trained on the QM9 dataset to predict 
molecular properties from raw atomic structure — no hand-crafted features.

## Background

Traditional molecular property prediction relies on hand-crafted chemical 
fingerprints that require domain expertise to construct. Graph neural networks 
offer a more principled approach — atoms are nodes, bonds are edges, and the 
model learns chemically meaningful representations directly from molecular 
topology through iterative message passing.

This project targets HOMO-LUMO gap prediction, the energy difference between 
the highest occupied and lowest unoccupied molecular orbital. This property 
governs reactivity, light absorption, and drug-protein interaction behavior, 
making it a core quantity in computational drug discovery.

## Approach

- NNConv-based message passing with edge-conditioned update functions
- 3-layer MPNN with global mean pooling and MLP readout head
- SHAP-based per-atom explainability to verify chemical interpretability
- Evaluation against MAE, RMSE, and R² on held-out QM9 test set

## Explainability

SHAP attribution maps are generated for individual molecules, coloring each 
atom by its contribution to the predicted gap. The model recovers chemically 
meaningful patterns without supervision — aromatic systems and heteroatoms 
show the strongest attribution, consistent with molecular orbital theory.

## Stack

PyTorch Geometric · RDKit · SHAP · Python

## Usage
```bash
git clone https://github.com/varun-dubagunta/mol-gnn
cd mol-gnn
pip install -r requirements.txt
python train.py --target 4 --epochs 100
python visualize.py
```