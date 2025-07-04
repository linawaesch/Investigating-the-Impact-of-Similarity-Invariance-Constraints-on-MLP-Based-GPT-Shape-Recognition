# Investigating-the-Impact-of-Similarity-Invariance-Constraints-on-MLP-Based-GPT-Shape-Recognition

This repository contains all code and data for Lina Wäsch’s Semester Thesis at ETH Zürich, where we train a physics-informed MLP to recover rotation, scale and translation invariants from first-order Generalized Polarization Tensors (GPTs).

## Repo structure

- **MLP.py**  
  Defines the feed-forward network, training loop (with consistency penalty) and test-set RMSE evaluation.
- **MLP_percentages.py**  
  Runs the same model to compute and plot shape-detection percentages over the test set.
- **compute_invariants.py**  
  Parses raw GPT entries (real/imag pairs) and computes the singular-value–ratio invariants.
- **data_generation.m**  
  MATLAB script that applies random rotation, scaling and translation to dictionary shapes and reconstructs noisy/theoretical GPTs.
- **mean_invariants_by_shape.csv**  
  CSV of mean invariants for each shape; used as ground-truth lookup in classification.
- **figure41_42.py**  
  Generates Figures 4.1–4.2 (impact of training-set size on RMSE and accuracy).
- **figure43_44.py**  
  Generates Figures 4.3–4.4 (robustness to additive noise).
- **figure45_47.py**  
  Generates Figures 4.5–4.7 (effect of consistency penalty weight λ).
