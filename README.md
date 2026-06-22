#<div align="center">

# 🔬 NephroBench-GNN  

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-Graph%20Learning-FF6F00?style=for-the-badge)
![RDKit](https://img.shields.io/badge/RDKit-Chemoinformatics-2E8B57?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Nephrotoxicity%20Prediction-blueviolet?style=for-the-badge)

---

### 🧪 A Comprehensive Benchmark of Graph Neural Networks for Drug-Induced Nephrotoxicity Prediction

</div>

---

## 📌 Overview

NephroBench is a research-grade benchmark framework for evaluating Graph Neural Networks (GCN, GAT, GIN) in the task of drug-induced nephrotoxicity prediction.

Unlike standard molecular machine learning pipelines, this benchmark focuses on realistic generalization scenarios, including scaffold-based splitting, out-of-distribution evaluation, and domain shift analysis.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Molecular Representation** | 25 atom-level features |
| 🔀 **Scaffold-Based Split** | Bemis–Murcko framework |
| 🌐 **OOD Evaluation** | Cross-dataset transfer (Tox21 ↔ ClinTox) |
| 📉 **Domain Shift** | Tox21 vs ClinTox distribution analysis |
| 📊 **Data Efficiency** | 10% → 100% training scaling |
| 🔬 **Feature Ablation** | Electronic, ring, neighbor features |
| 📈 **Statistical Robustness** | 5 seeds, 95% CI, Wilcoxon test |
| ❌ **Failure Analysis** | FP/FN characterization |

---

## ⚙️ Supported Models

| Model | Type | Reference |
|-------|------|-----------|
| **GCN** | Graph Convolutional Network | Kipf & Welling (2017) |
| **GAT** | Graph Attention Network | Veličković et al. (2018) |
| **GIN** | Graph Isomorphism Network | Xu et al. (2019) |
| **Random Forest** | Classical Baseline | ECFP fingerprints |

---

## 🧪 Dataset Summary

| Property | Value |
|----------|-------|
| Total molecules | **3,207** |
| Sources | Tox21 + ClinTox |
| Positive samples | 1,843 (57.5%) |
| Negative samples | 1,364 (42.5%) |
| Task | Binary classification (Nephrotoxicity) |

---

## 🔀 Evaluation Protocol

### Scaffold-Based Split (Primary Evaluation)

| Split | Size |
|-------|------|
| Training | 2,015 |
| Validation | 356 |
| Test | 836 |

### Out-of-Distribution Evaluation

- Train Tox21 → Test ClinTox
- Train ClinTox → Test Tox21

### Data Efficiency Experiments

- 10%, 25%, 50%, 75%, 100% training data

---

## 🧠 Best Model Architecture (GIN)

```text
Input Molecular Graph (25 features)
        ↓
    GIN Layer 1 (sum aggregation)
        ↓
    GIN Layer 2
        ↓
    Global Mean Pooling
        ↓
    MLP Classifier (128 → 1)
        ↓
    Sigmoid Output
```

Configuration:

· 2 GIN layers
· Hidden dimension: 128
· Dropout: 0.7
· Global mean pooling
· Adam optimizer (lr=0.001)

---

📊 Main Results (Scaffold Split)

Model ROC-AUC Accuracy
Random Forest 0.800 0.770
GCN 0.826 0.767
GAT 0.812 0.751
GIN 0.876 0.840

---

## Figures

### Figure 1 — GIN Model Architecture
![Figure 1](figures/figure_1_architecture.png)

### Figure 2 — ROC Curves Comparison
![Figure 2](figures/figure_2_roc_curves.png)

### Figure 3 — Data Efficiency Scaling
![Figure 3](figures/figure_3_data_efficiency.png)

### Figure 4 — Feature Ablation Study
![Figure 4](figures/figure_4_ablation.png)

### Figure 5 — Final Model Comparison
![Figure 5](figures/figure_5_final_comparison.png)

### Figure 6 — Statistical Robustness
![Figure 6](figures/figure_6_statistical.png)

### Figure 7 — Overall Pipeline
![Figure 7](figures/figure_pipeline.png)

### Figure 8 — OOD Generalization
![Figure 8](figures/figure_ood_comparison.png)

### Figure 9 — Domain Shift Analysis
![Figure 9](figures/domain_shift.png)

### Figure 10 — Failure Analysis
![Figure 10](figures/figure_error_taxonomy.png)
---

🌐 Out-of-Distribution Performance

Setting ROC-AUC
In-distribution (Scaffold) 0.876
OOD average ~0.579

⚠️ Performance drops significantly under cross-dataset evaluation, highlighting strong domain shift effects in molecular learning.

---

🧪 Domain Shift Analysis

Property ClinTox Tox21 Difference
Molecular Weight 384.7 Da 305.8 Da +20.5%
LogP 2.8 2.3 +17.9%
Heavy Atoms 25.1 20.3 +19.1%
Ring Count 2.4 1.9 +20.8%

---

❌ Failure Analysis

Category Count Percentage Avg MW
Correct 672 80.4% 303.2 Da
False Positives 105 12.6% 390.7 Da
False Negatives 59 7.1% 284.1 Da

Observation: False positives tend to be larger molecules; false negatives tend to be smaller.

---

📦 Reproducibility

· ✅ Fixed random seeds (42, 123, 456, 789, 2024)
· ✅ Scaffold-based split provided
· ✅ Full preprocessing pipeline included
· ✅ Deterministic training settings
· ✅ 95% confidence intervals reported
· ✅ Wilcoxon signed-rank test (p = 0.0625)

---

⚠️ Limitations

· Dataset size is moderate (~3K molecules)
· No 3D molecular conformations
· Public dataset label noise
· Limited to 2D graph representations

---

🧭 Future Work

· 3D geometric GNNs (DimeNet++, EGNN)
· Multi-task toxicity prediction
· Self-supervised molecular pretraining
· Uncertainty quantification (Bayesian GNNs)

---

📜 Citation

```bibtex
@article{NephroBench2026,
  title={A Comprehensive Benchmark of Graph Neural Networks for Drug-Induced Nephrotoxicity Prediction},
  author={Moafi, Sepideh},
  year={2026}
}


📄 License

MIT License — see LICENSE file for details.

