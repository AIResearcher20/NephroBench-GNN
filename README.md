🧠 NephroGNN-Benchmark

A Comprehensive Benchmark of Graph Neural Networks for Drug-Induced Nephrotoxicity Prediction


---

<p align="center">     

</p>
---

📌 Overview

NephroGNN is a research-grade benchmark framework for evaluating Graph Neural Networks (GCN, GAT, GIN) in the task of drug-induced nephrotoxicity prediction.

Unlike standard molecular machine learning pipelines, this benchmark focuses on realistic generalization scenarios, including scaffold-based splitting, out-of-distribution evaluation, and domain shift analysis.


---

🎯 Key Features

🧬 Graph-based molecular representation (25 atom-level features)

🔀 Scaffold-based train/test splitting (Bemis–Murcko)

🌐 Out-of-distribution (OOD) evaluation across datasets

📉 Domain shift quantification (Tox21 vs ClinTox)

📊 Data efficiency scaling analysis

🔬 Feature ablation studies

📈 Statistical robustness (multi-seed evaluation)

❌ Failure mode taxonomy



---

⚙️ Supported Models

Graph Convolutional Network (GCN)

Graph Attention Network (GAT)

Graph Isomorphism Network (GIN)

Random Forest baseline (ECFP fingerprints)



---

🧪 Dataset Summary

Total molecules: 3,207

Sources: Tox21 + ClinTox

Task: Binary classification (Nephrotoxicity prediction)



---

🔀 Evaluation Protocol

Scaffold-based split (primary evaluation)

OOD cross-dataset testing

Multi-seed statistical validation

Data efficiency experiments (10% → 100%)



---

🧠 Model Insight

Best performing model: GIN

2-layer GIN architecture

Hidden size: 128

Dropout: 0.7

Global mean pooling

MLP classifier with sigmoid output



---

📊 Main Results (Scaffold Split)

Model	ROC-AUC

Random Forest	0.800
GCN	0.826
GAT	0.812
GIN	0.876



---

📌 Key Figures

Architecture → figure_1_architecture.png

ROC Curves → figure_2_roc_curves.png

Data Efficiency → figure_3_data_efficiency.png

Ablation Study → figure_4_ablation.png

Final Comparison → figure_5_final_comparison.png

Statistical Analysis → figure_6_statistical.png

Pipeline → figure_pipeline.png

OOD Performance → figure_ood_comparison.png

Domain Shift → domain_shift.png

Error Taxonomy → figure_error_taxonomy.png



---

🌐 Out-of-Distribution Performance

Performance drops significantly under cross-dataset evaluation:

In-distribution (Scaffold): 0.876

Out-of-distribution: ~0.579


This highlights strong domain shift effects in molecular learning.


---

❌ Failure Analysis

False positives → larger molecular weight compounds

False negatives → smaller / simpler molecules


Indicates structural bias in learned representations.


---

📦 Reproducibility

Fixed random seeds (5 runs)

Scaffold-based splitting

Full training pipeline included

Deterministic evaluation setup

Confidence intervals reported



---

⚠️ Limitations

Dataset size is moderate (~3K molecules)

No 3D molecular conformations

Public dataset label noise

Limited to 2D graph representations



---

📜 Citation

@article{NephroGNN2026,
  title={A Comprehensive Benchmark of Graph Neural Networks for Drug-Induced Nephrotoxicity Prediction},
  author={Moafi , Sepideh},
  year={2026}
}


---

🪪 License

MIT License

