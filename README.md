🧠 NephroGNN-Benchmark

A Comprehensive Benchmark of Graph Neural Networks for Drug-Induced Nephrotoxicity Prediction


---

    


---

📁 Repository Structure

NephroGNN-Benchmark/
├── README.md
├── figures/
│   ├── figure_1_architecture.png
│   ├── figure_2_roc_curves.png
│   ├── figure_3_data_efficiency.png
│   ├── figure_4_ablation.png
│   ├── figure_5_final_comparison.png
│   ├── figure_6_statistical.png
│   ├── figure_pipeline.png
│   ├── figure_ood_comparison.png
│   ├── domain_shift.png
│   └── figure_error_taxonomy.png
├── configs/
├── data/
├── models/
├── training/
├── evaluation/
├── utils/
└── requirements.txt


---

🧬 NephroGNN Benchmark Overview

NephroGNN is a systematic benchmarking framework for evaluating Graph Neural Networks on drug-induced nephrotoxicity prediction.

Unlike conventional QSAR-based toxicity models, this framework explicitly focuses on:

Realistic scaffold-based generalization

Cross-dataset out-of-distribution robustness

Chemical domain shift quantification

Data efficiency scaling laws

Feature-level interpretability

Statistical stability across random seeds

Failure mode analysis in molecular space



---

🎯 Scientific Motivation

Drug-induced nephrotoxicity is a major cause of:

Late-stage drug attrition

Acute kidney injury (AKI)

Clinical trial failure


However, existing computational models suffer from:

> ❗ Over-optimistic evaluation due to random splits
❗ Poor generalization to unseen chemical scaffolds
❗ Lack of OOD validation protocols



NephroGNN addresses these limitations through a benchmark-first evaluation paradigm.


---

⚙️ Model Suite

We evaluate representative architectures across different expressive capacities:

Graph Neural Networks

GCN (Kipf & Welling)

GAT (Velickovic et al.)

GIN (Xu et al.)


Classical Baseline

Random Forest (ECFP fingerprints)



---

🧬 Molecular Representation

Each molecule is represented as a graph:

Nodes: Atoms

Edges: Chemical bonds

Node features (25-dimensional):

Atomic number

Degree

Formal charge

Hybridization

Aromaticity

Valence

Ring membership (3–8)

Electronic descriptors

Mass-normalized features




---

📊 Dataset Card

Sources

Tox21 (nuclear receptor + stress response assays)

ClinTox (clinical toxicity endpoints)


Final Dataset Statistics

Property	Value

Total molecules	3,207
Positive samples	1,843
Negative samples	1,364



---

🔀 Evaluation Protocol

Scaffold-Based Split (Primary Setting)

Split	Size

Train	2,015
Validation	356
Test	836


> Based on Bemis–Murcko scaffolds




---

Out-of-Distribution Evaluation

Cross-dataset transfer:

Tox21 → ClinTox

ClinTox → Tox21



---

Data Efficiency Evaluation

Training ratios:

10%

25%

50%

75%

100%



---

🧠 Model Architecture

Best-performing model: GIN

Input Molecular Graph (25 features)
        ↓
GIN Layer 1 (sum aggregation)
        ↓
GIN Layer 2
        ↓
Global Mean Pooling
        ↓
MLP Classifier
        ↓
Sigmoid Output


---

📈 Main Results (Scaffold Split)

Model	ROC-AUC

Random Forest	0.800
GCN	0.826
GAT	0.812
GIN	0.876



---

📉 Figures (Publication Quality)

🧬 Figure 1 — Model Architecture

figures/figure_1_architecture.png

📊 Figure 2 — ROC Curves

figures/figure_2_roc_curves.png

📈 Figure 3 — Data Efficiency Scaling

figures/figure_3_data_efficiency.png

🔬 Figure 4 — Feature Ablation

figures/figure_4_ablation.png

🏁 Figure 5 — Final Benchmark Comparison

figures/figure_5_final_comparison.png

📊 Figure 6 — Statistical Robustness

figures/figure_6_statistical.png


---

🌐 Out-of-Distribution Generalization

Setting	ROC-AUC

In-distribution (scaffold)	0.876
OOD average	~0.579


figures/figure_ood_comparison.png


---

🧪 Domain Shift Analysis

We observe systematic shifts between datasets:

Molecular weight ↑

LogP ↑

Ring complexity ↑

Heavy atom count ↑


👉 Indicates strong chemical distribution mismatch

figures/domain_shift.png


---

❌ Failure Mode Analysis

Error taxonomy reveals:

False positives → larger, more complex molecules

False negatives → smaller, less complex structures


figures/figure_error_taxonomy.png


---

🔬 Key Contributions

First unified scaffold + OOD nephrotoxicity benchmark

Systematic evaluation of GNN architectures

Domain shift quantification in chemical space

Data efficiency scaling curves

Feature ablation interpretability

Statistical robustness (5-seed CI analysis)

Failure mode taxonomy in molecular graphs



---

📦 Reproducibility

To ensure full reproducibility:

Fixed random seeds (42, 123, 456, 789, 2024)

Scaffold-based split provided

Full preprocessing pipeline included

Deterministic training settings

95% confidence intervals reported



---

⚠️ Limitations

Dataset size remains moderate (~3k molecules)

No 3D conformational modeling

Label noise in public toxicity datasets

Limited to 2D molecular graphs

No uncertainty-aware calibration in main models



---

🧭 Future Work

3D geometric GNNs (DimeNet++, EGNN)

Multi-task toxicity prediction

Self-supervised molecular pretraining

Uncertainty quantification (Bayesian GNNs)

Foundation model for toxicity prediction



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
