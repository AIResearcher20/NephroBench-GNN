#🧬 NephroTox-GNN

Graph Neural Networks for Predicting Drug-Induced Nephrotoxicity

<div align="center">   

</div>
---

📌 Abstract

NephroTox-GNN is a graph-based deep learning framework for predicting drug-induced nephrotoxicity using molecular graph representations.
The framework evaluates multiple Graph Neural Network architectures (GCN, GAT, GIN) under a realistic out-of-distribution (OOD) setting using scaffold-based splitting, which simulates real-world drug discovery scenarios.


---

🧠 Key Idea

Molecules are modeled as graphs:

🧪 Nodes → Atoms (25-dimensional feature vectors)

🔗 Edges → Chemical bonds

⚙️ Learning → Message passing via GNNs


The goal is robust prediction of nephrotoxicity under distribution shift conditions.


---

🚀 Model Architectures

We evaluate three state-of-the-art GNN models:

🔷 GCN – Graph Convolutional Network

⚡ GAT – Graph Attention Network

🔥 GIN – Graph Isomorphism Network (Best Performance)



---

📊 Experimental Results

Model	ROC-AUC	Accuracy

🔥 GIN	0.876	0.840
🔷 GCN	0.826	0.767
⚡ GAT	0.812	0.751


✔ Evaluation performed under scaffold split (OOD setting)
✔ Early stopping with validation monitoring
✔ BCE loss optimization


---

🧱 Repository Structure

NephroTox-GNN/
│
├── src/
│   ├── models/            # GCN, GAT, GIN implementations
│   ├── datasets/         # Graph construction + scaffold split
│   ├── training/         # Training pipeline with early stopping
│   ├── evaluation/       # Metrics & evaluation scripts
│
├── notebooks/            # Experiments & analysis
├── results/
│   ├── figures/          # All plots and visualizations
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md


---

⚙️ Installation

git clone https://github.com/AIRESEARCHER20/NephroTox-GNN.git
cd NephroTox-GNN

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt


---

▶️ Training

python src/training/train.py


---

📊 Evaluation Metrics

The model is evaluated using:

ROC-AUC

Accuracy

Precision / Recall / F1-score

Confusion Matrix

Statistical significance over 5 seeds



---

🧪 Methodology

Molecular graph generation using RDKit

25-dimensional atom-level feature encoding

Scaffold-based splitting (realistic OOD evaluation)

GNN-based representation learning

Early stopping for generalization stability



---

🧬 Scientific Contributions

✔ Robust benchmarking of GNN architectures in drug toxicity prediction
✔ Realistic scaffold-based OOD evaluation strategy
✔ Fully reproducible deep learning pipeline
✔ Comparative study of GCN, GAT, and GIN models


---

📈 Figures

🧬 Architecture



📊 ROC Curves



⚙️ Pipeline



🧪 Ablation Study



📉 Data Efficiency



🔬 OOD Analysis



🌐 Domain Shift



❌ Failure Analysis




---

🔁 Reproducibility

Fixed random seeds

Scaffold-based splitting

Full training pipeline provided

Modular and extensible architecture



---

📄 Citation

@misc{nephrotoxgnn2026,
  title={NephroTox-GNN: Graph Neural Networks for Nephrotoxicity Prediction},
  author={Sepideh Moafi},
  year={2026},
  howpublished={GitHub Repository}
}


---

📜 License

This project is licensed under the MIT License.
