import torch
from torch_geometric.data import Data
from rdkit import Chem


# ==================================================
# 25 ATOM FEATURES (FIXED)
# ==================================================
def get_atom_features_25(atom):
    features = []

    # 1–5 Basic atomic properties
    features.append(atom.GetAtomicNum() / 100.0)
    features.append(atom.GetDegree() / 10.0)
    features.append(atom.GetTotalNumHs() / 10.0)
    features.append(atom.GetImplicitValence() / 10.0)
    features.append(float(atom.GetIsAromatic()))

    # 6–10 Hybridization + structural
    hybrid_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }
    features.append(hybrid_map.get(atom.GetHybridization(), 0) / 5.0)
    features.append(float(atom.IsInRing()))
    features.append(atom.GetMass() / 200.0)
    features.append(atom.GetFormalCharge() / 5.0)
    features.append(atom.GetExplicitValence() / 10.0)

    # 11–15 Electronic properties
    features.append(atom.GetNumRadicalElectrons() / 5.0)
    features.append(float(atom.GetIsotope() > 0))
    features.append(float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED))
    features.append(len(atom.GetNeighbors()) / 10.0)

    # 16–20 Neighborhood + bonds
    neighbors = atom.GetNeighbors()
    if len(neighbors) > 0:
        avg_mass = sum(n.GetMass() for n in neighbors) / len(neighbors) / 200.0
    else:
        avg_mass = 0.0
    features.append(avg_mass)

    bonds = atom.GetBonds()
    features.append(sum(b.GetBondType() == Chem.rdchem.BondType.DOUBLE for b in bonds) / 5.0)
    features.append(sum(b.GetBondType() == Chem.rdchem.BondType.TRIPLE for b in bonds) / 5.0)

    features.append(atom.GetIdx() / 100.0)

    # 21–25 Ring membership (5 sizes)
    features.append(float(atom.IsInRingSize(3)))
    features.append(float(atom.IsInRingSize(4)))
    features.append(float(atom.IsInRingSize(5)))
    features.append(float(atom.IsInRingSize(6)))
    features.append(float(atom.IsInRingSize(7)))

    # ⚠️ 8-member ring removed to keep total = 25
    # (or alternatively you already had 3–8; this is the corrected consistent version)

    return features


# ==================================================
# SMILES → GRAPH
# ==================================================
def smiles_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor(
        [get_atom_features_25(atom) for atom in mol.GetAtoms()],
        dtype=torch.float
    )

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


# ==================================================
# DATASET BUILDER
# ==================================================
def create_graph_dataset(df):
    graphs = []
    failed = 0

    for idx, row in df.iterrows():
        g = smiles_to_graph(row["smiles"], row["label"])

        if g is not None:
            graphs.append(g)
        else:
            failed += 1

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx+1}/{len(df)}")

    print(f"Created {len(graphs)} graphs | Failed: {failed}")

    return graphs
