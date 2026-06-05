import torch
from torch_geometric.data import Data
from rdkit import Chem


# ==================================================
# 25 Atom Feature Extraction
# ==================================================
def get_atom_features_25(atom):
    features = []

    features.append(atom.GetAtomicNum() / 100.0)
    features.append(atom.GetDegree() / 10.0)
    features.append(atom.GetTotalNumHs() / 10.0)
    features.append(atom.GetImplicitValence() / 10.0)
    features.append(float(atom.GetIsAromatic()))

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
    features.append(atom.GetNumRadicalElectrons() / 5.0)
    features.append(float(atom.GetIsotope() > 0))

    features.append(float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED))

    features.append(0.0)  # stability placeholder

    neighbors = atom.GetNeighbors()
    features.append(len(neighbors) / 10.0)

    if len(neighbors) > 0:
        avg_mass = sum(n.GetMass() for n in neighbors) / len(neighbors) / 200.0
    else:
        avg_mass = 0.0
    features.append(avg_mass)

    bonds = atom.GetBonds()
    features.append(sum(b.GetBondType() == Chem.rdchem.BondType.DOUBLE for b in bonds) / 5.0)
    features.append(sum(b.GetBondType() == Chem.rdchem.BondType.TRIPLE for b in bonds) / 5.0)

    features.append(atom.GetIdx() / 100.0)

    for size in range(3, 9):
        features.append(float(atom.IsInRingSize(size)))

    return features


# ==================================================
# SMILES → Graph
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
# Dataset Builder
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
