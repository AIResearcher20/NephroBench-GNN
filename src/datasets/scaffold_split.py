from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
import numpy as np
import pandas as pd


# ==================================================
# Scaffold extraction
# ==================================================
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except:
        return None


# ==================================================
# Scaffold split (OOD evaluation)
# ==================================================
def scaffold_split(df, test_size=0.2, val_size=0.15, seed=42):
    """
    Scaffold-based splitting for realistic OOD evaluation
    """

    df = df.copy()  # ⚠️ avoid leakage

    # extract scaffolds
    df["scaffold"] = df["smiles"].apply(get_scaffold)

    # remove invalid molecules
    df = df[df["scaffold"].notna()].reset_index(drop=True)

    # group by scaffold
    scaffold_to_indices = defaultdict(list)

    for idx, scaffold in enumerate(df["scaffold"]):
        scaffold_to_indices[scaffold].append(idx)

    # shuffle scaffolds
    scaffolds = list(scaffold_to_indices.keys())
    random.seed(seed)
    random.shuffle(scaffolds)

    # train/test split on scaffold level
    split_idx = int((1 - test_size) * len(scaffolds))

    train_scaffolds = set(scaffolds[:split_idx])
    test_scaffolds = set(scaffolds[split_idx:])

    train_idx, test_idx = [], []

    for scaffold, indices in scaffold_to_indices.items():
        if scaffold in train_scaffolds:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)

    # further split train → train/val
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=df.loc[train_idx, "label"]
    )

    print("Scaffold Split Summary:")
    print(f"Train: {len(train_idx)}")
    print(f"Val:   {len(val_idx)}")
    print(f"Test:  {len(test_idx)}")

    return train_idx, val_idx, test_idx, df
