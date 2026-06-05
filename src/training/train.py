import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import copy


# ==================================================
# Training function with early stopping
# ==================================================
def train_model(
    model,
    train_data,
    val_data,
    test_data,
    epochs=50,
    lr=0.001,
    device=None,
    patience=10
):
    """
    Train GNN model with early stopping and evaluation
    """

    # --------------------------
    # Device handling (safe)
    # --------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)

    # --------------------------
    # Data loaders
    # --------------------------
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # --------------------------
    # Optimizer & loss
    # --------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # ==================================================
    # Training loop
    # ==================================================
    for epoch in range(epochs):

        # --------------------------
        # Train
        # --------------------------
        model.train()
        train_loss = 0.0

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            out = model(data)

            loss = criterion(out, data.y.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y.view(-1))
                val_loss += loss.item()

        # safe averaging
        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)

        # --------------------------
        # Early stopping
        # --------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

    # ==================================================
    # Load best model
    # ==================================================
    model.load_state_dict(best_model_state)

    # ==================================================
    # Test evaluation
    # ==================================================
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)

            preds.extend(out.cpu().numpy())
            labels.extend(data.y.cpu().numpy().reshape(-1))

    preds = np.array(preds)
    labels = np.array(labels)

    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (preds > 0.5).astype(int))

    print("\n📊 Test Results:")
    print(f"   ROC-AUC: {auc:.4f}")
    print(f"   Accuracy: {acc:.4f}")

    return model, auc, acc
