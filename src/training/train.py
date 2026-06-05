import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def set_seed(seed=42):
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model,
    train_data,
    val_data,
    test_data,
    epochs=50,
    lr=0.001,
    batch_size=32,
    device='cuda',
    patience=10,
    seed=42
):
    
    set_seed(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):

        # ======================
        # Training
        # ======================
        model.train()
        train_loss = 0

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            out = model(data).view(-1)

            loss = criterion(out, data.y.view(-1).float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ======================
        # Validation
        # ======================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data).view(-1)

                loss = criterion(out, data.y.view(-1).float())
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # ======================
        # Early Stopping
        # ======================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ======================
    # Load best model
    # ======================
    model.load_state_dict(best_model_state)

    # ======================
    # Test Evaluation
    # ======================
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data).view(-1)

            preds.extend(out.cpu().numpy())
            labels.extend(data.y.cpu().numpy().reshape(-1))

    preds = np.array(preds)
    labels = np.array(labels)

    # ======================
    # Metrics
    # ======================
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (preds > 0.5).astype(int))
    prec = precision_score(labels, (preds > 0.5).astype(int))
    rec = recall_score(labels, (preds > 0.5).astype(int))
    f1 = f1_score(labels, (preds > 0.5).astype(int))

    print("\n📊 Test Results")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    return model, {
        "auc": auc,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
  }
