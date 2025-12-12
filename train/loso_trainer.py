import numpy as np
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def train_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    correct, total = 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        correct += (out.argmax(1) == y).sum().item()
        total += len(y)

    return correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def run_loso(X, y, subjects, model_fn, results_dir,
             epochs=120, batch_size=64, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Using device:", device)

    results_dir = Path(results_dir)
    table_dir = results_dir / "tables"
    ckpt_dir = results_dir / "checkpoints"
    table_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    folds = []

    for test_subj in np.unique(subjects):

        mask = subjects != test_subj
        X_train, y_train = X[mask], y[mask]
        X_test, y_test = X[~mask], y[~mask]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train)

        tr_loader = DataLoader(EEGDataset(X_train, y_train),
                               batch_size, shuffle=True, pin_memory=True)
        va_loader = DataLoader(EEGDataset(X_val, y_val),
                               batch_size, pin_memory=True)
        te_loader = DataLoader(EEGDataset(X_test, y_test),
                               batch_size, pin_memory=True)

        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best_val, best_state = 0, None

        for epoch in range(1, epochs+1):
            train_epoch(model, tr_loader, opt, device)
            val_acc = eval_epoch(model, va_loader, device)

            if val_acc > best_val:
                best_val, best_state = val_acc, model.state_dict()

            if epoch % 20 == 0:
                print(f"Subject {test_subj} Epoch {epoch}"
                      f"| Val Acc: {val_acc:.3f}")

        model.load_state_dict(best_state)
        test_acc = eval_epoch(model, te_loader, device)

        torch.save(best_state, ckpt_dir / f"eegnet_subject_{test_subj}.pt")

        folds.append({"subject": int(test_subj),
                      "test_accuracy": float(test_acc)})

    df = pd.DataFrame(folds)
    df.loc["mean"] = ["mean", df["test_accuracy"].mean()]
    df.loc["std"] = ["std", df["test_accuracy"].std()]
    df.to_csv(table_dir / "eegnet_loso_results.csv", index=False)

    with open(results_dir / "eegnet_loso_results.json", "w") as f:
        json.dump(folds, f, indent=2)

    print(df)
