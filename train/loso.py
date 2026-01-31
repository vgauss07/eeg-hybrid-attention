import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .evaluate import train_one_epoch, eval_accuracy, count_params


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def run_loso_for_model(
    X, y, subjects,
    model_name: str,
    model_fn,
    out_dir: Path,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_ratio: float = 0.15,
    seed: int = 42,
    save_checkpoints: bool = False
):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    if save_checkpoints:
        (out_dir / "checkpoints").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)

    unique_subjects = np.unique(subjects)

    fold_rows = []
    for test_subj in unique_subjects:
        train_mask = subjects != test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=val_ratio,
            stratify=y_train,
            random_state=seed
        )

        tr_loader = DataLoader(EEGDataset(X_tr, y_tr),
                               batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(EEGDataset(X_val, y_val),
                               batch_size=batch_size, shuffle=False)
        te_loader = DataLoader(EEGDataset(X_test, y_test),
                               batch_size=batch_size, shuffle=False)

        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        params = count_params(model)
        best_val = -1.0
        best_state = None
        best_epoch = 0

        # simple early selection based on best val acc
        for ep in range(1, epochs + 1):
            _, _ = train_one_epoch(model, tr_loader, optimizer, device,
                                   use_amp=use_amp)
            val_acc = eval_accuracy(model, va_loader, device)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_epoch = ep

        # test
        model.load_state_dict(best_state, strict=True)
        test_acc = eval_accuracy(model, te_loader, device)

        if save_checkpoints:
            ckpt_path = out_dir / "checkpoints" / f"{model_name}_subj{int(test_subj)}.pt"
            torch.save(best_state, ckpt_path)

        fold_rows.append({
            "model": model_name,
            "subject": int(test_subj),
            "best_epoch": int(best_epoch),
            "val_acc_best": float(best_val),
            "test_acc": float(test_acc),
            "params": int(params),
        })

    df = pd.DataFrame(fold_rows)
    summary = {
        "model": model_name,
        "mean_test_acc": float(df["test_acc"].mean()),
        "std_test_acc": float(df["test_acc"].std(ddof=0)),
        "mean_params": float(df["params"].mean()),
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_ratio": val_ratio,
        "seed": seed,
    }

    # save
    df.to_csv(out_dir / "tables" / f"{model_name}_folds.csv", index=False)
    with open(out_dir / "logs" / f"{model_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return df, summary
