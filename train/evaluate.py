import torch
import torch.nn as nn


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, device, use_amp: bool):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_correct, total, total_loss = 0, 0, 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(X)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * X.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total += X.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    total_correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total += X.size(0)
    return total_correct / total
