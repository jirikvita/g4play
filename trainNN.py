#!/usr/bin/env python3
"""Train a tunable PyTorch NN for mu/pi/e PID from ROOT events."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pid_features import extract_event_features


CLASS_TO_LABEL = {"mu": 0, "pi": 1, "el": 2}
LABEL_TO_CLASS = ["mu", "pi", "el"]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_hidden(hidden_str: str) -> List[int]:
    return [int(x.strip()) for x in hidden_str.split(",") if x.strip()]


def stratified_split(
    y: np.ndarray, test_ratio: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    test_idx, keep_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        n_test = max(1, min(n_test, len(idx) - 1))
        test_idx.extend(idx[:n_test].tolist())
        keep_idx.extend(idx[n_test:].tolist())
    return np.asarray(sorted(keep_idx)), np.asarray(sorted(test_idx))


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            b = xb.size(0)
            total_loss += float(ce(out, yb).item()) * b
            total_acc += accuracy(out, yb) * b
            n += b
    return {"loss": total_loss / max(1, n), "acc": total_acc / max(1, n)}


def build_dataset(files: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x_parts, y_parts = [], []
    feature_names_ref = None
    for cls, path in files.items():
        _event_ids, x, feature_names = extract_event_features(path)
        if feature_names_ref is None:
            feature_names_ref = feature_names
        elif feature_names != feature_names_ref:
            raise RuntimeError("Feature schema mismatch between ROOT files.")
        x_parts.append(x)
        y_parts.append(np.full(len(x), CLASS_TO_LABEL[cls], dtype=np.int64))
    return np.vstack(x_parts), np.concatenate(y_parts), feature_names_ref or []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mu", default="events_muPos_E800MeV_1k.root")
    p.add_argument("--pi", default="events_piPos_E800MeV_1k.root")
    p.add_argument("--el", default="events_ePos_E800MeV_1k.root")
    p.add_argument("--hidden", default="256,128", help="Comma-separated hidden layer sizes")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--model-out", default="pid_model.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x, y, feature_names = build_dataset({"mu": args.mu, "pi": args.pi, "el": args.el})

    # Requested split: 1/2 for train+validation (then 90/10), 1/2 for test.
    trainval_idx, test_idx = stratified_split(y, test_ratio=0.5, rng=rng)
    y_trainval = y[trainval_idx]
    tr_idx_rel, val_idx_rel = stratified_split(y_trainval, test_ratio=0.1, rng=rng)
    train_idx = trainval_idx[tr_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    hidden = parse_hidden(args.hidden)
    model = MLP(x.shape[1], hidden, 3, args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best = {"val_acc": -1.0, "state": None, "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
        val_metrics = evaluate(model, val_dl, device)
        if val_metrics["acc"] > best["val_acc"]:
            best["val_acc"] = val_metrics["acc"]
            best["state"] = copy.deepcopy(model.state_dict())
            best["epoch"] = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            tr = evaluate(model, train_dl, device)
            print(
                f"epoch {epoch:3d} | "
                f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
            )

    if best["state"] is None:
        raise RuntimeError("Training failed to produce a model state.")
    model.load_state_dict(best["state"])
    test_metrics = evaluate(model, test_dl, device)
    print(
        f"best epoch {best['epoch']} | best val acc {best['val_acc']:.4f} | "
        f"test loss {test_metrics['loss']:.4f} acc {test_metrics['acc']:.4f}"
    )

    model_out = Path(args.model_out)
    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": x.shape[1],
        "hidden": hidden,
        "dropout": args.dropout,
        "classes": LABEL_TO_CLASS,
        "feature_names": feature_names,
        "input_mean": mean.astype(np.float32),
        "input_std": std.astype(np.float32),
        "seed": args.seed,
    }
    torch.save(ckpt, model_out)
    print(f"saved model: {model_out}")

    metrics = {
        "best_epoch": int(best["epoch"]),
        "best_val_acc": float(best["val_acc"]),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    metrics_path = model_out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
