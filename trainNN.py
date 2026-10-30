#!/usr/bin/env python3
"""Train a tunable PyTorch NN for mu/pi/e PID from ROOT events."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pid_features import extract_event_features, pdg_label

DEFAULT_INPUTS = [
    "events_muPos_E800MeV_1k.root",
    "events_piPos_E800MeV_1k.root",
    "events_ePos_E800MeV_1k.root",
]


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


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
            truth.append(yb.numpy())
    return np.concatenate(truth), np.concatenate(preds)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def tpr_fpr(cm: np.ndarray, classes: Sequence[int]) -> Dict[int, Dict[str, float]]:
    """One-vs-rest TPR and FPR per PDG code from a confusion matrix (rows=true)."""
    total = cm.sum()
    out = {}
    for i, pdg in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out[int(pdg)] = {
            "tpr": float(tp / max(1, tp + fn)),
            "fpr": float(fp / max(1, fp + tn)),
        }
    return out


def plot_history(history: Dict[str, List[float]], best_epoch: int, path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))
    ax_l.plot(epochs, history["train_loss"], label="train")
    ax_l.plot(epochs, history["val_loss"], label="validation")
    ax_l.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss evolution", yscale="log")
    ax_a.plot(epochs, history["train_acc"], label="train")
    ax_a.plot(epochs, history["val_acc"], label="validation")
    ax_a.set(xlabel="epoch", ylabel="accuracy", title="Accuracy evolution")
    for ax in (ax_l, ax_a):
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best epoch {best_epoch}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_results(cms: Dict[str, np.ndarray], classes: Sequence[int], path: Path) -> None:
    """Row per dataset: normalized confusion matrix and per-class TPR/FPR bars."""
    fig, axes = plt.subplots(len(cms), 2, figsize=(11, 4.5 * len(cms)), squeeze=False)
    n = len(classes)
    labels = [pdg_label(c) for c in classes]
    for r, (name, cm) in enumerate(cms.items()):
        ax = axes[r, 0]
        norm = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{cm[i, j]}\n{norm[i, j]:.3f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black",
                )
        ax.set(xticks=range(n), yticks=range(n), xticklabels=labels,
               yticklabels=labels, xlabel="predicted", ylabel="true (primaryPDG)",
               title=f"{name} confusion matrix")
        fig.colorbar(im, ax=ax, fraction=0.046)

        rates = tpr_fpr(cm, classes)
        ax = axes[r, 1]
        xs = np.arange(n)
        tpr = [rates[c]["tpr"] for c in classes]
        fpr = [rates[c]["fpr"] for c in classes]
        b1 = ax.bar(xs - 0.2, tpr, 0.4, label="TPR")
        b2 = ax.bar(xs + 0.2, fpr, 0.4, label="FPR")
        ax.bar_label(b1, fmt="%.3f", fontsize=8)
        ax.bar_label(b2, fmt="%.3f", fontsize=8)
        ax.set(xticks=xs, xticklabels=labels, ylim=(0, 1.1),
               title=f"{name} TPR / FPR (one-vs-rest)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_dataset(paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (features, truth primaryPDG per event, feature names) over all files."""
    x_parts, pdg_parts = [], []
    feature_names_ref = None
    for path in paths:
        _event_ids, x, feature_names, pdg = extract_event_features(path)
        if feature_names_ref is None:
            feature_names_ref = feature_names
        elif feature_names != feature_names_ref:
            raise RuntimeError("Feature schema mismatch between ROOT files.")
        x_parts.append(x)
        pdg_parts.append(pdg)
    return np.vstack(x_parts), np.concatenate(pdg_parts), feature_names_ref or []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs", nargs="+", default=DEFAULT_INPUTS,
        help="ROOT files; class labels are the per-event primaryPDG values in the 'silicon' tree",
    )
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

    x, pdg, feature_names = build_dataset(args.inputs)
    # Output neuron i <-> classes[i] (PDG code); the dict maps truth PDG -> class index.
    classes = [int(c) for c in np.unique(pdg)]
    pdg_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([pdg_to_idx[int(c)] for c in pdg], dtype=np.int64)
    print("classes:", ", ".join(f"{pdg_label(c)}: {int((pdg == c).sum())} events" for c in classes))
    if len(classes) < 2:
        raise RuntimeError("Need at least two distinct primaryPDG values to train.")

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
    model = MLP(x.shape[1], hidden, len(classes), args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best = {"val_acc": -1.0, "state": None, "epoch": -1}
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
        val_metrics = evaluate(model, val_dl, device)
        tr = evaluate(model, train_dl, device)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        if val_metrics["acc"] > best["val_acc"]:
            best["val_acc"] = val_metrics["acc"]
            best["state"] = copy.deepcopy(model.state_dict())
            best["epoch"] = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
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

    cms = {}
    rates = {}
    for name, dl in (("train", train_dl), ("test", test_dl)):
        yt, yp = predict(model, dl, device)
        cms[name] = confusion_matrix(yt, yp, len(classes))
        rates[name] = tpr_fpr(cms[name], classes)
        print(
            f"{name} confusion matrix (rows=true, cols=pred; "
            f"{[pdg_label(c) for c in classes]}):\n{cms[name]}"
        )
        for cls, r in rates[name].items():
            print(f"  {name} {pdg_label(cls)}: TPR {r['tpr']:.4f} FPR {r['fpr']:.4f}")

    history_plot = model_out.with_suffix(".history.png")
    results_plot = model_out.with_suffix(".results.png")
    plot_history(history, best["epoch"], history_plot)
    plot_results(cms, classes, results_plot)
    print(f"saved plots: {history_plot}, {results_plot}")

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": x.shape[1],
        "hidden": hidden,
        "dropout": args.dropout,
        "classes": classes,
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
        "classes": classes,
        "confusion_matrix": {k: v.tolist() for k, v in cms.items()},
        "tpr_fpr": rates,
        "history": history,
    }
    metrics_path = model_out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
