#!/usr/bin/env python3
"""Train a tunable PyTorch NN for mu/pi/e PID + primary momentum from ROOT events."""

from __future__ import annotations

import argparse
import copy
import json
import re
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

DEFAULT_INPUT_PATTERN = "train_test/events_*_10k.root"


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


def format_epoch_progress(epoch: int, total: int, width: int = 30) -> str:
    total = max(1, total)
    ratio = max(0.0, min(1.0, epoch / total))
    done = int(round(width * ratio))
    bar = "#" * done + "-" * (width - done)
    return f"[{bar}] {epoch}/{total}"


def hidden_signature(hidden: Sequence[int]) -> str:
    return "h" + "x".join(str(int(h)) for h in hidden)


def model_out_with_hidden(path: Path, hidden: Sequence[int]) -> Path:
    token = hidden_signature(hidden)
    stem = path.stem
    if re.search(r"(?:^|[_-])h\d+(?:x\d+)*$", stem):
        return path
    suffix = path.suffix or ".pt"
    return path.with_name(f"{stem}_{token}{suffix}")


def model_out_in_models_dir(path: Path) -> Path:
    models_dir = Path("models")
    if path.parts and path.parts[0] == "models":
        return path
    return models_dir / path.name


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    momentum_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_momentum_mae = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    with torch.no_grad():
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            out = model(xb)
            logits = out[:, :n_classes]
            p_pred = out[:, n_classes]
            b = xb.size(0)
            loss = ce(logits, yb) + momentum_loss_weight * mse(p_pred, eb)
            total_loss += float(loss.item()) * b
            total_acc += accuracy(logits, yb) * b
            total_momentum_mae += float(torch.mean(torch.abs(p_pred - eb)).item()) * b
            n += b
    return {
        "loss": total_loss / max(1, n),
        "acc": total_acc / max(1, n),
        "momentum_mae_norm": total_momentum_mae / max(1, n),
    }


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, truth, momentum_truth, momentum_pred = [], [], [], []
    with torch.no_grad():
        for xb, yb, eb in loader:
            out = model(xb.to(device))
            logits = out[:, :n_classes]
            preds.append(logits.argmax(dim=1).cpu().numpy())
            truth.append(yb.numpy())
            momentum_truth.append(eb.numpy())
            momentum_pred.append(out[:, n_classes].cpu().numpy())
    return (
        np.concatenate(truth),
        np.concatenate(preds),
        np.concatenate(momentum_truth),
        np.concatenate(momentum_pred),
    )


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
    ax_l.set(xlabel="epoch", ylabel="combined loss", title="Loss evolution", yscale="log")
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


def _format_true_values_for_label(values: np.ndarray) -> str:
    uniq = np.unique(np.round(values.astype(np.float64), 2))
    if len(uniq) <= 10:
        return ", ".join(f"{v:.2f}" for v in uniq)
    return ", ".join(f"{v:.2f}" for v in uniq[:10]) + f", ... ({len(uniq)} values)"


def plot_inferred_momentum_overlay(
    pred_momentum_mev_c: Dict[str, np.ndarray],
    true_momentum_mev_c: Dict[str, np.ndarray],
    png_path: Path,
    pdf_path: Path,
    bins: int = 320,
) -> None:
    all_pred = np.concatenate([pred_momentum_mev_c[k] for k in sorted(pred_momentum_mev_c)])
    edges = np.histogram_bin_edges(all_pred, bins=bins)
    fig, (ax, ax_leg) = plt.subplots(
        2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2]}
    )

    styles = {
        "train": ("tab:blue", "train inferred"),
        "test": ("tab:orange", "test inferred"),
    }
    for name in ("train", "test"):
        if name not in pred_momentum_mev_c:
            continue
        color, label = styles[name]
        mean_mev_c = float(np.mean(pred_momentum_mev_c[name]))
        std_mev_c = float(np.std(pred_momentum_mev_c[name]))
        ax.hist(
            pred_momentum_mev_c[name],
            bins=edges,
            histtype="stepfilled",
            alpha=0.3,
            color=color,
            label=f"{label} (mean={mean_mev_c:.2f}, std={std_mev_c:.2f} MeV/c)",
        )
        for p_true in np.unique(np.round(true_momentum_mev_c[name].astype(np.float64), 2)):
            ax.axvline(float(p_true), color=color, alpha=0.25, lw=1.0, ls="--")

    text = (
        f"train true p [MeV/c]: {_format_true_values_for_label(true_momentum_mev_c['train'])}\n"
        f"test true p [MeV/c]: {_format_true_values_for_label(true_momentum_mev_c['test'])}"
    )
    ax.text(
        0.99,
        0.99,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    ax.set_xlabel("Inferred momentum [MeV/c]")
    ax.set_ylabel("Events")
    ax.set_title("Inferred momentum overlay (train vs test)")
    ax.set_xlim(0.0, 1000.0)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax_leg.axis("off")
    ax_leg.legend(handles, labels, fontsize=10, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_dataset(paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return (features, truth primaryPDG, truth momentum [MeV/c], feature names)."""
    x_parts, pdg_parts, p_parts = [], [], []
    feature_names_ref = None
    for path in paths:
        _event_ids, x, feature_names, pdg = extract_event_features(path)
        p_mev = parse_beam_momentum_mev(path)
        p_parts.append(np.full(len(pdg), p_mev, dtype=np.float32))
        if feature_names_ref is None:
            feature_names_ref = feature_names
        elif feature_names != feature_names_ref:
            raise RuntimeError("Feature schema mismatch between ROOT files.")
        x_parts.append(x)
        pdg_parts.append(pdg)
    return (
        np.vstack(x_parts),
        np.concatenate(pdg_parts),
        np.concatenate(p_parts).astype(np.float32),
        feature_names_ref or [],
    )


def discover_default_inputs() -> List[str]:
    paths = sorted(Path(".").glob(DEFAULT_INPUT_PATTERN))
    selected = [
        str(p)
        for p in paths
        if re.match(r"^events_(?:(?:e|mu|pi|pbar|p|k|K)\w*)_10k\.root$", p.name)
    ]
    if not selected:
        raise RuntimeError(
            "No default inputs found. Expected particle files matching "
            f"{DEFAULT_INPUT_PATTERN} in the current directory."
        )
    return selected


def parse_beam_momentum_mev(path: str) -> float:
    m = re.search(r"_p([0-9]+(?:\.[0-9]+)?)MeV_", Path(path).name)
    if m is None:
        raise RuntimeError(f"Could not parse beam momentum from file name: {path}")
    return float(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs", nargs="+", default=None,
        help=(
            "ROOT files. Default: all supported particle files matching events_*_10k.root. "
            "Class labels are primaryPDG; momentum target is derived from file name."
        ),
    )
    p.add_argument("--hidden", default="256,128,128", help="Comma-separated hidden layer sizes")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum-loss-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--model-out",
        default="models/pid_model.pt",
        help=(
            "Base output model path under models/. Hidden-layer sizes are appended "
            "automatically (example: models/pid_model_h256x128.pt)."
        ),
    )
    args = p.parse_args()

    print("initializing random seeds...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("discovering input ROOT files...")
    inputs = args.inputs if args.inputs is not None else discover_default_inputs()
    print("training inputs:")
    for path in inputs:
        print(f"  - {path}")

    print("reading ROOT inputs and building feature dataset...")
    x, pdg, momentum_mev_c, feature_names = build_dataset(inputs)
    print(
        f"dataset loaded: events={len(pdg)}, features={x.shape[1]}, "
        f"momentum_targets={len(momentum_mev_c)}"
    )
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
    print("completed stratified split (train+val/test, then train/val)")

    x_train, y_train, p_train = x[train_idx], y[train_idx], momentum_mev_c[train_idx]
    x_val, y_val, p_val = x[val_idx], y[val_idx], momentum_mev_c[val_idx]
    x_test, y_test, p_test = x[test_idx], y[test_idx], momentum_mev_c[test_idx]

    print("normalizing input features from training split statistics...")
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    print("normalizing momentum targets from training split statistics...")
    momentum_mean = float(p_train.mean())
    momentum_std = float(p_train.std())
    if momentum_std < 1e-8:
        momentum_std = 1.0
    p_train_n = ((p_train - momentum_mean) / momentum_std).astype(np.float32)
    p_val_n = ((p_val - momentum_mean) / momentum_std).astype(np.float32)
    p_test_n = ((p_test - momentum_mean) / momentum_std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(p_train_n))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val), torch.from_numpy(p_val_n))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(p_test_n))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print("built DataLoaders for train/validation/test")

    hidden = parse_hidden(args.hidden)
    print(f"parsed hidden-layer layout: {hidden}")
    requested_model_out = model_out_in_models_dir(Path(args.model_out))
    model_out = model_out_with_hidden(requested_model_out, hidden)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"resolved model output path: {model_out}")
    if model_out != requested_model_out:
        print(f"model output renamed to encode hidden sizes: {requested_model_out} -> {model_out}")
    write_compat_alias = requested_model_out.name == "pid_model.pt" and model_out != requested_model_out

    print("building model and optimizer...")
    model = MLP(x.shape[1], hidden, len(classes) + 1, args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())

    print("training hyperparameters:")
    print(f"  model_out: {model_out}")
    print(f"  hidden: {hidden}")
    print(f"  dropout: {args.dropout}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  lr: {args.lr}")
    print(f"  momentum_loss_weight: {args.momentum_loss_weight}")
    print(f"  seed: {args.seed}")
    print(f"  device: {device}")
    print(f"  train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"  n_parameters: {n_params}")
    print("starting training loop...")

    best = {"val_acc": -1.0, "state": None, "epoch": -1}
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "train_momentum_mae_norm": [],
        "val_loss": [], "val_acc": [], "val_momentum_mae_norm": [],
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, eb in train_dl:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            opt.zero_grad()
            out = model(xb)
            logits = out[:, :len(classes)]
            p_pred = out[:, len(classes)]
            loss = ce(logits, yb) + args.momentum_loss_weight * mse(p_pred, eb)
            loss.backward()
            opt.step()
        val_metrics = evaluate(model, val_dl, device, len(classes), args.momentum_loss_weight)
        tr = evaluate(model, train_dl, device, len(classes), args.momentum_loss_weight)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["train_momentum_mae_norm"].append(tr["momentum_mae_norm"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_momentum_mae_norm"].append(val_metrics["momentum_mae_norm"])
        if val_metrics["acc"] > best["val_acc"]:
            best["val_acc"] = val_metrics["acc"]
            best["state"] = copy.deepcopy(model.state_dict())
            best["epoch"] = epoch
        progress = format_epoch_progress(epoch, args.epochs)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"{progress} | "
                f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
                f"val p_MAE(norm) {val_metrics['momentum_mae_norm']:.4f}"
            )
        else:
            print(f"\r{progress}", end="", flush=True)

    print()

    if best["state"] is None:
        raise RuntimeError("Training failed to produce a model state.")
    print("loading best validation checkpoint and evaluating on test split...")
    model.load_state_dict(best["state"])
    test_metrics = evaluate(model, test_dl, device, len(classes), args.momentum_loss_weight)
    print(
        f"best epoch {best['epoch']} | best val acc {best['val_acc']:.4f} | "
        f"test loss {test_metrics['loss']:.4f} acc {test_metrics['acc']:.4f} | "
        f"test p_MAE(norm) {test_metrics['momentum_mae_norm']:.4f}"
    )

    cms = {}
    rates = {}
    momentum_mae_mev_c = {}
    inferred_momentum_mev_c: Dict[str, np.ndarray] = {}
    true_momentum_by_split_mev_c: Dict[str, np.ndarray] = {}
    for name, dl in (("train", train_dl), ("test", test_dl)):
        print(f"computing predictions and metrics for {name} split...")
        yt, yp, et_n, ep_n = predict(model, dl, device, len(classes))
        cms[name] = confusion_matrix(yt, yp, len(classes))
        rates[name] = tpr_fpr(cms[name], classes)
        pt = et_n * momentum_std + momentum_mean
        pp = ep_n * momentum_std + momentum_mean
        true_momentum_by_split_mev_c[name] = pt.astype(np.float32)
        inferred_momentum_mev_c[name] = pp.astype(np.float32)
        momentum_mae_mev_c[name] = float(np.mean(np.abs(pp - pt)))
        print(
            f"{name} confusion matrix (rows=true, cols=pred; "
            f"{[pdg_label(c) for c in classes]}):\n{cms[name]}"
        )
        print(f"  {name} momentum MAE: {momentum_mae_mev_c[name]:.4f} MeV/c")
        for cls, r in rates[name].items():
            print(f"  {name} {pdg_label(cls)}: TPR {r['tpr']:.4f} FPR {r['fpr']:.4f}")

    history_plot = model_out.with_suffix(".history.png")
    results_plot = model_out.with_suffix(".results.png")
    momentum_overlay_png = model_out.with_suffix(".momentum_overlay.png")
    momentum_overlay_pdf = model_out.with_suffix(".momentum_overlay.pdf")
    plot_history(history, best["epoch"], history_plot)
    plot_results(cms, classes, results_plot)
    plot_inferred_momentum_overlay(
        inferred_momentum_mev_c,
        true_momentum_by_split_mev_c,
        momentum_overlay_png,
        momentum_overlay_pdf,
    )
    print("finished plot generation")
    print(f"saved plots: {history_plot}, {results_plot}, {momentum_overlay_png}, {momentum_overlay_pdf}")

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": x.shape[1],
        "hidden": hidden,
        "dropout": args.dropout,
        "output_dim": len(classes) + 1,
        "classes": classes,
        "predict_momentum": True,
        "target_momentum_kind": "p_MeV_c_from_filename",
        "momentum_loss_weight": float(args.momentum_loss_weight),
        "momentum_mean": np.float32(momentum_mean),
        "momentum_std": np.float32(momentum_std),
        "feature_names": list(feature_names),
        "input_mean": mean.astype(np.float32),
        "input_std": std.astype(np.float32),
        "seed": args.seed,
    }
    torch.save(ckpt, model_out)
    print(f"saved model: {model_out}")
    if write_compat_alias:
        torch.save(ckpt, requested_model_out)
        print(f"saved compatibility model alias: {requested_model_out}")

    metrics = {
        "best_epoch": int(best["epoch"]),
        "best_val_acc": float(best["val_acc"]),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "test_momentum_mae_norm": float(test_metrics["momentum_mae_norm"]),
        "test_momentum_mae_mev_c": float(momentum_mae_mev_c["test"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "classes": classes,
        "momentum_mean_mev_c": float(momentum_mean),
        "momentum_std_mev_c": float(momentum_std),
        "momentum_mae_mev_c": momentum_mae_mev_c,
        "confusion_matrix": {k: v.tolist() for k, v in cms.items()},
        "tpr_fpr": rates,
        "history": history,
    }
    metrics_path = model_out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics: {metrics_path}")
    print("training pipeline complete")


if __name__ == "__main__":
    main()
