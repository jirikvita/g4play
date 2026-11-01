#!/usr/bin/env python3
"""Train a tunable PyTorch NN for mu/pi/e PID + primary energy from ROOT events."""

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
MASS_MEV_BY_ABS_PDG = {
    11: 0.51099895,   # e-/e+
    13: 105.6583755,  # mu-/mu+
    211: 139.57039,   # pi-/pi+
}


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    energy_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_energy_mae = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    with torch.no_grad():
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            out = model(xb)
            logits = out[:, :n_classes]
            e_pred = out[:, n_classes]
            b = xb.size(0)
            loss = ce(logits, yb) + energy_loss_weight * mse(e_pred, eb)
            total_loss += float(loss.item()) * b
            total_acc += accuracy(logits, yb) * b
            total_energy_mae += float(torch.mean(torch.abs(e_pred - eb)).item()) * b
            n += b
    return {
        "loss": total_loss / max(1, n),
        "acc": total_acc / max(1, n),
        "energy_mae_norm": total_energy_mae / max(1, n),
    }


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, truth, energy_truth, energy_pred = [], [], [], []
    with torch.no_grad():
        for xb, yb, eb in loader:
            out = model(xb.to(device))
            logits = out[:, :n_classes]
            preds.append(logits.argmax(dim=1).cpu().numpy())
            truth.append(yb.numpy())
            energy_truth.append(eb.numpy())
            energy_pred.append(out[:, n_classes].cpu().numpy())
    return (
        np.concatenate(truth),
        np.concatenate(preds),
        np.concatenate(energy_truth),
        np.concatenate(energy_pred),
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


def _format_true_energies_for_label(energies_mev: np.ndarray) -> str:
    uniq = np.unique(np.round(energies_mev.astype(np.float64), 2))
    if len(uniq) <= 10:
        return ", ".join(f"{v:.2f}" for v in uniq)
    return ", ".join(f"{v:.2f}" for v in uniq[:10]) + f", ... ({len(uniq)} values)"


def plot_inferred_energy_overlay(
    pred_energy_mev: Dict[str, np.ndarray],
    true_energy_mev: Dict[str, np.ndarray],
    png_path: Path,
    pdf_path: Path,
    bins: int = 160,
) -> None:
    all_pred = np.concatenate([pred_energy_mev[k] for k in sorted(pred_energy_mev)])
    edges = np.histogram_bin_edges(all_pred, bins=bins)
    fig, ax = plt.subplots(figsize=(10, 5))

    styles = {
        "train": ("tab:blue", "train inferred"),
        "test": ("tab:orange", "test inferred"),
    }
    for name in ("train", "test"):
        if name not in pred_energy_mev:
            continue
        color, label = styles[name]
        mean_mev = float(np.mean(pred_energy_mev[name]))
        std_mev = float(np.std(pred_energy_mev[name]))
        ax.hist(
            pred_energy_mev[name],
            bins=edges,
            histtype="stepfilled",
            alpha=0.3,
            color=color,
            label=f"{label} (mean={mean_mev:.2f}, std={std_mev:.2f} MeV)",
        )
        for e_true in np.unique(np.round(true_energy_mev[name].astype(np.float64), 2)):
            ax.axvline(float(e_true), color=color, alpha=0.25, lw=1.0, ls="--")

    text = (
        f"train true E [MeV]: {_format_true_energies_for_label(true_energy_mev['train'])}\n"
        f"test true E [MeV]: {_format_true_energies_for_label(true_energy_mev['test'])}"
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
    ax.set_xlabel("Inferred energy [MeV]")
    ax.set_ylabel("Events")
    ax.set_title("Inferred energy overlay (train vs test)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_dataset(paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return (features, truth primaryPDG, truth energy [MeV], feature names)."""
    x_parts, pdg_parts, e_parts = [], [], []
    feature_names_ref = None
    for path in paths:
        _event_ids, x, feature_names, pdg = extract_event_features(path)
        p_mev = parse_beam_momentum_mev(path)
        e_parts.append(energy_from_momentum_and_pdg(p_mev, pdg))
        if feature_names_ref is None:
            feature_names_ref = feature_names
        elif feature_names != feature_names_ref:
            raise RuntimeError("Feature schema mismatch between ROOT files.")
        x_parts.append(x)
        pdg_parts.append(pdg)
    return (
        np.vstack(x_parts),
        np.concatenate(pdg_parts),
        np.concatenate(e_parts).astype(np.float32),
        feature_names_ref or [],
    )


def discover_default_inputs() -> List[str]:
    paths = sorted(Path(".").glob(DEFAULT_INPUT_PATTERN))
    selected = [str(p) for p in paths if re.match(r"^events_(?:e|mu|pi)\w*_10k\.root$", p.name)]
    if not selected:
        raise RuntimeError(
            "No default inputs found. Expected e/mu/pi files matching "
            f"{DEFAULT_INPUT_PATTERN} in the current directory."
        )
    return selected


def parse_beam_momentum_mev(path: str) -> float:
    m = re.search(r"_p([0-9]+(?:\.[0-9]+)?)MeV_", Path(path).name)
    if m is None:
        raise RuntimeError(f"Could not parse beam momentum from file name: {path}")
    return float(m.group(1))


def energy_from_momentum_and_pdg(momentum_mev: float, pdg: np.ndarray) -> np.ndarray:
    masses = np.asarray([MASS_MEV_BY_ABS_PDG.get(abs(int(v)), np.nan) for v in pdg], dtype=np.float64)
    if np.isnan(masses).any():
        bad = sorted({int(v) for v in pdg if abs(int(v)) not in MASS_MEV_BY_ABS_PDG})
        raise RuntimeError(
            f"Cannot derive energy for PDG values {bad}; only e/mu/pi are supported."
        )
    return np.sqrt(momentum_mev**2 + masses**2).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs", nargs="+", default=None,
        help=(
            "ROOT files. Default: all e/mu/pi files matching events_*_10k.root. "
            "Class labels are primaryPDG; energy target is derived from file momentum."
        ),
    )
    p.add_argument("--hidden", default="256,128", help="Comma-separated hidden layer sizes")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--energy-loss-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--model-out", default="pid_model.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    inputs = args.inputs if args.inputs is not None else discover_default_inputs()
    print("training inputs:")
    for path in inputs:
        print(f"  - {path}")

    x, pdg, energy_mev, feature_names = build_dataset(inputs)
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

    x_train, y_train, e_train = x[train_idx], y[train_idx], energy_mev[train_idx]
    x_val, y_val, e_val = x[val_idx], y[val_idx], energy_mev[val_idx]
    x_test, y_test, e_test = x[test_idx], y[test_idx], energy_mev[test_idx]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    energy_mean = float(e_train.mean())
    energy_std = float(e_train.std())
    if energy_std < 1e-8:
        energy_std = 1.0
    e_train_n = ((e_train - energy_mean) / energy_std).astype(np.float32)
    e_val_n = ((e_val - energy_mean) / energy_std).astype(np.float32)
    e_test_n = ((e_test - energy_mean) / energy_std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(e_train_n))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val), torch.from_numpy(e_val_n))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(e_test_n))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    hidden = parse_hidden(args.hidden)
    model = MLP(x.shape[1], hidden, len(classes) + 1, args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    best = {"val_acc": -1.0, "state": None, "epoch": -1}
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "train_energy_mae_norm": [],
        "val_loss": [], "val_acc": [], "val_energy_mae_norm": [],
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, eb in train_dl:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            opt.zero_grad()
            out = model(xb)
            logits = out[:, :len(classes)]
            e_pred = out[:, len(classes)]
            loss = ce(logits, yb) + args.energy_loss_weight * mse(e_pred, eb)
            loss.backward()
            opt.step()
        val_metrics = evaluate(model, val_dl, device, len(classes), args.energy_loss_weight)
        tr = evaluate(model, train_dl, device, len(classes), args.energy_loss_weight)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["train_energy_mae_norm"].append(tr["energy_mae_norm"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_energy_mae_norm"].append(val_metrics["energy_mae_norm"])
        if val_metrics["acc"] > best["val_acc"]:
            best["val_acc"] = val_metrics["acc"]
            best["state"] = copy.deepcopy(model.state_dict())
            best["epoch"] = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:3d} | "
                f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
                f"val E_MAE(norm) {val_metrics['energy_mae_norm']:.4f}"
            )

    if best["state"] is None:
        raise RuntimeError("Training failed to produce a model state.")
    model.load_state_dict(best["state"])
    test_metrics = evaluate(model, test_dl, device, len(classes), args.energy_loss_weight)
    print(
        f"best epoch {best['epoch']} | best val acc {best['val_acc']:.4f} | "
        f"test loss {test_metrics['loss']:.4f} acc {test_metrics['acc']:.4f} | "
        f"test E_MAE(norm) {test_metrics['energy_mae_norm']:.4f}"
    )

    model_out = Path(args.model_out)

    cms = {}
    rates = {}
    energy_mae_mev = {}
    inferred_energy_mev: Dict[str, np.ndarray] = {}
    true_energy_by_split_mev: Dict[str, np.ndarray] = {}
    for name, dl in (("train", train_dl), ("test", test_dl)):
        yt, yp, et_n, ep_n = predict(model, dl, device, len(classes))
        cms[name] = confusion_matrix(yt, yp, len(classes))
        rates[name] = tpr_fpr(cms[name], classes)
        et = et_n * energy_std + energy_mean
        ep = ep_n * energy_std + energy_mean
        true_energy_by_split_mev[name] = et.astype(np.float32)
        inferred_energy_mev[name] = ep.astype(np.float32)
        energy_mae_mev[name] = float(np.mean(np.abs(ep - et)))
        print(
            f"{name} confusion matrix (rows=true, cols=pred; "
            f"{[pdg_label(c) for c in classes]}):\n{cms[name]}"
        )
        print(f"  {name} energy MAE: {energy_mae_mev[name]:.4f} MeV")
        for cls, r in rates[name].items():
            print(f"  {name} {pdg_label(cls)}: TPR {r['tpr']:.4f} FPR {r['fpr']:.4f}")

    history_plot = model_out.with_suffix(".history.png")
    results_plot = model_out.with_suffix(".results.png")
    energy_overlay_png = model_out.with_suffix(".energy_overlay.png")
    energy_overlay_pdf = model_out.with_suffix(".energy_overlay.pdf")
    plot_history(history, best["epoch"], history_plot)
    plot_results(cms, classes, results_plot)
    plot_inferred_energy_overlay(
        inferred_energy_mev,
        true_energy_by_split_mev,
        energy_overlay_png,
        energy_overlay_pdf,
    )
    print(f"saved plots: {history_plot}, {results_plot}, {energy_overlay_png}, {energy_overlay_pdf}")

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": x.shape[1],
        "hidden": hidden,
        "dropout": args.dropout,
        "output_dim": len(classes) + 1,
        "classes": classes,
        "predict_energy": True,
        "target_energy_kind": "total_MeV_from_filename_momentum",
        "energy_loss_weight": float(args.energy_loss_weight),
        "energy_mean": np.float32(energy_mean),
        "energy_std": np.float32(energy_std),
        "feature_names": list(feature_names),
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
        "test_energy_mae_norm": float(test_metrics["energy_mae_norm"]),
        "test_energy_mae_mev": float(energy_mae_mev["test"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "classes": classes,
        "energy_mean_mev": float(energy_mean),
        "energy_std_mev": float(energy_std),
        "energy_mae_mev": energy_mae_mev,
        "confusion_matrix": {k: v.tolist() for k, v in cms.items()},
        "tpr_fpr": rates,
        "history": history,
    }
    metrics_path = model_out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
