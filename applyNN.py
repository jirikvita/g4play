#!/usr/bin/env python3
"""Apply a trained PID NN model to a ROOT file and write per-event predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import uproot
from torch import nn

from pid_features import build_matrix_for_feature_names, extract_event_features, pdg_name


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


def default_output_path(input_root: str) -> str:
    p = Path(input_root)
    return str(p.with_name(f"{p.stem}_nnPID.root"))


def save_energy_histogram(pred_energy_mev: np.ndarray, out_root_path: str, bins: int) -> None:
    out = Path(out_root_path)
    csv_path = out.with_suffix(".energy_hist.csv")
    png_path = out.with_suffix(".energy_hist.png")
    pdf_path = out.with_suffix(".energy_hist.pdf")

    counts, edges = np.histogram(pred_energy_mev, bins=bins)
    mean_mev = float(np.mean(pred_energy_mev))
    rms_mev = float(np.std(pred_energy_mev))
    with csv_path.open("w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["stat", "value"])
        w.writerow(["mean_MeV", mean_mev])
        w.writerow(["rms_MeV", rms_mev])
        w.writerow([])
        w.writerow(["bin_low_MeV", "bin_high_MeV", "count"])
        for i, c in enumerate(counts):
            w.writerow([float(edges[i]), float(edges[i + 1]), int(c)])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"Failed to import matplotlib for energy histogram PNG: {exc}") from exc

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stairs(counts, edges, fill=True, alpha=0.8)
    ax.set_xlabel("Predicted energy [MeV]")
    ax.set_ylabel("Events")
    ax.set_title("Inferred energy distribution")
    ax.text(
        0.98,
        0.98,
        f"Mean = {mean_mev:.3f} MeV\nRMS = {rms_mev:.3f} MeV",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved inferred-energy histogram: {csv_path}, {png_path}, {pdf_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to model .pt saved by trainNN.py")
    p.add_argument("--input", required=True, help="Input ROOT file")
    p.add_argument("--output", default=None, help="Output ROOT file (default: <input>_nnPID.root)")
    p.add_argument("--energy-hist-bins", type=int, default=120, help="Number of bins for inferred-energy histogram")
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    feature_names = ckpt.get("feature_names")
    if feature_names is None:
        # Older checkpoints lack names; the sorted names depend only on the tree layout.
        _, _, feature_names, _ = extract_event_features(args.input)
        if len(feature_names) != int(ckpt["input_dim"]):
            raise SystemExit(
                f"Checkpoint has no feature_names and input yields {len(feature_names)} "
                f"features, expected {ckpt['input_dim']}; retrain with the current trainNN.py."
            )
        print("warning: checkpoint has no feature_names; using names derived from the input file")
    x_event_ids, x, _truth_pdg = build_matrix_for_feature_names(args.input, feature_names)
    mean = np.asarray(ckpt["input_mean"], dtype=np.float32)
    std = np.asarray(ckpt["input_std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    x = (x - mean) / std

    model = MLP(
        in_dim=int(ckpt["input_dim"]),
        hidden=list(ckpt["hidden"]),
        out_dim=int(ckpt.get("output_dim", len(ckpt["classes"]))),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    classes = list(ckpt["classes"])
    n_classes = len(classes)
    predict_energy = bool(ckpt.get("predict_energy", False))

    with torch.no_grad():
        out = model(torch.from_numpy(x))
        logits = out[:, :n_classes]
        probs = torch.softmax(logits, dim=1).numpy()
    pred = np.argmax(probs, axis=1).astype(np.int32)

    out_path = args.output or default_output_path(args.input)
    branches = {
        "eventID": x_event_ids.astype(np.int32),
        "pred_class": pred,
        "pred_confidence": np.max(probs, axis=1).astype(np.float32),
    }
    for i, cls in enumerate(classes):
        branches[f"prob_{cls}"] = probs[:, i].astype(np.float32)
    pred_energy_mev = None
    if predict_energy:
        if "energy_mean" not in ckpt or "energy_std" not in ckpt:
            raise SystemExit("Checkpoint requests energy prediction but misses energy_mean/energy_std.")
        e_mean = float(ckpt["energy_mean"])
        e_std = float(ckpt["energy_std"])
        pred_energy_mev = (out[:, n_classes].numpy() * e_std + e_mean).astype(np.float32)
        branches["pred_energy_MeV"] = pred_energy_mev

    with uproot.recreate(out_path) as f:
        f["nnPID"] = branches
    print(f"wrote predictions to {out_path}")
    for i, cls in enumerate(classes):
        print(f"  {int(cls):>4} ({pdg_name(int(cls))}): {(pred == i).sum()} events")
    if pred_energy_mev is not None:
        if args.energy_hist_bins <= 0:
            raise SystemExit("--energy-hist-bins must be > 0.")
        save_energy_histogram(pred_energy_mev, out_path, args.energy_hist_bins)


if __name__ == "__main__":
    main()
