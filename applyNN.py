#!/usr/bin/env python3
"""Apply a trained PID NN model to a ROOT file and write per-event predictions."""

from __future__ import annotations

import argparse
import contextlib
import csv
import re
import sys
from pathlib import Path
from typing import IO, List, Sequence

import numpy as np
import torch
import uproot
from torch import nn

from pid_features import build_matrix_for_feature_names, extract_event_features, pdg_name

MASS_MEV_BY_PREFIX = {
    "e": 0.51099895,
    "mu": 105.6583755,
    "pi": 139.57039,
    "p": 938.2720813,
    "pbar": 938.2720813,
    "k": 493.677,
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


class Tee:
    def __init__(self, *streams: IO[str]) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def default_output_path(input_root: str) -> str:
    p = Path(input_root)
    return str(p.with_name(f"{p.stem}_nnPID.root"))


def mass_from_input_filename(input_root: str) -> float:
    name = Path(input_root).name
    m = re.match(r"^events_(?P<token>.+?)_p[0-9]+(?:\.[0-9]+)?MeV_", name)
    if m is None:
        raise SystemExit(f"Cannot infer particle type from filename for momentum conversion: {name}")
    token = m.group("token")
    low = token.lower()
    if low.startswith("mu"):
        return MASS_MEV_BY_PREFIX["mu"]
    if low.startswith("pi"):
        return MASS_MEV_BY_PREFIX["pi"]
    if low.startswith("pbar"):
        return MASS_MEV_BY_PREFIX["pbar"]
    if low.startswith("p"):
        return MASS_MEV_BY_PREFIX["p"]
    if low.startswith("k"):
        return MASS_MEV_BY_PREFIX["k"]
    if low.startswith("e"):
        return MASS_MEV_BY_PREFIX["e"]
    raise SystemExit(f"Unsupported particle token in filename for momentum conversion: {token}")


def save_momentum_histogram(pred_momentum_mev_c: np.ndarray, out_root_path: str, bins: int) -> None:
    out = Path(out_root_path)
    csv_path = out.with_suffix(".momentum_hist.csv")
    png_path = out.with_suffix(".momentum_hist.png")
    pdf_path = out.with_suffix(".momentum_hist.pdf")

    counts, edges = np.histogram(pred_momentum_mev_c, bins=bins)
    mean_mev_c = float(np.mean(pred_momentum_mev_c))
    rms_mev_c = float(np.std(pred_momentum_mev_c))
    with csv_path.open("w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["stat", "value"])
        w.writerow(["mean_MeV_c", mean_mev_c])
        w.writerow(["rms_MeV_c", rms_mev_c])
        w.writerow([])
        w.writerow(["bin_low_MeV_c", "bin_high_MeV_c", "count"])
        for i, c in enumerate(counts):
            w.writerow([float(edges[i]), float(edges[i + 1]), int(c)])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"Failed to import matplotlib for momentum histogram PNG: {exc}") from exc

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stairs(counts, edges, fill=True, alpha=0.8)
    ax.set_xlabel("Predicted momentum [MeV/c]")
    ax.set_ylabel("Events")
    ax.set_title("Inferred momentum distribution")
    ax.set_xlim(0.0, 1000.0)
    ax.text(
        0.98,
        0.98,
        f"Mean = {mean_mev_c:.3f} MeV/c\nRMS = {rms_mev_c:.3f} MeV/c",
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
    print(f"saved inferred-momentum histogram: {csv_path}, {png_path}, {pdf_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to model .pt saved by trainNN.py")
    p.add_argument("--input", required=True, help="Input ROOT file")
    p.add_argument("--output", default=None, help="Output ROOT file (default: <input>_nnPID.root)")
    p.add_argument(
        "--log-txt",
        default=None,
        help="Path to text log file (default: <output>.applyNN.log.txt)",
    )
    p.add_argument(
        "--momentum-hist-bins", type=int, default=240, help="Number of bins for inferred-momentum histogram"
    )
    args = p.parse_args()
    out_path = args.output or default_output_path(args.input)
    log_path = Path(args.log_txt) if args.log_txt is not None else Path(out_path).with_suffix(".applyNN.log.txt")

    with log_path.open("w", encoding="utf-8") as logf:
        tee = Tee(sys.stdout, logf)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"logging to {log_path}")

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
            predict_momentum = bool(ckpt.get("predict_momentum", False))
            predict_energy = bool(ckpt.get("predict_energy", False))

            with torch.no_grad():
                out = model(torch.from_numpy(x))
                logits = out[:, :n_classes]
                probs = torch.softmax(logits, dim=1).numpy()
            pred = np.argmax(probs, axis=1).astype(np.int32)

            branches = {
                "eventID": x_event_ids.astype(np.int32),
                "pred_class": pred,
                "pred_pdg": np.asarray([int(classes[i]) for i in pred], dtype=np.int32),
                "pred_confidence": np.max(probs, axis=1).astype(np.float32),
            }
            for i, cls in enumerate(classes):
                branches[f"prob_{cls}"] = probs[:, i].astype(np.float32)
            pred_momentum_mev_c = None
            if predict_momentum:
                if "momentum_mean" not in ckpt or "momentum_std" not in ckpt:
                    raise SystemExit("Checkpoint requests momentum prediction but misses momentum_mean/momentum_std.")
                p_mean = float(ckpt["momentum_mean"])
                p_std = float(ckpt["momentum_std"])
                pred_momentum_mev_c = (out[:, n_classes].numpy() * p_std + p_mean).astype(np.float32)
                branches["pred_momentum_MeV_c"] = pred_momentum_mev_c
            elif predict_energy:
                if "energy_mean" not in ckpt or "energy_std" not in ckpt:
                    raise SystemExit("Checkpoint requests energy prediction but misses energy_mean/energy_std.")
                e_mean = float(ckpt["energy_mean"])
                e_std = float(ckpt["energy_std"])
                pred_energy_mev = (out[:, n_classes].numpy() * e_std + e_mean).astype(np.float32)
                branches["pred_energy_MeV"] = pred_energy_mev
                mass_mev = mass_from_input_filename(args.input)
                pred_momentum_mev_c = np.sqrt(
                    np.maximum(0.0, pred_energy_mev.astype(np.float64) ** 2 - mass_mev**2)
                ).astype(np.float32)
                branches["pred_momentum_MeV_c"] = pred_momentum_mev_c

            with uproot.recreate(out_path) as f:
                f["nnPID"] = branches
            print(f"wrote predictions to {out_path}")
            for i, cls in enumerate(classes):
                print(f"  {int(cls):>4} ({pdg_name(int(cls))}): {(pred == i).sum()} events")
            if pred_momentum_mev_c is not None:
                if args.momentum_hist_bins <= 0:
                    raise SystemExit("--momentum-hist-bins must be > 0.")
                save_momentum_histogram(pred_momentum_mev_c, out_path, args.momentum_hist_bins)


if __name__ == "__main__":
    main()
