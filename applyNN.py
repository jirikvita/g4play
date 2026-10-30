#!/usr/bin/env python3
"""Apply a trained PID NN model to a ROOT file and write per-event predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import uproot
from torch import nn

from pid_features import build_matrix_for_feature_names


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to model .pt saved by trainNN.py")
    p.add_argument("--input", required=True, help="Input ROOT file")
    p.add_argument("--output", default=None, help="Output ROOT file (default: <input>_nnPID.root)")
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    feature_names = ckpt["feature_names"]
    x_event_ids, x = build_matrix_for_feature_names(args.input, feature_names)
    mean = np.asarray(ckpt["input_mean"], dtype=np.float32)
    std = np.asarray(ckpt["input_std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    x = (x - mean) / std

    model = MLP(
        in_dim=int(ckpt["input_dim"]),
        hidden=list(ckpt["hidden"]),
        out_dim=len(ckpt["classes"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probs = torch.softmax(logits, dim=1).numpy()
    pred = np.argmax(probs, axis=1).astype(np.int32)

    out_path = args.output or default_output_path(args.input)
    branches = {
        "eventID": x_event_ids.astype(np.int32),
        "pred_class": pred,
        "pred_confidence": np.max(probs, axis=1).astype(np.float32),
    }
    classes = list(ckpt["classes"])
    for i, cls in enumerate(classes):
        branches[f"prob_{cls}"] = probs[:, i].astype(np.float32)

    with uproot.recreate(out_path) as f:
        f["nnPID"] = branches
    print(f"wrote predictions to {out_path}")
    for i, cls in enumerate(classes):
        print(f"  {cls:>2}: {(pred == i).sum()} events")


if __name__ == "__main__":
    main()
