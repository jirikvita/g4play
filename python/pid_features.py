#!/usr/bin/env python3
"""Feature extraction for PID training/inference from Geant4 ROOT outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import awkward as ak
import numpy as np
import uproot


PDG_NAMES: Dict[int, str] = {
    13: "mu-", -13: "mu+", 211: "pi+", -211: "pi-", 11: "e-", -11: "e+",
    2212: "p", -2212: "anti_p", 22: "gamma", 321: "K+", -321: "K-",
}


def pdg_name(pdg: int) -> str:
    return PDG_NAMES.get(int(pdg), "pdg")


def pdg_label(pdg: int) -> str:
    return f"{pdg_name(pdg)} ({int(pdg)})"


def read_primary_pdg(silicon) -> np.ndarray:
    if "primaryPDG" not in silicon.keys():
        raise RuntimeError(
            "Tree 'silicon' has no 'primaryPDG' branch; regenerate the ROOT file "
            "with the current waterCherenkovSim."
        )
    return np.asarray(silicon["primaryPDG"].array(library="np"), dtype=np.int64)


def _to_vec(value) -> np.ndarray:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([value], dtype=np.float64)
    return np.asarray(ak.to_numpy(value), dtype=np.float64)


def _add_stats(row: Dict[str, float], prefix: str, vec: np.ndarray) -> None:
    row[f"{prefix}_count"] = float(len(vec))
    if len(vec) == 0:
        row[f"{prefix}_sum"] = 0.0
        row[f"{prefix}_mean"] = 0.0
        row[f"{prefix}_std"] = 0.0
        row[f"{prefix}_min"] = 0.0
        row[f"{prefix}_max"] = 0.0
        return
    row[f"{prefix}_sum"] = float(np.sum(vec))
    row[f"{prefix}_mean"] = float(np.mean(vec))
    row[f"{prefix}_std"] = float(np.std(vec))
    row[f"{prefix}_min"] = float(np.min(vec))
    row[f"{prefix}_max"] = float(np.max(vec))


def extract_event_features(
    root_path: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Return (event_ids, feature_matrix, feature_names, truth_pdg) for one ROOT file.

    truth_pdg is the per-event primaryPDG from the tree (-> required; never a feature).
    """
    f = uproot.open(root_path)
    silicon = f["silicon"]
    truth_pdg = read_primary_pdg(silicon)
    s = silicon.arrays(library="ak")

    event_ids = np.asarray(ak.to_numpy(s["eventID"]), dtype=np.int64)
    n_events = len(event_ids)
    rows: List[Dict[str, float]] = [dict() for _ in range(n_events)]
    event_to_row = {int(eid): i for i, eid in enumerate(event_ids)}

    # primaryPDG is the truth label and must never be used as an input feature
    silicon_branches = [b for b in silicon.keys() if b not in ("eventID", "primaryPDG")]
    for i in range(n_events):
        row = rows[i]
        for b in silicon_branches:
            _add_stats(row, f"silicon_{b}", _to_vec(s[b][i]))

        x = _to_vec(s["x_mm"][i])
        y = _to_vec(s["y_mm"][i])
        z = _to_vec(s["z_mm"][i])
        nh = _to_vec(s["nHits"][i])
        if len(nh) > 0 and np.sum(nh) > 0:
            w = nh / np.sum(nh)
            row["silicon_weighted_x_mean"] = float(np.sum(w * x))
            row["silicon_weighted_y_mean"] = float(np.sum(w * y))
            row["silicon_weighted_z_mean"] = float(np.sum(w * z))
            row["silicon_total_hits"] = float(np.sum(nh))
        else:
            row["silicon_weighted_x_mean"] = 0.0
            row["silicon_weighted_y_mean"] = 0.0
            row["silicon_weighted_z_mean"] = 0.0
            row["silicon_total_hits"] = 0.0

    muon_branches: Sequence[str] = []
    mu_by_event: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    if "muonDecay" in f:
        muon = f["muonDecay"]
        muon_branches = [b for b in muon.keys() if b != "eventID"]
        m = muon.arrays(library="np")
        mids = np.asarray(m["eventID"], dtype=np.int64)
        for j, eid in enumerate(mids):
            ievt = int(eid)
            for b in muon_branches:
                mu_by_event[ievt][b].append(float(m[b][j]))

    for i, eid in enumerate(event_ids):
        row = rows[i]
        vals = mu_by_event.get(int(eid), {})
        row["muonDecay_has_row"] = 1.0 if vals else 0.0
        for b in muon_branches:
            _add_stats(
                row,
                f"muonDecay_{b}",
                np.asarray(vals.get(b, []), dtype=np.float64),
            )

    feature_names = sorted({k for r in rows for k in r})
    x = np.zeros((n_events, len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            x[i, j] = np.float32(row.get(name, 0.0))
    return event_ids, x, feature_names, truth_pdg


def build_matrix_for_feature_names(
    root_path: str, expected_feature_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (event_ids, matrix, truth_pdg) aligned to expected_feature_names."""
    event_ids, x_raw, names, truth_pdg = extract_event_features(root_path)
    name_to_col = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(event_ids), len(expected_feature_names)), dtype=np.float32)
    for j, name in enumerate(expected_feature_names):
        if name in name_to_col:
            out[:, j] = x_raw[:, name_to_col[name]]
    return event_ids, out, truth_pdg
