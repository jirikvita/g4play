#!/usr/bin/env python3
"""Run applyNN.py on all non-nnPID ROOT files in a directory."""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import uproot
from pid_features import pdg_name

MASS_MEV_BY_PREFIX = {
    "e": 0.51099895,
    "mu": 105.6583755,
    "pi": 139.57039,
    "p": 938.2720813,
    "pbar": 938.2720813,
    "k": 493.677,
}


def parse_particle_and_momentum(path: Path) -> tuple[str, float]:
    name = path.name
    m = re.match(r"^events_(?P<token>.+?)_p(?P<p>[0-9]+(?:\.[0-9]+)?)MeV_", name)
    if m is None:
        raise RuntimeError(f"Unsupported ROOT filename format: {name}")
    return m.group("token"), float(m.group("p"))


def species_key_from_token(token: str) -> str:
    low = token.lower()
    if low.startswith("mu"):
        return "mu"
    if low.startswith("pi"):
        return "pi"
    if low.startswith("pbar"):
        return "pbar"
    if low.startswith("p"):
        return "p"
    if low.startswith("k"):
        return "k"
    if low.startswith("e"):
        return "e"
    raise RuntimeError(f"Unsupported particle token in filename: {token}")


def true_pdg_from_token(token: str) -> int:
    low = token.lower()
    if low.startswith("mu"):
        return 13 if ("neg" in low or "-" in token) else -13
    if low.startswith("pi"):
        return -211 if ("neg" in low or "-" in token) else 211
    if low.startswith("e"):
        return 11 if ("neg" in low or "-" in token) else -11
    if low.startswith("pbar"):
        return -2212
    if low.startswith("p"):
        return 2212
    if low.startswith("k"):
        return -321 if ("neg" in low or "-" in token) else 321
    raise RuntimeError(f"Unsupported particle token in filename: {token}")


def particle_mass_mev(token: str) -> float:
    key = species_key_from_token(token)
    if key == "mu":
        return MASS_MEV_BY_PREFIX["mu"]
    if key == "pi":
        return MASS_MEV_BY_PREFIX["pi"]
    if key == "pbar":
        return MASS_MEV_BY_PREFIX["pbar"]
    if key == "p":
        return MASS_MEV_BY_PREFIX["p"]
    if key == "k":
        return MASS_MEV_BY_PREFIX["k"]
    if key == "e":
        return MASS_MEV_BY_PREFIX["e"]
    raise RuntimeError(f"Unsupported particle token in filename: {token}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        default="test_only_10k",
        help="Directory with input ROOT files (default: test_only_10k)",
    )
    p.add_argument(
        "--model",
        default="models/pid_model.pt",
        help="Model checkpoint passed to applyNN.py (default: models/pid_model.pt)",
    )
    p.add_argument(
        "--apply-script",
        default="python/applyNN.py",
        help="Path to applyNN.py (default: python/applyNN.py)",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run applyNN.py (default: current interpreter)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under --dir",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run files even if <input>_nnPID.root already exists",
    )
    return p.parse_args()


def is_input_root(path: Path) -> bool:
    return path.suffix == ".root" and not path.name.endswith("_nnPID.root")


def expected_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}_nnPID.root")


def expected_hist_pdf(path: Path) -> Path:
    return expected_output(path).with_suffix(".momentum_hist.pdf")


def expected_hist_png(path: Path) -> Path:
    return expected_output(path).with_suffix(".momentum_hist.png")


def parse_true_momentum_mev_c(path: Path) -> float:
    _token, p_mev = parse_particle_and_momentum(path)
    return p_mev


def read_pred_momentum_mev_c(out_root: Path, input_root: Path) -> np.ndarray:
    with uproot.open(out_root) as f:
        if "nnPID" not in f:
            raise RuntimeError(f"Tree nnPID missing in {out_root}")
        tree = f["nnPID"]
        if "pred_momentum_MeV_c" in tree.keys():
            return np.asarray(tree["pred_momentum_MeV_c"].array(library="np"), dtype=np.float32)
        if "pred_energy_MeV" in tree.keys():
            token, _ = parse_particle_and_momentum(input_root)
            mass = particle_mass_mev(token)
            pred_e = np.asarray(tree["pred_energy_MeV"].array(library="np"), dtype=np.float32)
            return np.sqrt(np.maximum(0.0, pred_e.astype(np.float64) ** 2 - mass * mass)).astype(np.float32)
        raise RuntimeError(f"Neither pred_momentum_MeV_c nor pred_energy_MeV found in {out_root}")


def create_inferred_momentum_overlay(files: list[Path], out_png: Path, out_pdf: Path) -> tuple[int, int]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"Failed to import matplotlib for inferred-momentum overlay: {exc}") from exc

    datasets = []
    missing = 0
    for root_path in files:
        out_root = expected_output(root_path)
        if not out_root.exists():
            missing += 1
            continue
        pred = read_pred_momentum_mev_c(out_root, root_path)
        try:
            true_p = parse_true_momentum_mev_c(root_path)
        except RuntimeError as exc:
            print(f"warning: skipping true-momentum label for {root_path.name}: {exc}")
            missing += 1
            continue
        datasets.append((root_path.name, true_p, pred))

    if not datasets:
        return 0, missing

    all_pred = np.concatenate([d[2] for d in datasets])
    edges = np.histogram_bin_edges(all_pred, bins=320)

    fig, (ax, ax_leg) = plt.subplots(
        2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2]}
    )
    cmap = plt.get_cmap("tab20")
    for i, (name, true_p, pred) in enumerate(datasets):
        color = cmap(i % cmap.N)
        display = name.removeprefix("events_").removesuffix(".root")
        mean_mev = float(np.mean(pred))
        std_mev = float(np.std(pred))
        ax.hist(pred, bins=edges, histtype="stepfilled", alpha=0.3, color=color,
                label=(
                    f"{display} (true={true_p:.2f} MeV/c, "
                    f"mean={mean_mev:.2f}, std={std_mev:.2f} MeV/c)"
                ))
        ax.axvline(true_p, color=color, alpha=0.6, lw=1.0, ls="--")

    ax.set_xlabel("Inferred momentum [MeV/c]")
    ax.set_ylabel("Events")
    ax.set_title("Inferred momentum overlay across test files")
    ax.set_xlim(0.0, 1000.0)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax_leg.axis("off")
    ax_leg.legend(handles, labels, fontsize=10, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    return len(datasets), missing


def create_momentum_scatter_and_residuals(files: list[Path], out_png: Path, out_pdf: Path) -> tuple[int, int]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"Failed to import matplotlib for momentum scatter/residuals: {exc}") from exc

    species_data: dict[str, dict[str, object]] = {}
    missing = 0
    for root_path in files:
        out_root = expected_output(root_path)
        if not out_root.exists():
            missing += 1
            continue
        try:
            token, true_p = parse_particle_and_momentum(root_path)
            species = species_key_from_token(token)
            pred = read_pred_momentum_mev_c(out_root, root_path)
        except RuntimeError as exc:
            print(f"warning: skipping scatter/residual input for {root_path.name}: {exc}")
            missing += 1
            continue
        if species not in species_data:
            species_data[species] = {"true": [], "pred": [], "res_by_p": {}}
        p_true_arr = np.full(pred.shape, true_p, dtype=np.float32)
        p_pred_arr = pred.astype(np.float32)
        species_data[species]["true"].append(p_true_arr)  # type: ignore[operator]
        species_data[species]["pred"].append(p_pred_arr)  # type: ignore[operator]
        residual = (p_pred_arr - p_true_arr) / np.maximum(1e-12, p_true_arr)
        res_by_p = species_data[species]["res_by_p"]  # type: ignore[index]
        if float(true_p) not in res_by_p:
            res_by_p[float(true_p)] = []
        res_by_p[float(true_p)].append(residual)  # type: ignore[index]

    if not species_data:
        return 0, missing

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), squeeze=False)
    total_points = 0
    for row, species in enumerate(("e", "mu", "pi")):
        ax_sc = axes[row][0]
        ax_res = axes[row][1]
        if species not in species_data:
            ax_sc.text(0.5, 0.5, f"no {species} data", ha="center", va="center", transform=ax_sc.transAxes)
            ax_res.text(0.5, 0.5, f"no {species} data", ha="center", va="center", transform=ax_res.transAxes)
            ax_sc.set_xlim(0.0, 1000.0)
            ax_sc.set_ylim(0.0, 1000.0)
            ax_res.set_xlim(0.0, 1000.0)
            ax_res.set_ylim(-0.5, 0.5)
            continue

        p_true = np.concatenate(species_data[species]["true"])  # type: ignore[arg-type]
        p_pred = np.concatenate(species_data[species]["pred"])  # type: ignore[arg-type]
        total_points += int(p_true.size)
        ax_sc.scatter(p_true, p_pred, s=4, alpha=0.2, linewidths=0, color="tab:blue")
        ax_sc.plot([0.0, 1000.0], [0.0, 1000.0], ls="--", lw=1.2, color="black")
        ax_sc.set_xlim(0.0, 1000.0)
        ax_sc.set_ylim(0.0, 1000.0)
        ax_sc.set_xlabel("True momentum [MeV/c]")
        ax_sc.set_ylabel("Inferred momentum [MeV/c]")
        ax_sc.set_title(f"{species}: inferred vs true momentum")
        ax_sc.grid(alpha=0.3)

        res_by_p = species_data[species]["res_by_p"]  # type: ignore[index]
        p_vals = sorted(float(p) for p in res_by_p.keys())
        residual_data = [np.concatenate(res_by_p[p]) for p in p_vals]
        ax_res.boxplot(
            residual_data,
            positions=p_vals,
            widths=20.0,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
            boxprops={"facecolor": "tab:orange", "alpha": 0.3},
            whiskerprops={"color": "tab:orange"},
            capprops={"color": "tab:orange"},
            medianprops={"color": "black", "linewidth": 1.2},
        )
        ax_res.axhline(0.0, ls="--", lw=1.2, color="black")
        ax_res.set_xlim(0.0, 1000.0)
        ax_res.set_ylim(-0.5, 0.5)
        ax_res.set_xlabel("True momentum [MeV/c]")
        ax_res.set_ylabel("(inferred - true) / true")
        ax_res.set_title(f"{species}: residual box plots")
        ax_res.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    return total_points, missing


def create_confusion_by_momentum_grid(files: list[Path], out_png: Path, out_pdf: Path) -> tuple[int, int]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"Failed to import matplotlib for confusion-by-momentum plots: {exc}") from exc

    class_pdgs: list[int] | None = None
    cm_by_momentum: dict[float, np.ndarray] = {}
    missing = 0

    for root_path in files:
        out_root = expected_output(root_path)
        if not out_root.exists():
            missing += 1
            continue
        try:
            token, true_p = parse_particle_and_momentum(root_path)
            true_pdg = true_pdg_from_token(token)
            with uproot.open(out_root) as f:
                if "nnPID" not in f:
                    raise RuntimeError(f"Tree nnPID missing in {out_root}")
                tree = f["nnPID"]
                prob_branches = [b for b in tree.keys() if b.startswith("prob_")]
                if not prob_branches:
                    raise RuntimeError(f"No prob_* branches in {out_root}")
                file_class_pdgs = sorted(int(b[len("prob_"):]) for b in prob_branches)
                if "pred_pdg" in tree.keys():
                    pred_pdg = np.asarray(tree["pred_pdg"].array(library="np"), dtype=np.int64)
                else:
                    pred_class = np.asarray(tree["pred_class"].array(library="np"), dtype=np.int64)
                    probs = np.stack(
                        [np.asarray(tree[b].array(library="np"), dtype=np.float32) for b in prob_branches], axis=1
                    )
                    branch_pdgs = [int(b[len("prob_"):]) for b in prob_branches]
                    class_to_pdg = infer_class_to_pdg_mapping(pred_class, probs, branch_pdgs)
                    pred_pdg = np.asarray([class_to_pdg[int(c)] for c in pred_class], dtype=np.int64)
        except RuntimeError as exc:
            print(f"warning: skipping confusion input for {root_path.name}: {exc}")
            missing += 1
            continue

        if class_pdgs is None:
            class_pdgs = file_class_pdgs
        elif file_class_pdgs != class_pdgs:
            raise RuntimeError(
                f"Inconsistent class order between files: {out_root} has {file_class_pdgs}, expected {class_pdgs}"
            )

        if true_pdg not in class_pdgs:
            print(f"warning: true PDG {true_pdg} missing from class list in {root_path.name}; skipping")
            missing += 1
            continue
        n_classes = len(class_pdgs)
        true_idx = class_pdgs.index(true_pdg)
        pred_idxs = np.asarray([class_pdgs.index(int(p)) for p in pred_pdg if int(p) in class_pdgs], dtype=np.int64)
        if pred_idxs.size == 0:
            continue

        if true_p not in cm_by_momentum:
            cm_by_momentum[true_p] = np.zeros((n_classes, n_classes), dtype=np.int64)
        cm_by_momentum[true_p][true_idx] += np.bincount(pred_idxs, minlength=n_classes)

    if not cm_by_momentum or class_pdgs is None:
        return 0, missing

    momenta = sorted(cm_by_momentum.keys())
    n_panels = len(momenta)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    labels = [f"{pdg_name(p)} ({p})" for p in class_pdgs]
    for i, p in enumerate(momenta):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        cm = cm_by_momentum[p]
        norm = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for rr in range(len(class_pdgs)):
            for cc in range(len(class_pdgs)):
                v = norm[rr, cc]
                ax.text(
                    cc,
                    rr,
                    f"{cm[rr, cc]}\n{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v > 0.5 else "black",
                    fontsize=8,
                )
        ax.set(
            xticks=np.arange(len(class_pdgs)),
            yticks=np.arange(len(class_pdgs)),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="predicted",
            ylabel="true",
            title=f"p = {p:.0f} MeV/c (N={int(cm.sum())})",
        )
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")

    for j in range(n_panels, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    return n_panels, missing


def infer_class_to_pdg_mapping(
    pred_class: np.ndarray, probs: np.ndarray, branch_pdgs: list[int]
) -> list[int]:
    """Infer class-index->PDG mapping from pred_class and prob_* arrays."""
    n_classes = probs.shape[1]
    if probs.shape[0] != pred_class.shape[0]:
        raise RuntimeError("pred_class/prob_* size mismatch while inferring class mapping.")
    if len(branch_pdgs) != n_classes:
        raise RuntimeError("prob_* branch count mismatch while inferring class mapping.")
    if n_classes > 8:
        raise RuntimeError("Too many classes to infer class mapping safely.")

    scores = np.zeros((n_classes, n_classes), dtype=np.float64)
    for cls_idx in range(n_classes):
        sel = pred_class == cls_idx
        if np.any(sel):
            scores[cls_idx] = probs[sel].mean(axis=0)

    best_perm: tuple[int, ...] | None = None
    best_score = -np.inf
    for perm in itertools.permutations(range(n_classes)):
        s = sum(scores[i, perm[i]] for i in range(n_classes))
        if s > best_score:
            best_score = s
            best_perm = perm
    if best_perm is None:
        raise RuntimeError("Could not infer class-to-PDG mapping.")
    return [int(branch_pdgs[j]) for j in best_perm]


def create_summary_pdf(files: list[Path], out_pdf: Path) -> tuple[int, int]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        raise SystemExit(f"Failed to import matplotlib for summary PDF creation: {exc}") from exc

    included = 0
    missing = 0
    with PdfPages(out_pdf) as pdf:
        for root_path in files:
            out_root = expected_output(root_path)
            single_pdf = out_root.with_suffix(".momentum_hist.pdf")
            png = out_root.with_suffix(".momentum_hist.png")
            if not single_pdf.exists() or not png.exists():
                single_pdf = out_root.with_suffix(".energy_hist.pdf")
                png = out_root.with_suffix(".energy_hist.png")
            if not single_pdf.exists() or not png.exists():
                missing += 1
                continue
            img = plt.imread(png)
            fig, ax = plt.subplots(figsize=(11.0, 8.5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                f"{root_path.name}\nsource: {single_pdf.name}",
                fontsize=11,
                pad=10,
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            included += 1
    return included, missing


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_dir = Path(args.dir)
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    iterator = input_dir.rglob("*.root") if args.recursive else input_dir.glob("*.root")
    files = sorted(path for path in iterator if is_input_root(path))
    if not files:
        print(f"No input ROOT files found in {input_dir}")
        return

    apply_script = Path(args.apply_script)
    if not apply_script.is_absolute():
        apply_script = project_root / apply_script
    model = Path(args.model)
    if not model.is_absolute():
        model = project_root / model
    if not apply_script.exists():
        raise SystemExit(f"applyNN script not found: {apply_script}")
    if not model.exists():
        raise SystemExit(f"Model checkpoint not found: {model}")

    print(f"Found {len(files)} candidate ROOT files in {input_dir}")
    processed = 0
    skipped = 0
    failed: list[Path] = []
    for i, root_path in enumerate(files, start=1):
        out_root = expected_output(root_path)
        if out_root.exists() and not args.force:
            print(f"[{i}/{len(files)}] skip {root_path} (already has {out_root.name})")
            skipped += 1
            continue

        cmd = [
            args.python,
            str(apply_script),
            "--model",
            str(model),
            "--input",
            str(root_path),
            "--output",
            str(out_root),
        ]
        print(f"[{i}/{len(files)}] run {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=project_root, check=False).returncode
        if rc != 0:
            failed.append(root_path)
            print(f"  -> FAILED (exit code {rc})")
        else:
            processed += 1

    print(
        f"Done. processed={processed}, skipped={skipped}, failed={len(failed)}, total={len(files)}"
    )
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  - {f}")
        raise SystemExit(1)

    summary_pdf = input_dir / "all_tests_momentum_hists.pdf"
    included, missing = create_summary_pdf(files, summary_pdf)
    if included == 0:
        raise SystemExit(
            "No per-test histogram PDFs/PNGs found to include in summary PDF. "
            "Run with --force to regenerate outputs."
        )
    print(f"Saved summary PDF: {summary_pdf} (included={included}, missing={missing})")

    overlay_png = input_dir / "all_tests_inferred_momentum_overlay.png"
    overlay_pdf = input_dir / "all_tests_inferred_momentum_overlay.pdf"
    overlay_included, overlay_missing = create_inferred_momentum_overlay(files, overlay_png, overlay_pdf)
    if overlay_included == 0:
        raise SystemExit(
            "No inferred-momentum outputs available for overlay plot. "
            "Run with --force to regenerate outputs."
        )
    print(
        f"Saved overlay plots: {overlay_png}, {overlay_pdf} "
        f"(included={overlay_included}, missing={overlay_missing})"
    )

    scatter_png = input_dir / "all_tests_momentum_scatter_residuals.png"
    scatter_pdf = input_dir / "all_tests_momentum_scatter_residuals.pdf"
    scatter_points, scatter_missing = create_momentum_scatter_and_residuals(files, scatter_png, scatter_pdf)
    if scatter_points == 0:
        raise SystemExit(
            "No inferred-momentum outputs available for scatter/residual plots. "
            "Run with --force to regenerate outputs."
        )
    print(
        f"Saved scatter/residual plots: {scatter_png}, {scatter_pdf} "
        f"(points={scatter_points}, missing={scatter_missing})"
    )

    cm_png = input_dir / "all_tests_confusion_by_momentum.png"
    cm_pdf = input_dir / "all_tests_confusion_by_momentum.pdf"
    n_momenta, cm_missing = create_confusion_by_momentum_grid(files, cm_png, cm_pdf)
    if n_momenta == 0:
        raise SystemExit(
            "No valid inferred outputs available for confusion-by-momentum plots. "
            "Run with --force to regenerate outputs."
        )
    print(
        f"Saved confusion-by-momentum plots: {cm_png}, {cm_pdf} "
        f"(momenta={n_momenta}, missing={cm_missing})"
    )


if __name__ == "__main__":
    main()
