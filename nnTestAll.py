#!/usr/bin/env python3
"""Run applyNN.py on all non-nnPID ROOT files in a directory."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import uproot

MASS_MEV_BY_PREFIX = {
    "e": 0.51099895,
    "mu": 105.6583755,
    "pi": 139.57039,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        default="test_only_1k",
        help="Directory with input ROOT files (default: test_only_1k)",
    )
    p.add_argument(
        "--model",
        default="pid_model.pt",
        help="Model checkpoint passed to applyNN.py (default: pid_model.pt)",
    )
    p.add_argument(
        "--apply-script",
        default="applyNN.py",
        help="Path to applyNN.py (default: applyNN.py)",
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
    return expected_output(path).with_suffix(".energy_hist.pdf")


def expected_hist_png(path: Path) -> Path:
    return expected_output(path).with_suffix(".energy_hist.png")


def parse_true_energy_mev(path: Path) -> float:
    name = path.name
    if not name.startswith("events_"):
        raise RuntimeError(f"Unsupported ROOT filename format: {name}")
    token = name[len("events_"):].split("_p", 1)[0]
    if "_p" not in name or "MeV_" not in name:
        raise RuntimeError(f"Missing momentum token in filename: {name}")
    p_str = name.split("_p", 1)[1].split("MeV_", 1)[0]
    p_mev = float(p_str)

    if token.startswith("mu"):
        mass = MASS_MEV_BY_PREFIX["mu"]
    elif token.startswith("pi"):
        mass = MASS_MEV_BY_PREFIX["pi"]
    elif token.startswith("e"):
        mass = MASS_MEV_BY_PREFIX["e"]
    else:
        raise RuntimeError(f"Unsupported particle token in filename: {token}")
    return math.sqrt(p_mev * p_mev + mass * mass)


def create_inferred_energy_overlay(files: list[Path], out_png: Path, out_pdf: Path) -> tuple[int, int]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"Failed to import matplotlib for inferred-energy overlay: {exc}") from exc

    datasets = []
    missing = 0
    for root_path in files:
        out_root = expected_output(root_path)
        if not out_root.exists():
            missing += 1
            continue
        with uproot.open(out_root) as f:
            if "nnPID" not in f:
                raise RuntimeError(f"Tree nnPID missing in {out_root}")
            tree = f["nnPID"]
            if "pred_energy_MeV" not in tree.keys():
                raise RuntimeError(f"Branch pred_energy_MeV missing in {out_root}")
            pred = np.asarray(tree["pred_energy_MeV"].array(library="np"), dtype=np.float32)
        true_e = parse_true_energy_mev(root_path)
        datasets.append((root_path.name, true_e, pred))

    if not datasets:
        return 0, missing

    all_pred = np.concatenate([d[2] for d in datasets])
    edges = np.histogram_bin_edges(all_pred, bins=160)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("tab20")
    for i, (name, true_e, pred) in enumerate(datasets):
        color = cmap(i % cmap.N)
        display = name.removeprefix("events_").removesuffix(".root")
        mean_mev = float(np.mean(pred))
        std_mev = float(np.std(pred))
        ax.hist(pred, bins=edges, histtype="stepfilled", alpha=0.3, color=color,
                label=(
                    f"{display} (true={true_e:.2f} MeV, "
                    f"mean={mean_mev:.2f}, std={std_mev:.2f} MeV)"
                ))
        ax.axvline(true_e, color=color, alpha=0.6, lw=1.0, ls="--")

    ax.set_xlabel("Inferred energy [MeV]")
    ax.set_ylabel("Events")
    ax.set_title("Inferred energy overlay across test files")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    return len(datasets), missing


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
            single_pdf = expected_hist_pdf(root_path)
            png = expected_hist_png(root_path)
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
    input_dir = Path(args.dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    iterator = input_dir.rglob("*.root") if args.recursive else input_dir.glob("*.root")
    files = sorted(path for path in iterator if is_input_root(path))
    if not files:
        print(f"No input ROOT files found in {input_dir}")
        return

    apply_script = Path(args.apply_script)
    model = Path(args.model)
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
        rc = subprocess.run(cmd, check=False).returncode
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

    summary_pdf = input_dir / "all_tests_energy_hists.pdf"
    included, missing = create_summary_pdf(files, summary_pdf)
    if included == 0:
        raise SystemExit(
            "No per-test histogram PDFs/PNGs found to include in summary PDF. "
            "Run with --force to regenerate outputs."
        )
    print(f"Saved summary PDF: {summary_pdf} (included={included}, missing={missing})")

    overlay_png = input_dir / "all_tests_inferred_energy_overlay.png"
    overlay_pdf = input_dir / "all_tests_inferred_energy_overlay.pdf"
    overlay_included, overlay_missing = create_inferred_energy_overlay(files, overlay_png, overlay_pdf)
    if overlay_included == 0:
        raise SystemExit(
            "No inferred-energy outputs available for overlay plot. "
            "Run with --force to regenerate outputs."
        )
    print(
        f"Saved overlay plots: {overlay_png}, {overlay_pdf} "
        f"(included={overlay_included}, missing={overlay_missing})"
    )


if __name__ == "__main__":
    main()
