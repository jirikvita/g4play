#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


DIRS = [
    "test_only_10k",
    "test_only_10k_PoissSmear_100s",
    "test_only_10k_PoissSmear_addNoise_100s",
    "test_only_10k_PoissSmear_addNoise_50s",
    "test_only_10k_PoissSmear_50s",
]


def model_tag(model_path: Path) -> str:
    tag = model_path.stem
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")
    return tag or "model"


def collect_result_files(test_dir: Path) -> list[Path]:
    files: dict[str, Path] = {}
    for pattern in ("*_nnPID*", "all_tests_*"):
        for path in test_dir.glob(pattern):
            if path.is_file():
                files[path.name] = path
    return sorted(files.values())


def move_results_to_model_subdir(test_dir: Path, model_name_tag: str) -> tuple[Path, int]:
    result_dir = test_dir / f"model_{model_name_tag}"
    result_dir.mkdir(exist_ok=True)
    moved = 0
    for src in collect_result_files(test_dir):
        dst = result_dir / src.name
        if dst.exists():
            dst.unlink()
        src.replace(dst)
        moved += 1
    return result_dir, moved


def build_html_gallery(result_dir: Path, html_script: Path, title: str) -> None:
    pngs = list(result_dir.glob("*.png"))
    if not pngs:
        print(f"Skipping HTML gallery in {result_dir}: no PNG files found")
        return
    env = os.environ.copy()
    env["TITLE"] = title
    cmd = ["bash", str(html_script)]
    print("Running:", " ".join(cmd), f"(cwd={result_dir}, TITLE={title})")
    result = subprocess.run(cmd, cwd=result_dir, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"Error: HTML gallery generation failed in '{result_dir}' "
            f"(exit code {result.returncode})"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run nnTestAll.py over standard test directories, move generated outputs "
            "into per-model subdirectories, and build HTML galleries."
        )
    )
    p.add_argument(
        "model_arg",
        nargs="?",
        default=None,
        help="Optional positional model checkpoint path (same as --model).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model checkpoint passed to nnTestAll.py (default: models/pid_model.pt)",
    )
    p.add_argument(
        "--dirs",
        nargs="+",
        default=DIRS,
        help="Test directories to process under results/ by default (default: built-in list)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_root = project_root / "results"
    script_path = script_dir / "nnTestAll.py"
    html_script = project_root / "scripts" / "CreateHTMLpage_noConv.sh"
    model_input = args.model if args.model is not None else args.model_arg
    if model_input is None:
        model_input = "models/pid_model.pt"
    model = Path(model_input)
    if not model.is_absolute():
        model = project_root / model
    model_name = model.name
    model_name_tag = model_tag(model)

    if not script_path.exists():
        raise SystemExit(f"Error: nnTestAll script not found: {script_path}")
    if not html_script.exists():
        raise SystemExit(f"Error: HTML script not found: {html_script}")
    if not model.exists():
        raise SystemExit(f"Error: model checkpoint not found: {model}")

    for directory in args.dirs:
        test_dir = Path(directory)
        if not test_dir.is_absolute():
            if test_dir.parts and test_dir.parts[0] == "results":
                test_dir = project_root / test_dir
            else:
                test_dir = results_root / test_dir
        cmd = [sys.executable, str(script_path), "--model", str(model), "--dir", str(test_dir)]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=project_root, check=False)
        if result.returncode != 0:
            print(f"Error: nnTestAll.py failed for '{directory}' (exit code {result.returncode})")
            return result.returncode
        result_dir, moved = move_results_to_model_subdir(test_dir, model_name_tag)
        print(f"Moved {moved} result files into {result_dir}")
        build_html_gallery(result_dir, html_script, model_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
