#!/usr/bin/env python3
"""Generate per-test HTML pages comparing model result images side-by-side."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate HTML pages under each results/* subdirectory that compare "
            "images from all model_* subdirectories in side-by-side columns."
        )
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Results directory root (default: <project-root>/results).",
    )
    parser.add_argument(
        "--output-name",
        default="models_cmp.html",
        help="Name of generated HTML files inside each results subdirectory.",
    )
    return parser.parse_args()


def collect_model_images(model_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.name] = path
    return images


def build_html(test_dir: Path, model_to_images: dict[str, dict[str, Path]]) -> str:
    model_names = sorted(model_to_images)
    image_names = sorted(
        {
            image_name
            for images in model_to_images.values()
            for image_name in images
        }
    )
    title = f"Model image comparison: {test_dir.name}"
    lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'/>",
        f"  <title>{html.escape(title)}</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 1rem; }",
        "    table { border-collapse: collapse; width: 100%; }",
        "    th, td { border: 1px solid #ddd; padding: 0.5rem; vertical-align: top; }",
        "    th { position: sticky; top: 0; background: #f7f7f7; z-index: 1; }",
        "    td img { max-width: 100%; height: auto; display: block; }",
        "    td.missing { color: #9a9a9a; font-style: italic; text-align: center; }",
        "    td.filename { white-space: nowrap; font-family: monospace; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{html.escape(title)}</h1>",
        "  <table>",
        "    <thead>",
        "      <tr>",
        "        <th>Image</th>",
    ]
    for model_name in model_names:
        lines.append(f"        <th>{html.escape(model_name)}</th>")
    lines.extend(
        [
            "      </tr>",
            "    </thead>",
            "    <tbody>",
        ]
    )

    for image_name in image_names:
        lines.append("      <tr>")
        lines.append(f"        <td class='filename'>{html.escape(image_name)}</td>")
        for model_name in model_names:
            image_path = model_to_images[model_name].get(image_name)
            if image_path is None:
                lines.append("        <td class='missing'>missing</td>")
                continue
            rel_path = image_path.relative_to(test_dir).as_posix()
            escaped_image_name = html.escape(image_name)
            escaped_rel_path = html.escape(rel_path)
            lines.append("        <td>")
            lines.append(
                f"          <a href='{escaped_rel_path}' target='_blank'>"
                f"<img src='{escaped_rel_path}' alt='{escaped_image_name}'/></a>"
            )
            lines.append("        </td>")
        lines.append("      </tr>")

    lines.extend(
        [
            "    </tbody>",
            "  </table>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(lines)


def generate_for_test_dir(test_dir: Path, output_name: str) -> bool:
    model_dirs = sorted(
        [
            path
            for path in test_dir.iterdir()
            if path.is_dir() and path.name.startswith("model_")
        ],
        key=lambda path: path.name,
    )
    if not model_dirs:
        return False

    model_to_images = {model_dir.name: collect_model_images(model_dir) for model_dir in model_dirs}
    output_path = test_dir / output_name
    output_path.write_text(build_html(test_dir, model_to_images), encoding="utf-8")
    print(f"Generated {output_path}")
    return True


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_root = Path(args.results_root) if args.results_root else (project_root / "results")
    results_root = results_root.resolve()

    if not results_root.exists() or not results_root.is_dir():
        raise SystemExit(f"Error: results root does not exist or is not a directory: {results_root}")

    generated_count = 0
    for test_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        if generate_for_test_dir(test_dir, args.output_name):
            generated_count += 1
        else:
            print(f"Skipping {test_dir}: no model_* directories found")

    print(f"Done. Generated {generated_count} comparison page(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
