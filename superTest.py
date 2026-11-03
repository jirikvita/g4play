

#!/usr/bin/env python3

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


def main() -> int:
    script_path = Path(__file__).with_name("nnTestAll.py")

    for directory in DIRS:
        cmd = [sys.executable, str(script_path), "--model", "pid_model.pt", "--dir", directory]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Error: nnTestAll.py failed for '{directory}' (exit code {result.returncode})")
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
