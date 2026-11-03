#!/usr/bin/env python3
"""Run multiple run_scan.py particle scans concurrently.

By default, starts scans for e+, mu+, and pi+ at the same momenta.
Usage examples:
  python3 superRun.py
  python3 superRun.py 200 300 400
  python3 superRun.py --particles mu+ pi+ 250 350 450
  python3 superRun.py -n 200
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_PARTICLES = ["e+", "mu+", "pi+"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("momenta", nargs="*", type=float, help="beam momenta in MeV/c")
    parser.add_argument(
        "--particles",
        nargs="+",
        choices=["e-", "e+", "mu-", "mu+", "pi-", "pi+", "p", "pbar"],
        default=DEFAULT_PARTICLES,
        help="particles to run in parallel (default: e+ mu+ pi+)",
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="only print generated commands/macros")
    args = parser.parse_args()

    run_scan = Path(__file__).resolve().with_name("run_scan.py")
    if not run_scan.exists():
        print(f"Missing {run_scan}", file=sys.stderr)
        return 2

    momenta_args = [f"{p:g}" for p in args.momenta]
    procs = []
    try:
        for particle in args.particles:
            cmd = [sys.executable, str(run_scan), "-p", particle]
            if args.dry_run:
                cmd.append("-n")
            cmd.extend(momenta_args)
            print(f"[superRun] starting {particle}: {' '.join(cmd)}", flush=True)
            procs.append((particle, subprocess.Popen(cmd, cwd=run_scan.parent)))
    except Exception:
        for _, proc in procs:
            proc.terminate()
        raise

    failed = []
    try:
        for particle, proc in procs:
            rc = proc.wait()
            if rc != 0:
                failed.append((particle, rc))
    except KeyboardInterrupt:
        for _, proc in procs:
            proc.terminate()
        return 130

    if failed:
        for particle, rc in failed:
            print(f"[superRun] {particle} failed with exit code {rc}", file=sys.stderr)
        return 1
    print("[superRun] all requested scans finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
