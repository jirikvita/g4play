#!/usr/bin/env python3
"""Run run.mac for mu+ at several energies; each output is muonDecay_<E>MeV.root.

Usage: ./run_scan.py [E1 E2 ...]   (energies in MeV; default below)
Override the Geant4 environment script with G4_SH=/path/to/geant4.sh.
"""
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PARTICLE = "mu+"
DEFAULT_ENERGIES_MEV = [200, 300, 400, 500, 600, 800]
G4_SH = os.environ.get("G4_SH", "/home/qitek/install/geant4-v11.1.1-install/bin/geant4.sh")

here = Path(__file__).resolve().parent
template = (here / "run.mac").read_text()
energies = [float(e) for e in sys.argv[1:]] or DEFAULT_ENERGIES_MEV

for energy in energies:
    tag = f"{energy:g}"
    out_name = f"muonDecay_{tag}MeV.root"

    macro = re.sub(r"^/gun/particle .*$", f"/gun/particle {PARTICLE}", template, flags=re.MULTILINE)
    macro = re.sub(r"^/gun/energy .*$", f"/gun/energy {tag} MeV", macro, flags=re.MULTILINE)

    with tempfile.NamedTemporaryFile("w", suffix=".mac", delete=False) as tmp:
        tmp.write(macro)
        macro_path = tmp.name

    try:
        print(f"=== {PARTICLE} {tag} MeV -> {out_name}", flush=True)
        cmd = f"source {shlex.quote(G4_SH)} && exec ./waterCherenkovSim {shlex.quote(macro_path)}"
        env = dict(os.environ, MUON_OUT=out_name)
        result = subprocess.run(["bash", "-c", cmd], cwd=here, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    finally:
        os.unlink(macro_path)

    if result.returncode != 0:
        sys.exit(f"run failed for {tag} MeV:\n{result.stderr[-2000:]}")
    if not (here / out_name).exists():
        sys.exit(f"{out_name} was not created (no muons stopped/decayed?)")

print("done")
