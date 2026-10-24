#!/usr/bin/env python3
"""Run run.mac at several beam momenta; each output is muonDecay_p<P>MeV.root.

Usage: ./run_scan.py [-p PARTICLE] [P1 P2 ...]   (momenta in MeV/c; default below)
PARTICLE is one of: e-, e+, mu-, mu+, pi-, pi+, p, pbar (default mu+).
The beam momentum (not kinetic energy) is set with /gun/momentumAmp.
Override the Geant4 environment script with G4_SH=/path/to/geant4.sh.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PARTICLES = {
    "e-": "e-", "e+": "e+",
    "mu-": "mu-", "mu+": "mu+",
    "pi-": "pi-", "pi+": "pi+",
    "p": "proton", "pbar": "anti_proton",
}
DEFAULT_MOMENTA_MEV = [200, 300, 400, 500, 600, 800]
G4_SH = os.environ.get("G4_SH", "/home/qitek/install/geant4-v11.1.1-install/bin/geant4.sh")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("momenta", nargs="*", type=float, help="beam momenta in MeV/c")
parser.add_argument("-p", "--particle", default="mu+", choices=sorted(PARTICLES))
args = parser.parse_args()

particle = PARTICLES[args.particle]
momenta = args.momenta or DEFAULT_MOMENTA_MEV

here = Path(__file__).resolve().parent
template = (here / "run.mac").read_text()

for p in momenta:
    tag = f"{p:g}"
    out_name = f"muonDecay_p{tag}MeV.root"

    macro, n1 = re.subn(r"^/gun/particle .*$", f"/gun/particle {particle}", template, flags=re.MULTILINE)
    # momentumAmp must follow /gun/particle: it derives kinetic energy from the particle mass
    macro, n2 = re.subn(r"^/gun/(?:energy|momentumAmp) .*$", f"/gun/momentumAmp {tag} MeV",
                        macro, flags=re.MULTILINE)
    if not (n1 and n2):
        sys.exit("run.mac must contain /gun/particle and /gun/energy (or /gun/momentumAmp) lines")

    with tempfile.NamedTemporaryFile("w", suffix=".mac", delete=False) as tmp:
        tmp.write(macro)
        macro_path = tmp.name

    try:
        print(f"=== {particle} p = {tag} MeV/c -> {out_name}", flush=True)
        cmd = f"source {shlex.quote(G4_SH)} && exec ./waterCherenkovSim {shlex.quote(macro_path)}"
        env = dict(os.environ, MUON_OUT=out_name)
        result = subprocess.run(["bash", "-c", cmd], cwd=here, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    finally:
        os.unlink(macro_path)

    if result.returncode != 0:
        sys.exit(f"run failed for p = {tag} MeV/c:\n{result.stderr[-2000:]}")
    if not (here / out_name).exists():
        sys.exit(f"{out_name} was not created (no muons stopped/decayed?)")

print("done")
