#!/usr/bin/env python3
"""Run run.mac at several beam momenta; each output is muonDecay_p<P>MeV.root.

Usage: ./run_scan.py [-n] [-p PARTICLE] [P1 P2 ...]   (momenta in MeV/c; default below)
PARTICLE is one of: e-, e+, mu-, mu+, pi-, pi+, p, pbar (default mu+).
-n / --dry-run only prints the commands and generated macros; nothing is run.
The beam momentum (not kinetic energy) is set with /gun/momentumAmp.
Override the Geant4 environment script with G4_SH=/path/to/geant4.sh.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import deque
from pathlib import Path

PARTICLES = {
    "e-": "e-", "e+": "e+",
    "mu-": "mu-", "mu+": "mu+",
    "pi-": "pi-", "pi+": "pi+",
    "p": "proton", "pbar": "anti_proton",
}
DEFAULT_MOMENTA_MEV = [200] #, 300, 400, 500, 600, 800]
G4_SH = os.environ.get("G4_SH", "/home/qitek/install/geant4-v11.1.1-install/bin/geant4.sh")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("momenta", nargs="*", type=float, help="beam momenta in MeV/c")
parser.add_argument("-p", "--particle", default="mu+", choices=sorted(PARTICLES))
parser.add_argument("-n", "--dry-run", action="store_true", help="only print the commands")
args = parser.parse_args()

particle = PARTICLES[args.particle]
momenta = args.momenta or DEFAULT_MOMENTA_MEV

here = Path(__file__).resolve().parent
template = "\n".join(
    line for line in (here / "run.mac").read_text().splitlines()
    # scans run without visualization, so no g4_*.wrl files are written
    if not line.startswith("/vis/") and not line.startswith("/control/getEnv VIS_DRIVER")
)

m = re.search(r"^/run/beamOn\s+(\d+)", template, flags=re.MULTILINE)
nevents = int(m.group(1)) if m else 0
EVENT = re.compile(r"--> Event (\d+) starts")
is_tty = sys.stdout.isatty()


def show_bar(frac, text):
    width = 30
    filled = int(width * frac)
    line = f"[{'#' * filled}{'-' * (width - filled)}] {frac * 100:5.1f}%  {text}"
    pad = max(shutil.get_terminal_size().columns - 1, len(line))
    print("\r" + line.ljust(pad), end="", flush=True)


for k, p in enumerate(momenta):
    tag = f"{p:g}"
    out_name = f"muonDecay_p{tag}MeV.root"

    macro, n1 = re.subn(r"^/gun/particle .*$", f"/gun/particle {particle}", template, flags=re.MULTILINE)
    # momentumAmp must follow /gun/particle: it derives kinetic energy from the particle mass
    macro, n2 = re.subn(r"^/gun/(?:energy|momentumAmp) .*$", f"/gun/momentumAmp {tag} MeV",
                        macro, flags=re.MULTILINE)
    if not (n1 and n2):
        sys.exit("run.mac must contain /gun/particle and /gun/energy (or /gun/momentumAmp) lines")
    if nevents:  # ask Geant4 for ~200 progress messages per run
        macro = re.sub(r"^/run/beamOn", f"/run/printProgress {max(1, nevents // 200)}\n/run/beamOn",
                       macro, count=1, flags=re.MULTILINE)

    def command(macro_path):
        return f"source {shlex.quote(G4_SH)} && exec ./waterCherenkovSim {shlex.quote(macro_path)}"

    env = dict(os.environ, MUON_OUT=out_name)
    label = f"[{k + 1}/{len(momenta)}] {particle} p = {tag} MeV/c -> {out_name}"

    if args.dry_run:
        print(f"# {label}")
        print(f"MUON_OUT={shlex.quote(out_name)} bash -c {shlex.quote(command('<macro>.mac'))}")
        print("# <macro>.mac:")
        print(textwrap.indent(macro, "#   "))
        continue

    with tempfile.NamedTemporaryFile("w", suffix=".mac", delete=False) as tmp:
        tmp.write(macro)
        macro_path = tmp.name

    tail = deque(maxlen=40)  # last output lines, shown if the run fails
    try:
        if is_tty:
            show_bar(k / len(momenta), label)
        else:
            print(f"=== {label}", flush=True)
        proc = subprocess.Popen(["bash", "-c", command(macro_path)], cwd=here, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        seen = 0
        for line in proc.stdout:
            tail.append(line)
            ev = EVENT.search(line)
            if ev and nevents and is_tty:
                seen = max(seen, int(ev.group(1)))
                show_bar((k + min(seen / nevents, 1.0)) / len(momenta), label)
        returncode = proc.wait()
    finally:
        os.unlink(macro_path)

    if returncode != 0:
        if is_tty:
            print()
        sys.exit(f"run failed for p = {tag} MeV/c:\n{''.join(tail)}")
    if not (here / out_name).exists():
        if is_tty:
            print()
        sys.exit(f"{out_name} was not created (no muons stopped/decayed?)")
    if is_tty:
        show_bar((k + 1) / len(momenta), label)

if is_tty and not args.dry_run:
    print()
print("done" if not args.dry_run else "dry run: nothing was run")
