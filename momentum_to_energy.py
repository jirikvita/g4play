#!/usr/bin/env python3
"""Print kinetic/total energy for beam momenta, as set by run_scan.py (/gun/momentumAmp).

Usage: ./momentum_to_energy.py [-p PARTICLE] [P1 P2 ...]   (momenta in MeV/c)
PARTICLE is one of: e-, e+, mu-, mu+, pi-, pi+, p, pbar (default mu+).
"""
import argparse
import math

MASS_MEV = {
    "e-": 0.51099895, "e+": 0.51099895,
    "mu-": 105.6583755, "mu+": 105.6583755,
    "pi-": 139.57039, "pi+": 139.57039,
    "p": 938.2720813, "pbar": 938.2720813,
}
DEFAULT_MOMENTA_MEV = [200, 300, 400, 500, 600, 800]  # same as run_scan.py

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("momenta", nargs="*", type=float, help="beam momenta in MeV/c")
parser.add_argument("-p", "--particle", default="mu+", choices=sorted(MASS_MEV))
args = parser.parse_args()

m = MASS_MEV[args.particle]
print(f"{args.particle}  (m = {m:g} MeV/c^2)")
print(f"{'p [MeV/c]':>10} {'T [MeV]':>10} {'E [MeV]':>10} {'beta':>8} {'gamma':>8}")
for p in args.momenta or DEFAULT_MOMENTA_MEV:
    e = math.hypot(p, m)
    print(f"{p:10g} {e - m:10.2f} {e:10.2f} {p / e:8.4f} {e / m:8.3f}")
