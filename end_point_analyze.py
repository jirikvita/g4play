#!/usr/bin/env python3
"""Overlay muon end-point x, z, y and the mid-plane angle from each muonDecay ROOT file (2x2 canvas).

The beam momentum label is parsed from the file name, e.g. muonDecay_p300MeV.root.
Usage: ./end_point_analyze.py [files.root ...]   (default: muonDecay_p*.root)
Make files with: ./run_scan.py [-p PARTICLE] [P1 P2 ...]
Press any key in the terminal to close the canvas.
"""
import glob
import re
import select
import sys
import termios
import tty

import ROOT

ROOT.gROOT.SetBatch(False)
ROOT.gStyle.SetOptStat(0)

files = sys.argv[1:] or sorted(glob.glob("muonDecay_p*.root"))
if not files:
    sys.exit("no ROOT files found")

MOMENTUM = re.compile(r"_p(\d+(?:\.\d+)?)\s*(keV|MeV|GeV|TeV)", re.IGNORECASE)
SCALE = {"kev": 1e-3, "mev": 1.0, "gev": 1e3, "tev": 1e6}
ALPHA = 0.3


def momentum_mev(name):
    m = MOMENTUM.search(name)
    return float(m.group(1)) * SCALE[m.group(2).lower()] if m else float("inf")


def label(name):
    m = MOMENTUM.search(name)
    return f"p = {m.group(1)} {m.group(2)}/c" if m else name


def wait_for_key():
    """Block until any key is pressed in the terminal, keeping the canvas responsive."""
    if not sys.stdin.isatty():
        return
    print("Press any key to close...", flush=True)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not select.select([sys.stdin], [], [], 0.05)[0]:
            ROOT.gSystem.ProcessEvents()
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


files.sort(key=momentum_mev)

colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2, ROOT.kMagenta, ROOT.kOrange + 7, ROOT.kCyan + 2]

# One pad per quantity; shared binning so curves are comparable.
# (key, expression, cut, title/axis, nbins, xmin, xmax)
# angle_deg is the muon direction vs. the gun direction at the plane through the tank
# centre perpendicular to the gun direction; -1 means the plane was never crossed.
QUANTITIES = [
    ("x", "x_mm", "", "End point x;x [mm];entries", 200, -2000.0, 2000.0),
    ("z", "z_mm", "", "End point z;z [mm];entries", 200, -1000.0, 1000.0),
    ("angle", "angle_deg", "angle_deg >= 0", "Angle at mid-plane;angle to gun direction [deg];entries",
     100, 0.0, 40.0),
    ("y", "y_mm", "", "End point y;y [mm];entries", 200, -1000.0, 1000.0),
]

canvas = ROOT.TCanvas("c", "muon end points and mid-plane angle", 1200, 900)
canvas.Divide(2, 2)

hists = {q[0]: [] for q in QUANTITIES}  # keep references alive
legends = {q[0]: [] for q in QUANTITIES}
files_open = []
for i, name in enumerate(files):
    f = ROOT.TFile.Open(name)
    if not f or f.IsZombie():
        print(f"skipping {name}: cannot open")
        continue
    tree = f.Get("muonDecay")
    if not tree or tree.GetEntries() == 0:
        print(f"skipping {name}: no entries")
        continue
    files_open.append(f)

    color = colors[i % len(colors)]
    for key, expr, cut, title, nbins, lo, hi in QUANTITIES:
        if not tree.GetBranch(expr):
            print(f"{name}: no branch {expr}, skipped in that pad (rerun run_scan.py)")
            continue
        h = ROOT.TH1D(f"h_{key}_{i}", title, nbins, lo, hi)
        tree.Draw(f"{expr}>>h_{key}_{i}", cut, "goff")
        if h.GetEntries() == 0:
            continue  # e.g. muons that stop before the mid-plane have no angle
        h.SetDirectory(0)
        h.SetLineColor(color)
        h.SetLineWidth(2)
        h.SetFillStyle(1001)
        h.SetFillColorAlpha(color, ALPHA)
        hists[key].append((label(name), h))

if not any(hists.values()):
    sys.exit("nothing to plot")

for pad, (key, *_rest) in enumerate(QUANTITIES, start=1):
    canvas.cd(pad)
    if not hists[key]:
        continue
    legend = ROOT.TLegend(0.50, 0.60, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetHeader("Beam momentum, mean", "C")
    ymax = max(h.GetMaximum() for _, h in hists[key])
    for k, (lab, h) in enumerate(hists[key]):
        if k == 0:
            h.SetMaximum(1.2 * ymax)
        h.Draw("HIST" if k == 0 else "HIST SAME")
        legend.AddEntry(h, f"{lab}, mean {h.GetMean():.4g}", "lf")
    legend.Draw()
    legends[key].append(legend)
canvas.Update()

for ext in ("png", "pdf"):
    canvas.SaveAs(f"analyzed_end_points.{ext}")
print("wrote analyzed_end_points.png and analyzed_end_points.pdf")

wait_for_key()
