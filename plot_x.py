#!/usr/bin/env python3
"""Overlay the x_mm distribution from each muonDecay ROOT file, labelled by muon energy.

The energy is parsed from the file name, e.g. muonDecay_300MeV.root or muonDecay_1GeV.root.
Usage: ./plot_x.py [files.root ...]   (default: muonDecay_*.root)
Make files with: MUON_OUT=muonDecay_300MeV.root ./waterCherenkovSim run.mac
"""
import glob
import re
import sys

import ROOT

ROOT.gROOT.SetBatch(False)
ROOT.gStyle.SetOptStat(0)

files = sys.argv[1:] or sorted(glob.glob("muonDecay_*.root"))
if not files:
    sys.exit("no ROOT files found")

ENERGY = re.compile(r"(\d+(?:\.\d+)?)\s*(keV|MeV|GeV|TeV)", re.IGNORECASE)
SCALE = {"kev": 1e-3, "mev": 1.0, "gev": 1e3, "tev": 1e6}


def energy_mev(name):
    m = ENERGY.search(name)
    return float(m.group(1)) * SCALE[m.group(2).lower()] if m else float("inf")


def label(name):
    m = ENERGY.search(name)
    return f"{m.group(1)} {m.group(2)}" if m else name


files.sort(key=energy_mev)

colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2, ROOT.kMagenta, ROOT.kOrange + 7, ROOT.kCyan + 2]

# Shared binning so curves are comparable
xmin, xmax, nbins = -2000.0, 2000.0, 200

canvas = ROOT.TCanvas("c", "x distribution", 900, 650)
legend = ROOT.TLegend(0.65, 0.65, 0.88, 0.88)
legend.SetBorderSize(0)

hists = []  # keep references alive
files_open = []
ymax = 0.0
for i, name in enumerate(files):
    f = ROOT.TFile.Open(name)
    if not f or f.IsZombie():
        print(f"skipping {name}: cannot open")
        continue
    tree = f.Get("muonDecay")
    if not tree or tree.GetEntries() == 0:
        print(f"skipping {name}: no entries")
        continue

    h = ROOT.TH1D(f"h{i}", ";x [mm];entries", nbins, xmin, xmax)
    tree.Draw(f"x_mm>>h{i}", "", "goff")
    h.SetDirectory(0)
    h.SetLineColor(colors[i % len(colors)])
    h.SetLineWidth(2)
    ymax = max(ymax, h.GetMaximum())
    legend.AddEntry(h, label(name), "l")
    hists.append(h)
    files_open.append(f)

if not hists:
    sys.exit("nothing to plot")

hists[0].SetMaximum(1.2 * ymax)
hists[0].Draw("HIST")
for h in hists[1:]:
    h.Draw("HIST SAME")
legend.Draw()

stuff = []
stuff.append(hists)

canvas.SaveAs("x_distribution.png")
print("wrote x_distribution.png")
ROOT.gApplication.Run()
