#!/usr/bin/env python3
"""3D event display of photon hits on the silicon plates of the water Cherenkov tank.

Usage: ./eventDisplayPMTs.py [file.root]   (default: muonDecay.root, tree "silicon")
Keys: Left/Right = previous/next event.
"""
import sys

import matplotlib

if "--test" not in sys.argv:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.colors import LogNorm, Normalize
from matplotlib.widgets import Button, CheckButtons

BG = "#101014"
FG = "#dddddd"

# Silicon centres of all modules, mirroring BuildSensors() in waterCherenkovSim.cpp (mm)
R_IN, HALF_Z = 1746.0, 1996.0
R_OUT, SI_T, GAP, PITCH = 40.0, 5.0, 0.5, 100.0


def all_sensor_positions():
    r_si = R_IN - GAP - SI_T / 2
    n_phi = int(2 * np.pi * R_IN / PITCH)
    nz = int((HALF_Z - 100.0) / PITCH)
    pts = []
    for k in range(-nz, nz + 1):
        for i in range(n_phi):
            phi = 2 * np.pi * i / n_phi
            pts.append((r_si * np.cos(phi), r_si * np.sin(phi), k * PITCH))
    nxy = int((R_IN - R_OUT) / PITCH)
    for sgn in (-1, 1):
        z = sgn * (HALF_Z - GAP - SI_T / 2)
        for i in range(-nxy - 1, nxy + 2):
            for j in range(-nxy - 1, nxy + 2):
                x, y = i * PITCH, j * PITCH
                if np.hypot(x, y) + R_OUT <= R_IN:
                    pts.append((x, y, z))
    return np.array(pts)


class EventDisplay:
    def __init__(self, path):
        t = uproot.open(path)["silicon"]
        d = t.arrays(["eventID", "x_mm", "y_mm", "z_mm", "nHits"], library="np")
        self.ids = d["eventID"]
        self.x, self.y, self.z, self.n = d["x_mm"], d["y_mm"], d["z_mm"], d["nHits"]
        self.order = np.argsort(self.ids, kind="stable")  # merged files may be unordered
        self.pos = 0
        self.all = all_sensor_positions()
        self.vmax = max(1, max((a.max() for a in self.n if len(a)), default=1))
        self.show_all = True
        self.log = True

        self.fig = plt.figure(figsize=(11, 8), facecolor=BG)
        self.fig.canvas.manager.set_window_title(f"Event display - {path}")
        self.ax = self.fig.add_axes([0.0, 0.1, 0.87, 0.9], projection="3d", facecolor=BG)
        self.cax = self.fig.add_axes([0.9, 0.2, 0.02, 0.65], facecolor=BG)
        self.cbar = None

        bkw = dict(color="#2a2a35", hovercolor="#44445a")
        self.bprev = Button(self.fig.add_axes([0.25, 0.02, 0.12, 0.05]), "Previous", **bkw)
        self.bnext = Button(self.fig.add_axes([0.40, 0.02, 0.12, 0.05]), "Next", **bkw)
        for b in (self.bprev, self.bnext):
            b.label.set_color(FG)
        self.bprev.on_clicked(lambda _e: self.step(-1))
        self.bnext.on_clicked(lambda _e: self.step(+1))
        cax = self.fig.add_axes([0.60, 0.01, 0.2, 0.08], facecolor=BG)
        self.chk = CheckButtons(cax, ["show all plates", "log colour"], [True, True])
        for lab in self.chk.labels:
            lab.set_color(FG)
        self.chk.on_clicked(self.toggle)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def toggle(self, label):
        if label == "show all plates":
            self.show_all = not self.show_all
        else:
            self.log = not self.log
        self.draw()

    def on_key(self, ev):
        if ev.key == "right":
            self.step(+1)
        elif ev.key == "left":
            self.step(-1)

    def step(self, d):
        self.pos = (self.pos + d) % len(self.ids)
        self.draw()

    def draw(self):
        ax = self.ax
        # keep the user's current camera when switching events
        elev, azim = ax.elev, ax.azim
        ax.clear()
        ax.set_facecolor(BG)
        i = self.order[self.pos]
        x, y, z, n = self.x[i], self.y[i], self.z[i], self.n[i]

        if self.show_all:
            a = self.all
            ax.scatter(a[:, 0], a[:, 2], a[:, 1], s=4, c="#2c2c36", marker="o", depthshade=False)
        if len(n):
            norm = LogNorm(1, max(2, self.vmax)) if self.log else Normalize(0, self.vmax)
            sc = ax.scatter(x, z, y, c=n, cmap="plasma", norm=norm, s=26, marker="o",
                            depthshade=False, edgecolors="none")
            self.cax.clear()
            self.cbar = self.fig.colorbar(sc, cax=self.cax)
            self.cbar.set_label("detected photons", color=FG)
            self.cbar.ax.yaxis.set_tick_params(color=FG, labelcolor=FG)
        else:
            self.cax.clear()
            self.cax.set_axis_off()

        # Detector axis (z) is drawn vertically
        lim = HALF_Z
        ax.set_xlim(-R_IN, R_IN)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-R_IN, R_IN)
        ax.set_box_aspect((2 * R_IN, 2 * lim, 2 * R_IN))
        ax.set_xlabel("x [mm]", color=FG)
        ax.set_ylabel("z [mm]", color=FG)
        ax.set_zlabel("y [mm]", color=FG)
        ax.tick_params(colors=FG, labelsize=7)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.06, 0.06, 0.08, 1.0))
            axis._axinfo["grid"]["color"] = (0.25, 0.25, 0.3, 0.5)
        ax.view_init(elev=elev, azim=azim)
        self.fig.suptitle(
            f"Event {self.ids[i]}  ({self.pos + 1}/{len(self.ids)})   "
            f"plates hit: {len(n)}   total photons: {int(np.sum(n))}",
            color=FG, y=0.97,
        )
        self.fig.canvas.draw_idle()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    disp = EventDisplay(args[0] if args else "muonDecay.root")
    if "--test" in sys.argv:
        disp.step(1)
        disp.fig.savefig("/tmp/evd.png", facecolor=BG)
        return
    plt.show()


if __name__ == "__main__":
    main()
