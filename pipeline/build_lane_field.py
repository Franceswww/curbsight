"""
build_lane_field.py - where vehicles go, learned from the vehicles.

    python build_lane_field.py --tracks ../out/tracks.csv --out ../calibration/lane_field_sanford_cedar.json

The ground is cut into 2 m cells. For each cell this records how many distinct vehicles ever drove
through it and the direction they were going, as an axial mean (a lane carries traffic both ways,
so north-bound and south-bound agree) with its concentration. A vehicle position in a cell that no
other vehicle ever used, or a vehicle step that runs across the traffic in a cell where everyone
else agrees on the direction, is not a vehicle on the road. export_tracks.py uses this file to
leave such points out.

Only steps of 0.3 m or more count toward the direction: a standing vehicle's jitter has no
direction and would only dilute the lane. Build it from the longest recording you have. One hour
of traffic draws the lanes; one minute does not.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

VEHICLE = {"Passenger Vehicle", "Motorcycle", "Bus", "Truck"}


def build(tracks_csv, cell_m=2.0, min_step=0.3):
    rows = defaultdict(list)
    with open(tracks_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["cls"] in VEHICLE:
                rows[int(r["gid"])].append((float(r["t_sec"]), float(r["E"]), float(r["N"])))
    vehicles = defaultdict(set)
    sx, sy, n = defaultdict(float), defaultdict(float), defaultdict(int)
    for g, p in rows.items():
        p.sort()
        P = np.array([[e, nn] for _, e, nn in p])
        d = np.diff(P, axis=0)
        L = np.hypot(*d.T)
        for i in range(len(d)):
            if L[i] < min_step:
                continue
            c = (int(math.floor(P[i][0] / cell_m)), int(math.floor(P[i][1] / cell_m)))
            vehicles[c].add(g)
            th2 = 2 * math.atan2(d[i][1], d[i][0])            # doubled: both directions of a lane agree
            sx[c] += math.cos(th2); sy[c] += math.sin(th2); n[c] += 1
    cells = {}
    for c, vs in vehicles.items():
        R = math.hypot(sx[c], sy[c]) / n[c]
        cells["%d,%d" % c] = dict(vehicles=len(vs), steps=n[c], concentration=round(R, 3),
                                  lane_deg=round(math.degrees(0.5 * math.atan2(sy[c], sx[c])), 1))
    return dict(cell_m=cell_m, min_step_m=min_step, source=os.path.basename(tracks_csv),
                vehicles=len(rows), cells=cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="tracks.csv from export_tracks.py, the longer the better")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=2.0)
    a = ap.parse_args()
    field = build(a.tracks, a.cell)
    json.dump(field, open(a.out, "w"), indent=0)
    busy = sum(1 for c in field["cells"].values() if c["vehicles"] >= 3)
    firm = sum(1 for c in field["cells"].values() if c["vehicles"] >= 3 and c["concentration"] > 0.7)
    print("%d vehicles -> %d cells with traffic, %d with 3+ vehicles, %d of those with a firm lane direction -> %s"
          % (field["vehicles"], len(field["cells"]), busy, firm, a.out))


if __name__ == "__main__":
    main()
