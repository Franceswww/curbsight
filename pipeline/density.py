"""
density.py - presence on the ground, for recordings too busy to draw as lines.

    python density.py --tracks ../out/tracks.csv --out ../out/density_pedestrians.png
    python density.py --tracks ../out/tracks.csv --out ../out/density_vehicles.png --vehicles

Every row of tracks.csv is one object seen in one processed frame, so counting rows per cell counts
TIME: how long that half-metre of ground was occupied. Two thousand pedestrian paths drawn as lines
are an unreadable tangle; the same two thousand as time per cell show the footways, the crossings,
the waiting areas - and the benches, where people sit for minutes without going anywhere.

The square root is there because the range is enormous: a cell on a busy footway collects a few
seconds per person walking through, a bench collects whole minutes. Linear scaling would leave the
walking routes invisible beside the sitting spots.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "calibration")
VEHICLE = {"Passenger Vehicle", "Motorcycle", "Bus", "Truck"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="tracks.csv from export_tracks.py")
    ap.add_argument("--out", required=True, help="where to write the .png")
    ap.add_argument("--vehicles", action="store_true", help="vehicles instead of everyone else")
    ap.add_argument("--cell", type=float, default=0.5, help="grid cell, metres")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--skip", type=int, default=2, help="every Nth frame was processed")
    a = ap.parse_args()

    ext = json.load(open(os.path.join(CAL, "aerial_wide.json")))
    E0, N1, RES = ext["E0"], ext["N1"], ext["res"]
    base = cv2.imread(os.path.join(CAL, "aerial_wide.png"))
    H, W = base.shape[:2]
    gw, gh = int(W * RES / a.cell) + 1, int(H * RES / a.cell) + 1
    grid = np.zeros((gh, gw), np.float32)

    n, t_lo, t_hi = 0, None, None
    with open(a.tracks, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["cls"] in VEHICLE) != a.vehicles:
                continue
            gx = int((float(r["E"]) - E0) / a.cell)
            gy = int((N1 - float(r["N"])) / a.cell)
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy, gx] += 1
                n += 1
            t = r.get("time")
            if t:
                t_lo = t if t_lo is None else min(t_lo, t)
                t_hi = t if t_hi is None else max(t_hi, t)

    secs = grid * (a.skip / a.fps)                              # rows -> seconds of occupancy
    big = cv2.GaussianBlur(cv2.resize(secs, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 0.8 / RES)
    v = np.sqrt(np.clip(big, 0, np.percentile(big[big > 0], 99.5))) if (big > 0).any() else big
    v /= v.max() if v.max() > 0 else 1

    heat = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = np.clip(v * 2.2, 0, 1)[..., None]                   # see through to the photo where nobody goes
    img = (base * 0.55 * (1 - alpha) + base * 0.55 * alpha * 0.25 + heat * alpha * 0.75).astype(np.uint8)

    span = " %s-%s" % (t_lo[:5], t_hi[:5]) if t_lo else ""
    label = "%s presence%s; brightest cell %.0f s in %.1f m" % (
        "vehicle" if a.vehicles else "pedestrian", span, secs.max(), a.cell)
    cv2.putText(img, label, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, label, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(a.out, img)
    print("%d positions binned, brightest cell %.0f s -> %s" % (n, secs.max(), a.out))


if __name__ == "__main__":
    main()
