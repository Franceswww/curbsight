"""export_tracks.py - fused tracks out of the pipeline as map data and as a trajectory picture.

    python export_tracks.py --fused report_20261103/crossing2_fused.csv --out export/crossing2 \
        [--start 15:02:18] [--date 2026-10-19] [--min-s 3] [--lot-file ...] [--lane-field ...]

Reads a fuse_tracks.py fused.csv (east/north metres in the aerial frame) and writes

    tracks.csv          one row per object per processed frame, with wall clock, lat/lon and
                        aerial-image pixel beside the metres
    tracks_summary.csv  one row per object: class, first/last time, duration, path length, mean
                        and top speed, straight-line start -> end
    trajectories_vehicles.png, trajectories_pedestrians.png
                        every path drawn on the aerial photo

What is drawn, and what is not. Every object is drawn from its smoothed centre, never from the raw
foot point (the mask's bottom edge wobbles half a metre between frames and draws thorns).

Vehicles pass two physical tests (vehicle_runs): the position must lie on the lane network learned
from the traffic itself (--lane-field, built by build_lane_field.py: a 2 m cell fewer than three
vehicles ever used is not road), and the path must turn no tighter than the body can (curvature_ok:
over 90 degrees within 3 m, or 1.5 times the minimum-radius allowance within 8 m, is a loop or a
hairpin). A standing vehicle is held at one position for the whole stop (hold_standing), so a car
at the light is a point on its line and not a scribble, and is not cut. Vehicles inside the painted
bays of --lot-file are not traffic and are not drawn; the aisle is a road.

People and bicycles pass two (person_runs): no step faster than the body can move (4 m/s, 10 for a
bicycle) and no position beyond what the cameras can see (55 m from the pole).

What fails a test is left out and the path breaks there; stretches too short to mean anything are
discarded (2 s and 5 m for a vehicle unless it is a stand, 1 s for a person). At the end two reports
print: plausibility (what was drawn that should not have been) and retention (what was not drawn
that might have been), because a rule that deletes a good track makes the first report look better.
"""
import argparse
import csv
import glob
from collections import defaultdict
import json
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTR = os.path.join(HERE, "..", "calibration")
LAT0, LON0 = 33.94760, -83.37630                                  # the aerial frame's origin
KX, KY = 111320.0 * math.cos(math.radians(LAT0)), 111320.0
VEHICLE = {"Passenger Vehicle", "Motorcycle", "Bus", "Truck"}
PALETTE = [(60, 76, 231), (74, 195, 139), (219, 152, 52), (182, 89, 155), (60, 180, 240),
           (34, 126, 230), (113, 204, 46), (156, 100, 12), (43, 57, 192), (96, 174, 39),
           (18, 156, 243), (133, 160, 22), (80, 62, 44), (90, 33, 41), (125, 185, 222)]


MIN_RADIUS = {"Passenger Vehicle": 4.5, "Truck": 6.0, "Bus": 9.0, "Motorcycle": 2.5}   # m, generous


def load_lane_field(path):
    """build_lane_field.py's output: per 2 m cell, how many vehicles used it and which way they went"""
    j = json.load(open(path))
    cells = {tuple(int(x) for x in k.split(",")): (v["vehicles"], v["concentration"], math.radians(v["lane_deg"]))
             for k, v in j["cells"].items()}
    return dict(cell=float(j["cell_m"]), cells=cells)


def on_lane(field, p_prev, p, min_vehicles=3, min_conc=0.7, max_dev_deg=45.0):
    """Is this vehicle position somewhere vehicles go, heading the way they go there?

    A cell fewer than min_vehicles distinct vehicles ever drove through is not road: that is where
    a runaway centre, a merged pair of cars averaged onto the plaza, or a car parked on the grass
    ends up. Where the cell's traffic agrees on a direction (concentration above min_conc), a step
    running across it is not a vehicle either. In the junction box traffic goes every way, the
    concentration is low, and the direction test stays out of it, so turns are never touched.
    """
    c = field["cell"]
    i, j = math.floor(p[0] / c), math.floor(p[1] / c)
    f = field["cells"].get((i, j))
    # No borrowing from neighbouring cells. Borrowing was tried to keep the thin far end of the
    # lanes, where a bus 50 m out crossed cells only one or two vehicles had used in 71 minutes, and
    # it let the edge of a plaza one cell from the kerb count as road. A vehicle where no vehicle
    # goes costs more than the last few metres of coverage, so the far end is the price.
    if f is None or f[0] < min_vehicles:
        return False
    nveh, conc, lane = f
    step = (p[0] - p_prev[0], p[1] - p_prev[1])
    if math.hypot(*step) < 0.3 or conc < min_conc:
        return True
    dev = abs((math.atan2(step[1], step[0]) - lane + math.pi / 2) % math.pi - math.pi / 2)
    return dev <= math.radians(max_dev_deg)


def curvature_ok(P, cls, standing=None, step_m=1.0):
    """Per point: False where the path turns tighter than a body with this turning radius can.

    This is the loop and hairpin test. A standing bus whose solved centre swings around the box
    draws a circle three metres across; a vehicle leaving the frame half-visible hooks back on
    itself; nothing on wheels does either. Headings come from points step_m apart along the path,
    and the rotation is measured over the last 3 m (over 90 degrees is a hairpin for any class)
    and the last 8 m (more than 1.5 times what the minimum radius allows). The whole stretch that
    turned is marked, not just its last point.

    step_m is 1 m, not less: at 0.3 m the heading of a car creeping to the kerb is noise, and a
    car pulling over, standing 33 s and pulling away lost 294 of its 340 rows to "turns" that were
    nothing but that noise. A loop or a hairpin spans two or three metres and is seen just as well
    at 1 m. Rows the caller has marked standing are never failed here: a held position has no
    heading to judge.
    """
    n = len(P)
    ok = np.ones(n, bool)
    idx = [0]
    for i in range(1, n):
        if np.hypot(*(P[i] - P[idx[-1]])) >= step_m:
            idx.append(i)
    if len(idx) < 3:
        return ok
    Q = P[idx]
    d = np.diff(Q, axis=0)
    L = np.hypot(*d.T)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    s = np.concatenate([[0.0], np.cumsum(L)])
    lim3, lim8 = math.radians(90), 1.5 * (8.0 / MIN_RADIUS.get(cls, 4.5))
    j3 = j8 = 0
    for i in range(len(th)):
        while s[i] - s[j3] > 3.0:
            j3 += 1
        while s[i] - s[j8] > 8.0:
            j8 += 1
        if abs(th[i] - th[j3]) > lim3:
            ok[idx[j3] + 1: idx[i + 1] + 1] = False
        elif abs(th[i] - th[j8]) > lim8:
            ok[idx[j8] + 1: idx[i + 1] + 1] = False
    if standing is not None:
        ok[standing] = True
    return ok


def hold_standing(T, P, win_s=1.5, move_m=2.0, hold_m=1.0):
    """A vehicle that is not going anywhere is drawn where it stands, once.

    While a vehicle stands, its solved centre still wanders as the cameras see a little more or
    less of it, and over a long stop that wander draws a circle or a scribble around a car that
    never moved. Cutting those rows was the first answer and it emptied the map of every car
    waiting at the light while the camera views still showed them. So the standing rows are kept
    and given one position, the median of the stretch: a car at the light is a box that does not
    twitch, and in the path figure a stop is a point on the line, not a knot.

    Standing: the position moved less than move_m within the surrounding +-win_s. 2 m over 3 s is
    0.67 m/s, slower than a vehicle drives but not slower than its solved centre wanders: on the
    two-minute sample the centre of a vehicle going nowhere moved under 1.25 m in 90 % of 3 s
    windows and under 1.96 m in 95 %. At 1 m the buses waiting at the far stop fell out of the
    standing test, their wander then failed the curvature test, and they were not drawn for 10 to
    20 s each while the cameras plainly showed them. At 2 m those stands are kept (57 % of vehicle
    rows drawn instead of 54 %) with the same zero reversals.

    A stand that creeps, a queue moving up a car length over twenty seconds, passes the standing
    test row by row yet spans several metres as a whole; one median for all of it would make the
    drawn position jump those metres in a single frame (5 m, 78 m/s, on the 71-minute export). So
    a stand is held in steps: each held stretch spans at most hold_m, and the position moves on to
    the next step's median, never further than hold_m at a time.
    """
    n = len(P)
    held = P.copy()
    if n < 3:
        return held, np.zeros(n, bool)
    standing = np.zeros(n, bool)
    for i in range(n):
        j0 = np.searchsorted(T, T[i] - win_s)
        j1 = np.searchsorted(T, T[i] + win_s, side="right")
        seg = P[j0:j1]
        standing[i] = np.hypot(*(seg.max(0) - seg.min(0))) < move_m
    i = 0
    while i < n:
        if not standing[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and standing[j + 1]:
            j += 1
        s = i
        while s <= j:
            e = s
            while e + 1 <= j and np.hypot(*(P[s:e + 2].max(0) - P[s:e + 2].min(0))) <= hold_m:
                e += 1
            held[s:e + 1] = np.median(P[s:e + 1], axis=0)
            s = e + 1
        i = j + 1
    return held, standing


def vehicle_runs(T, P, cls, field, min_s=2.0, min_m=5.0):
    """The stretches of a vehicle path worth drawing: on the lane network, turning like a vehicle,
    and each at least min_s long and either min_m from end to end or a stand of at least min_s
    (a car waiting at the light goes nowhere and is still a car). Standing stretches are held at one
    position first (hold_standing). Returns (list of index arrays, the positions to draw)."""
    P, standing = hold_standing(T, P)
    keep = np.ones(len(P), bool)
    if field is not None and len(P) > 1:
        for i in range(1, len(P)):
            keep[i] = on_lane(field, P[i - 1], P[i])
        keep[0] = keep[1]
    keep &= curvature_ok(P, cls, standing)
    runs, i, n = [], 0, len(P)
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and keep[j + 1]:
            j += 1
        if T[j] - T[i] >= min_s and (np.hypot(*(P[j] - P[i])) >= min_m or standing[i:j + 1].mean() > 0.5):
            runs.append(np.arange(i, j + 1))
        i = j + 1
    return runs, P


HEAD_M, HEAD_HALFW_M = 0.8, 0.26                                  # arrow head, metres on the ground


MAX_SPEED = {"Pedestrian": 4.0, "Bicycle": 10.0}      # m/s; a sprint is 3 and a fast cyclist 9
CAMERA_REACH_M = 55.0                                    # detections stop at 45 m; past this is extrapolation


def load_pole():
    """the cameras' mean position, from the solved models; the reach test is measured from here"""
    cs = []
    for p in glob.glob(os.path.join(OUTR, "*_fit_pnp", "cam.json")):
        cs.append(json.load(open(p))["centre"][:2])
    return np.mean(cs, axis=0) if cs else None


def person_runs(T, P, cls, pole, min_s=1.0):
    """The stretches of a pedestrian or bicycle path worth drawing: moving no faster than a body of
    that kind can, and within what the cameras can see. Returns a list of index arrays.

    The position is the smoothed centre and nothing else. An earlier version fell back to the
    measured foot point when the centre looked lost, and the foot point is the mask's bottom edge:
    it wobbles half a metre between frames, and 313 pedestrian paths came out as rows of thorns.
    The centre is smooth where the smoother was following its object; where it was not, the
    centre either jumps (caught by speed) or drifts away (caught by reach), and either way that
    stretch is cut and the path breaks there.
    """
    n = len(P)
    keep = np.ones(n, bool)
    vmax = MAX_SPEED.get(cls, 4.0)
    for i in range(1, n):
        dt = max(T[i] - T[i - 1], 1e-3)
        if np.hypot(*(P[i] - P[i - 1])) / dt > vmax:
            keep[i] = False
    if pole is not None:
        keep &= np.hypot(*(P - pole).T) <= CAMERA_REACH_M
    runs, i = [], 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and keep[j + 1]:
            j += 1
        if T[j] - T[i] >= min_s:
            runs.append(np.arange(i, j + 1))
        i = j + 1
    return runs


def arrow_head(img, a, b, col, res=0.04):
    """A slim filled triangle at b, pointing away from a.

    cv2.arrowedLine opens its head at 45 degrees and sizes it as a fraction of the shaft, so a long
    path gets a huge head and a short one gets none. Here the head is a fixed size on the ground,
    which is what the reader is actually judging, and narrow enough not to bury the path's own end.
    """
    v = np.array(b, float) - np.array(a, float)
    L = float(np.hypot(*v))
    if L < 1:
        return
    u = v / L
    n = np.array([-u[1], u[0]])
    back = np.array(b, float) - u * (HEAD_M / res)
    cv2.fillConvexPoly(img, np.array([b, back + n * (HEAD_HALFW_M / res),
                                      back - n * (HEAD_HALFW_M / res)], np.int32), col, cv2.LINE_AA)


def plausibility(by_gid, fps, step_m=0.4):
    """Count the things a vehicle cannot physically do, and say so before anyone looks at the figure.

    A reversal is two consecutive steps, each longer than step_m, that turn by more than 120
    degrees: a car that goes forward a metre and then straight back. Real traffic does this never;
    a position that flips between two estimates does it every frame. The first version of the
    centre clamp in this file did exactly that to 22 bus paths and nobody noticed until the figure
    was already published, because nothing counted. Now something counts.
    """
    bad = []
    slow = 0
    for gid, os_ in by_gid.items():
        if len(os_) < 20:
            continue
        F = np.array([o["frame"] for o in os_])
        inside = np.diff(F) <= 3                                           # steps within one drawn piece only
        if os_[0]["cls"] not in VEHICLE:
            P = np.array([[o["E"], o["N"]] for o in os_]); T = np.array([o["t_sec"] for o in os_])
            v = np.hypot(*np.diff(P, axis=0).T) / np.maximum(np.diff(T), 1.0 / fps)
            slow += bool((v[inside] > 1.05 * MAX_SPEED.get(os_[0]["cls"], 4.0)).any())   # 5% for t_sec rounding
            continue
        P = np.array([[o["E"], o["N"]] for o in os_])
        T = np.array([o["t_sec"] for o in os_])
        d = np.diff(P, axis=0)
        L = np.hypot(*d.T)
        dt = np.maximum(np.diff(T), 1.0 / fps)                            # a gap is real time, not one frame
        rev = sum(1 for i in range(1, len(d))
                  if L[i] > step_m and L[i - 1] > step_m and float(d[i] @ d[i - 1]) / (L[i] * L[i - 1]) < -0.5)
        top = float(np.max(L / dt)) if len(L) else 0.0                   # fastest step, m/s over its real duration
        if rev or top > 30:
            bad.append((rev, top, gid, os_[0]["cls"]))
    bad.sort(reverse=True)
    print("plausibility: %d vehicle paths with a reversal or a step over 30 m/s; %d pedestrian or bicycle "
          "paths with a step faster than the body can move" % (len(bad), slow))
    for rev, top, gid, cls in bad[:8]:
        print("   gid %-6d %-18s %3d reversals, fastest step %5.1f m/s" % (gid, cls, rev, top))
    return len(bad)


def retention(fused_rows, by_gid, fps):
    """Say what was thrown away, not only what was kept.

    The plausibility count above looks for bad geometry that IS drawn. It cannot see good data that
    is NOT drawn: a rule that deleted 294 of a car's 340 rows, its entire 33 s stop included, made
    the plausibility count better, and nobody noticed until the car was missing from the video. So
    this reports, per class, how many of the fused rows reached the output, and names the vehicles
    that lost the most of their time, so a rule that eats a good track shows up as a number.
    """
    # seconds come from frame spans, not row counts: the sample was processed at every second frame,
    # the long recording at every frame, and a row count would halve one or double the other
    span = lambda rs_: (max(int(r["frame"]) for r in rs_) - min(int(r["frame"]) for r in rs_) + 1) / fps if rs_ else 0.0
    had = {g: len(rs) for g, rs in fused_rows.items()}
    had_s = {g: span(rs) for g, rs in fused_rows.items()}
    kept = {g: len(os_) for g, os_ in by_gid.items()}
    kept_s = {g: span(os_) for g, os_ in by_gid.items()}
    cls_of = {g: rs[0]["cls"] for g, rs in fused_rows.items()}
    per = defaultdict(lambda: [0, 0])
    for g, n in had.items():
        k = "vehicles" if cls_of[g] in VEHICLE else cls_of[g]
        per[k][0] += n
        per[k][1] += kept.get(g, 0)
    print("retention: " + "; ".join("%s %d%% of rows" % (k, round(100 * v[1] / v[0])) for k, v in sorted(per.items()) if v[0]))
    lost = sorted((had_s[g] - kept_s.get(g, 0.0), g, cls_of[g], had_s[g], kept_s.get(g, 0.0))
                  for g in had if cls_of[g] in VEHICLE and had_s[g] >= 5.0)
    for lost_s, g, cls, h_s, k_s in sorted(lost, reverse=True)[:5]:
        if lost_s >= 5:
            print("   vehicle gid %-6d %-18s had %5.1f s, drawn %5.1f s, lost %5.1f s" % (g, cls, h_s, k_s, lost_s))


def pieces(os_, mpx, fps, gap_s=1.0):
    """a path split wherever a processed frame is missing, so a cut stretch is a break and never a
    line drawn across it; every second frame was processed, so consecutive rows are two frames apart"""
    out, run = [], []
    for o in os_:
        if run and o["frame"] - run[-1]["frame"] > 3:
            out.append(run); run = []
        run.append(o)
    if run:
        out.append(run)
    return [[mpx(o["E"], o["N"]) for o in seg] for seg in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=15.0, help="frame rate the frame column counts in")
    ap.add_argument("--start", default=None, help="wall-clock time of frame 0, HH:MM:SS")
    ap.add_argument("--date", default=None)
    ap.add_argument("--lot-file", default=None, help="a parking area drawn by hand (see "
                    "calibration/lot_sanford_cedar.json): vehicles in its bays are not traffic")
    ap.add_argument("--lane-field", default=os.path.join(OUTR, "lane_field_sanford_cedar.json"),
                    help="build_lane_field.py output; vehicle positions off it are not drawn")
    ap.add_argument("--min-s", type=float, default=3.0, help="paths shorter than this are drawn grey")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Same hand-drawn parking area the renderer uses, so the exported tracks and the demo agree.
    lot = json.load(open(a.lot_file)) if a.lot_file else None

    def zone(e, n):
        if lot is None:
            return "aisle"
        R = lot["rect"]
        if not (R[0] <= e <= R[2] and R[1] <= n <= R[3]):
            return "aisle"
        if e < lot["ignore_west_of"]:
            return "ignore"
        for b in lot["bays"]:
            lo, hi = b["e_range"]
            if not lo <= e <= hi:
                continue
            (e1, n1_), (e2, n2_) = b["line"]
            edge = n1_ + (n2_ - n1_) * (e - e1) / (e2 - e1)
            back = edge + b["depth"] if b["side"] == "north" else edge - b["depth"]
            if (edge <= n <= back) if b["side"] == "north" else (back <= n <= edge):
                return "bay"
        return "aisle"

    rows = {}
    with open(a.fused, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(int(r["gid"]), []).append(r)
    for rs in rows.values():
        rs.sort(key=lambda r: int(r["frame"]))

    t0 = None
    if a.start:
        hh, mm, ss = (float(v) for v in a.start.split(":"))
        t0 = hh * 3600 + mm * 60 + ss

    field = load_lane_field(a.lane_field) if a.lane_field and os.path.exists(a.lane_field) else None
    if field is None:
        print("no lane field: vehicles are checked for curvature only (build one with build_lane_field.py)")
    pole = load_pole()
    out_rows, summary, lot_hidden, not_vehicle, not_person, cut_rows = [], [], 0, 0, 0, 0
    for gid, rs in sorted(rows.items()):
        cls = rs[0]["cls"]
        if cls in VEHICLE:
            # A vehicle is drawn from its smoothed centre, and only where that centre behaves like a
            # vehicle: on the lane network and turning no tighter than its body can. Everything the
            # earlier row-by-row guards tried to catch (centres flying off, two cars averaged onto a
            # plaza, parked cars wandering, loops at a bus stop) fails one of those two tests.
            if lot is not None:                                   # bays and the dead end are not traffic
                rs = [r for r in rs if zone(float(r["Ec"] or r["E"]), float(r["Nc"] or r["N"])) == "aisle"]
                if len(rs) < 2:
                    lot_hidden += 1
                    continue
            frames = np.array([int(r["frame"]) for r in rs])
            P = np.array([[float(r["Ec"] or r["E"]), float(r["Nc"] or r["N"])] for r in rs])
            runs, P = vehicle_runs(frames / a.fps, P, cls, field)
            if not runs:
                not_vehicle += 1
                continue
            keep = np.concatenate(runs)
            cut_rows += len(rs) - len(keep)
            rs = [rs[i] for i in keep]
            pts = [(int(frames[i]), float(P[i][0]), float(P[i][1])) for i in keep]
        else:
            frames = np.array([int(r["frame"]) for r in rs])
            P = np.array([[float(r["Ec"] or r["E"]), float(r["Nc"] or r["N"])] for r in rs])
            runs = person_runs(frames / a.fps, P, cls, pole)
            if not runs:
                not_person += 1
                continue
            keep = np.concatenate(runs)
            cut_rows += len(rs) - len(keep)
            rs = [rs[i] for i in keep]
            pts = [(int(frames[i]), float(P[i][0]), float(P[i][1])) for i in keep]

        path = 0.0
        for k in range(len(pts) - 1):
            path += math.dist(pts[k][1:], pts[k + 1][1:])
        dur = (pts[-1][0] - pts[0][0]) / a.fps
        speeds = []
        for k in range(len(pts) - 1):
            dt = (pts[k + 1][0] - pts[k][0]) / a.fps
            if dt > 0:
                speeds.append(math.dist(pts[k][1:], pts[k + 1][1:]) / dt)
        for r, (fr, e, n) in zip(rs, pts):
            t = fr / a.fps
            o = {"gid": gid, "cls": cls, "frame": fr, "t_sec": round(t, 3),
                 "E": round(e, 3), "N": round(n, 3),
                 "lat": round(LAT0 + n / KY, 7), "lon": round(LON0 + e / KX, 7),
                 "hdg": r["hdg"], "n_cams": r["n_cams"], "filled": r["filled"]}
            if t0 is not None:
                s_ = t0 + t
                o["time"] = "%02d:%02d:%06.3f" % (int(s_ // 3600), int(s_ % 3600 // 60), s_ % 60)
                if a.date:
                    o["date"] = a.date
            out_rows.append(o)
        summary.append({"gid": gid, "cls": cls, "frames": len(pts),
                        "t_start": round(pts[0][0] / a.fps, 2), "t_end": round(pts[-1][0] / a.fps, 2),
                        "duration_s": round(dur, 2), "path_m": round(path, 2),
                        "straight_m": round(math.dist(pts[0][1:], pts[-1][1:]), 2),
                        "mean_speed_ms": round(path / dur, 2) if dur > 0 else "",
                        "top_speed_ms": round(max(speeds), 2) if speeds else "",
                        "E_start": round(pts[0][1], 2), "N_start": round(pts[0][2], 2),
                        "E_end": round(pts[-1][1], 2), "N_end": round(pts[-1][2], 2)})

    with open(os.path.join(a.out, "tracks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    with open(os.path.join(a.out, "tracks_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    base_full = cv2.imread(os.path.join(OUTR, "aerial_wide.png"))
    ext = json.load(open(os.path.join(OUTR, "aerial_wide.json")))
    E0, N1, RES = ext["E0"], ext["N1"], ext["res"]
    mpx = lambda e, n: (int(round((e - E0) / RES)), int(round((N1 - n) / RES)))

    by_gid = {}
    for o in out_rows:
        by_gid.setdefault(o["gid"], []).append(o)
    for tag, want in (("vehicles", True), ("pedestrians", False)):
        img = base_full.copy()
        drawn = 0
        for gid, os_ in sorted(by_gid.items()):
            if (os_[0]["cls"] in VEHICLE) != want:
                continue
            segs = pieces(os_, mpx, a.fps)
            p = [q for seg in segs for q in seg]
            dur = os_[-1]["t_sec"] - os_[0]["t_sec"]
            if dur < a.min_s:
                for seg in segs:
                    cv2.polylines(img, [np.array(seg, np.int32)], False, (150, 150, 150), 1, cv2.LINE_AA)
                continue
            col = PALETTE[gid % len(PALETTE)]
            for seg in segs:
                cv2.polylines(img, [np.array(seg, np.int32)], False, col, 3, cv2.LINE_AA)
            cv2.circle(img, p[0], 7, col, -1)                      # start
            cv2.circle(img, p[0], 7, (255, 255, 255), 1, cv2.LINE_AA)
            last = segs[-1]                                        # the arrow belongs to the last piece
            if len(last) > 2:
                k = len(last) - 1
                while k > 0 and math.dist(last[k], last[-1]) < 3.0 / RES:
                    k -= 1
                arrow_head(img, last[k], last[-1], col, RES)
            drawn += 1
        label = "%s: %d paths >= %.0f s" % (tag, drawn, a.min_s)
        cv2.putText(img, label, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, label, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(a.out, "trajectories_%s.png" % tag), img)
        print("%s: %d paths drawn" % (tag, drawn))

    plausibility(by_gid, a.fps)
    retention(rows, by_gid, a.fps)
    print("%d objects, %d rows written; %d vehicles hidden in the parking bays; %d vehicles and %d people "
          "with no stretch that moves like one; %d rows cut (off the lanes, inside a turn too tight, "
          "faster than a body can move, or beyond the cameras' reach)"
          % (len(summary), len(out_rows), lot_hidden, not_vehicle, not_person, cut_rows))


if __name__ == "__main__":
    main()
