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

Parked vehicles. A vehicle that drives in and stops in a bay keeps the heading it had in the aisle,
because the heading is only believed after 2.5 m of travel, so the drawn rectangle sits across the
painted bay. The approach itself is traffic at the junction and is kept; the standing part is not
wanted. So a vehicle's LEADING and TRAILING stationary run is cut (it stays within --stop-r of its
first / last position for at least --stop-s): a car pulling out of a bay starts at the moment it
moves, a car parking ends at the moment it stops. A stop in the MIDDLE of a journey - a red light, a
queue, the bike box - is never cut. Pedestrians are never cut; standing at a kerb is real behaviour.
"""
import argparse
import csv
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


def curvature_ok(P, cls):
    """Per point: False where the path turns tighter than a body with this turning radius can.

    This is the loop and hairpin test. A standing bus whose solved centre swings around the box
    draws a circle three metres across; a vehicle leaving the frame half-visible hooks back on
    itself; nothing on wheels does either. Headings come from steps of at least 0.3 m, because a
    smaller step has no direction, and the rotation is measured over the last 3 m (over 90 degrees
    is a hairpin for any class) and the last 8 m (more than 1.5 times what the minimum radius
    allows). The whole stretch that turned is marked, not just its last point.
    """
    n = len(P)
    ok = np.ones(n, bool)
    idx = [0]
    for i in range(1, n):
        if np.hypot(*(P[i] - P[idx[-1]])) >= 0.3:
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
    return ok


def vehicle_runs(T, P, cls, field, min_s=2.0, min_m=5.0):
    """The stretches of a vehicle path worth drawing: on the lane network, turning like a vehicle,
    and each at least min_s long and min_m from end to end. Returns a list of index arrays."""
    keep = np.ones(len(P), bool)
    if field is not None and len(P) > 1:
        for i in range(1, len(P)):
            keep[i] = on_lane(field, P[i - 1], P[i])
        keep[0] = keep[1]
    keep &= curvature_ok(P, cls)
    runs, i, n = [], 0, len(P)
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and keep[j + 1]:
            j += 1
        if T[j] - T[i] >= min_s and np.hypot(*(P[j] - P[i])) >= min_m:
            runs.append(np.arange(i, j + 1))
        i = j + 1
    return runs


BODY = {"Pedestrian": (0.6, 0.5), "Passenger Vehicle": (4.5, 1.9), "Truck": (5.5, 2.1),
        "Bus": (10.5, 2.6), "Motorcycle": (2.0, 0.8), "Bicycle": (1.8, 0.6)}    # metres, length x width


def reach(cls):
    """how far an object's centre can possibly sit from its own foot point: half its body diagonal,
    plus 30 cm for where the foot point actually lands. The same bound centre_from_box already
    enforces inside its own solve (render_demo.py), applied once more to the fused result."""
    L, W = BODY.get(cls, (4.5, 1.9))
    return 0.5 * math.hypot(L, W) + 0.3


def centre_or_foot(rows, cls, clamp, runaway_share=0.05):
    """For one object's rows in frame order: where to draw it, or None for a row not to draw.
    Returns [(e, n, fell_back) | None, ...].

    Each row carries a measured foot point and a Kalman-smoothed centre. The centre is the better
    position while the smoother is tracking its object, and a runaway when it is not: on this
    recording one pedestrian's centre ends 676 m from a foot point that never left a 60 m circle.
    The push, the distance between the two, tells them apart: a real centre cannot be further from
    its foot point than the body allows, and every class's 99th-percentile push sits near one reach.

    The decision is made once per track, never per row. Before any of this existed, the figures
    drew the centre alone and had no reversal and no step over 30 m/s on 396 vehicle paths. Three
    row-wise versions then each drew artefacts: a hard switch flipped a bus every frame (489
    reversals), hysteresis made the switch itself a 6.8 m jump, and a continuous blend jumped
    whenever the foot point it was blending toward re-anchored on another part of the car. The
    centre is smooth and the foot point is not, and any row-wise mix of the two inherits the jumps.

    So: if more than `runaway_share` of a track's rows have a push beyond `clamp` reaches, the
    smoother has lost this object and the whole track is drawn from the median-smoothed foot
    point. Otherwise the whole track is drawn from the centre, and the rare rows beyond the bound
    are dropped, which the caller draws as a break in the line rather than a line across the gap.
    """
    if not rows or clamp <= 0 or not rows[0].get("Ec"):
        return [(float(r["E"]), float(r["N"]), False) for r in rows]
    hi = clamp * reach(cls)
    E = np.array([float(r["E"]) for r in rows]); N = np.array([float(r["N"]) for r in rows])
    half = 2
    Es = np.array([np.median(E[max(0, i - half):i + half + 1]) for i in range(len(E))])
    Ns = np.array([np.median(N[max(0, i - half):i + half + 1]) for i in range(len(N))])
    has = np.array([bool(r.get("Ec")) for r in rows])
    Ec = np.array([float(r["Ec"]) if r.get("Ec") else np.nan for r in rows])
    Nc = np.array([float(r["Nc"]) if r.get("Nc") else np.nan for r in rows])
    push = np.where(has, np.hypot(Ec - Es, Nc - Ns), 0.0)
    runaway = push > hi
    if runaway.mean() > runaway_share:                               # the smoother lost this object
        # For a vehicle there is nothing left to draw. The centre flies off, and the foot point on
        # such a track jumps several metres in a frame too (eight of these were the last paths the
        # plausibility check flagged, every one in this state). A vehicle is reported only when
        # its smoothed position stayed with its measurement. A pedestrian's centre is its foot
        # point give or take half a metre, so the smoothed foot point is a fair place to draw it.
        if cls in VEHICLE:
            return [None] * len(rows)
        return [(float(Es[i]), float(Ns[i]), True) for i in range(len(rows))]
    out = []
    for i in range(len(rows)):
        if not has[i]:
            out.append((float(E[i]), float(N[i]), False))
        elif runaway[i]:
            out.append(None)                                         # a break, not a line
        else:
            out.append((float(Ec[i]), float(Nc[i]), False))
    return out


HEAD_M, HEAD_HALFW_M = 0.8, 0.26                                  # arrow head, metres on the ground


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
    for gid, os_ in by_gid.items():
        if os_[0]["cls"] not in VEHICLE or len(os_) < 20:
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
    print("plausibility: %d vehicle paths with a reversal or a step over 30 m/s" % len(bad))
    for rev, top, gid, cls in bad[:8]:
        print("   gid %-6d %-18s %3d reversals, fastest step %5.1f m/s" % (gid, cls, rev, top))
    return len(bad)


def pieces(os_, mpx, fps, gap_s=1.0):
    """a path split wherever rows are missing, so a dropped stretch is a break and not a long line"""
    out, run = [], []
    for o in os_:
        if run and o["t_sec"] - run[-1]["t_sec"] > gap_s:
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
    ap.add_argument("--centre-clamp", type=float, default=3.0,
                    help="pedestrians and bicycles: a track with more than 5%% of rows pushed beyond "
                         "this many body-reaches from its own foot point is drawn from the smoothed "
                         "foot point; 0 disables the check")
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
    out_rows, summary, clamped, emptied, lot_hidden, not_vehicle, cut_rows = [], [], 0, 0, 0, 0, 0
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
            runs = vehicle_runs(frames / a.fps, P, cls, field)
            if not runs:
                not_vehicle += 1
                continue
            keep = np.concatenate(runs)
            cut_rows += len(rs) - len(keep)
            rs = [rs[i] for i in keep]
            pts = [(int(frames[i]), float(P[i][0]), float(P[i][1])) for i in keep]
        else:
            placed = centre_or_foot(rs, cls, a.centre_clamp)
            pts, rs2 = [], []
            for r, pos in zip(rs, placed):
                if pos is None:
                    clamped += 1
                    continue
                e, n, fell_back = pos
                pts.append((int(r["frame"]), e, n)); rs2.append(r)
                clamped += fell_back
            rs = rs2
            if len(pts) < 2:
                emptied += 1
                continue

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
    print("%d objects, %d rows written; vehicles: %d hidden in the parking bays, %d with no stretch "
          "that moves like a vehicle on the lanes, %d rows cut off the lanes or inside a turn too tight; "
          "pedestrians: %d rows placed at or dropped for a runaway centre, %d objects emptied"
          % (len(summary), len(out_rows), lot_hidden, not_vehicle, cut_rows, clamped, emptied))


if __name__ == "__main__":
    main()
