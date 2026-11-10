"""export_tracks.py - fused tracks out of the pipeline as map data and as a trajectory picture.

    python export_tracks.py --fused report_20261103/crossing2_fused.csv --out export/crossing2 \
        [--start 15:02:18] [--date 2026-10-19] [--min-s 3]

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


def max_move(pts, fps, win=2.0):
    """Largest distance the object covers inside any `win`-second window.

    A parked vehicle's solved centre drifts as the cameras see more or less of it - up to 2.5 m over
    a minute here - so "it moved" cannot be read off the whole track. Over 2 s that drift stays small
    while anything actually driving clears several metres.
    """
    best, j = 0.0, 0
    for i in range(len(pts)):
        while (pts[i][0] - pts[j][0]) / fps > win:
            j += 1
        best = max(best, math.dist(pts[i][1:], pts[j][1:]))
    return best


def trim_parked(pts, stop_r, stop_s, fps):
    """pts: [(frame, E, N), ...] in frame order -> (i, j) to keep, or None if nothing moves."""
    n = len(pts)
    i = 0
    while i + 1 < n and math.dist(pts[i + 1][1:], pts[0][1:]) <= stop_r:
        i += 1
    if (pts[i][0] - pts[0][0]) / fps < stop_s:                     # the head is not a real stop
        i = 0
    j = n - 1
    while j > 0 and math.dist(pts[j - 1][1:], pts[-1][1:]) <= stop_r:
        j -= 1
    if (pts[-1][0] - pts[j][0]) / fps < stop_s:
        j = n - 1
    return (i, j) if j > i else None


GATE = {"Pedestrian": 1.2, "Bicycle": 1.5, "Motorcycle": 1.5,
        "Passenger Vehicle": 3.0, "Truck": 5.0, "Bus": 6.0}        # fuse_tracks.py's own linking gate


def camera_points(clipdir, cams):
    """every camera track's own ground point, per frame, for checking the fused average against"""
    import render_demo as RD
    out = {}
    for cam in cams:
        path = os.path.join(clipdir, cam, "trajectories.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rs = list(csv.DictReader(f))
        g = RD.Cam(cam, cam + "_fit_pnp").ground([[float(r["foot_x"]), float(r["foot_y"])] for r in rs])
        for r, p in zip(rs, g):
            out[(cam, int(r["track_id"]), int(r["frame"]))] = p
    return out


def cameras_disagree(r, cls, cam_pts, mult):
    """True when the cameras that made this fused row cannot have been looking at one object.

    fuse_tracks.py merges camera tracks with union-find. Each single union is inside its gate, but
    nothing checks the set that comes out: one object here is built from eight camera tracks and
    spans 66 m, so its position is the average of two different vehicles and lands where neither
    ever drove. Over 71 minutes half of all multi-camera rows have their cameras disagreeing by
    more than 5 m.

    There is no position to rescue in such a row. Taking the nearest camera's own measurement was
    tried and is worse: the nearest camera changes, so the drawn path teleports between the two
    vehicles and leaves a straight line across the scene. So the row is dropped, and the path is
    drawn in pieces, with a break wherever rows are missing.
    """
    if not cam_pts or mult <= 0:
        return False
    srcs = [q for q in r.get("src", "").split("|") if ":" in q]
    if len(srcs) < 2:
        return False
    pts = []
    for q in srcs:
        c, tid = q.rsplit(":", 1)
        p = cam_pts.get((c, int(tid), int(r["frame"])))
        if p is not None:
            pts.append(p)
    if len(pts) < 2:
        return False
    return max(math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:]) > mult * GATE.get(cls, 3.0)


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
    ap.add_argument("--stop-r", type=float, default=1.5, help="m a parked vehicle may wander")
    ap.add_argument("--stop-s", type=float, default=5.0, help="s before a standing vehicle is cut")
    ap.add_argument("--min-move", type=float, default=5.0,
                    help="m a vehicle must cover inside one 2 s window to count as driving")
    ap.add_argument("--min-net", type=float, default=8.0,
                    help="m between a vehicle's first and last kept position to count as going somewhere")
    ap.add_argument("--lot-file", default=None, help="a parking area drawn by hand (see "
                    "lot_sanford_cedar.json): rows in its bays, or in the dead end behind them, are "
                    "not traffic at the junction and are dropped")
    ap.add_argument("--min-s", type=float, default=3.0, help="paths shorter than this are drawn grey")
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--clipdir", default=None,
                    help="the directory holding <camera>/trajectories.csv; needed for --max-disagree")
    ap.add_argument("--cams", default="cam02_1019,cam03_1019,cam04_1019")
    ap.add_argument("--max-disagree", type=float, default=2.0,
                    help="drop a fused row whose cameras disagree by more than this many linking "
                         "gates: they were not looking at one object. 0 disables")
    ap.add_argument("--centre-clamp", type=float, default=3.0,
                    help="a track with more than 5%% of its rows pushed beyond this many body-reaches "
                         "from its own foot point is drawn from the smoothed foot point; on other "
                         "tracks such rows are left out; 0 disables the check")
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

    cam_pts = camera_points(a.clipdir, a.cams.split(",")) if a.clipdir else {}
    out_rows, summary, cut, dropped, clamped, split, emptied, lost_runaway = [], [], 0, 0, 0, 0, 0, 0
    for gid, rs in sorted(rows.items()):
        cls = rs[0]["cls"]
        kept_rs = [r for r in rs if not cameras_disagree(r, cls, cam_pts, a.max_disagree)]
        split += len(rs) - len(kept_rs)
        placed = centre_or_foot(kept_rs, cls, a.centre_clamp)
        if kept_rs and all(p is None for p in placed):              # a vehicle the smoother lost
            lost_runaway += 1
            continue
        pts, rs2 = [], []
        for r, pos in zip(kept_rs, placed):
            if pos is None:                                          # a runaway row on a sound track
                clamped += 1
                continue
            e, n, fell_back = pos
            pts.append((int(r["frame"]), e, n)); rs2.append(r)
            clamped += fell_back
        kept_rs = rs2
        if len(pts) < 2:
            emptied += 1
            continue
        rs = kept_rs
        lo, hi = 0, len(rs) - 1
        if cls in VEHICLE and not a.no_trim:
            keep = trim_parked(pts, a.stop_r, a.stop_s, a.fps)
            if keep is None:                                       # stands the whole time
                cut += len(rs); dropped += 1
                continue
            lo, hi = keep
            kept = pts[lo:hi + 1]
            # A standing car's solved centre drifts as the cameras see more or less of it: over
            # twenty minutes that wanders tens of metres in total while the car goes nowhere, and
            # a change of occlusion can jump it 4 m at once. So neither distance alone nor speed
            # alone tells drift from a slow real manoeuvre. Traffic has to do one of the two:
            # get somewhere, or at some point move at a driving pace.
            if (math.dist(kept[0][1:], kept[-1][1:]) < a.min_net
                    and max_move(kept, a.fps) < a.min_move):
                cut += len(rs); dropped += 1
                continue
            cut += len(rs) - len(kept)
        rs, pts = rs[lo:hi + 1], pts[lo:hi + 1]

        path = 0.0
        for k in range(len(pts) - 1):
            path += math.dist(pts[k][1:], pts[k + 1][1:])
        dur = (pts[-1][0] - pts[0][0]) / a.fps
        speeds = []
        for k in range(len(pts) - 1):
            dt = (pts[k + 1][0] - pts[k][0]) / a.fps
            if dt > 0:
                speeds.append(math.dist(pts[k][1:], pts[k + 1][1:]) / dt)
        if cls in VEHICLE and lot is not None:                     # bays and the dead end are not traffic
            keep_rows = [(r, p) for r, p in zip(rs, pts) if zone(p[1], p[2]) == "aisle"]
            cut += len(rs) - len(keep_rows)
            if len(keep_rows) < 2:
                dropped += 1
                continue
            rs = [r for r, _ in keep_rows]
            pts = [p for _, p in keep_rows]
        for r, (fr, e, n) in zip(rs, pts):
            t = fr / a.fps
            o = {"gid": gid, "cls": cls, "frame": fr, "t_sec": round(t, 3),
                 "E": round(e, 3), "N": round(n, 3),
                 "lat": round(LAT0 + n / KY, 7), "lon": round(LON0 + e / KX, 7),
                 "hdg": r["hdg"], "n_cams": r["n_cams"], "filled": r["filled"]}
            if t0 is not None:
                s = t0 + t
                o["time"] = "%02d:%02d:%06.3f" % (int(s // 3600), int(s % 3600 // 60), s % 60)
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
    print("%d objects, %d rows written; %d rows cut as standing, %d vehicles dropped as never "
          "moving; %d rows placed at or dropped for a runaway centre, %d vehicles not drawn because "
          "their centre ran away; %d rows dropped and %d objects lost entirely because the cameras "
          "disagreed"
          % (len(summary), len(out_rows), cut, dropped, clamped, lost_runaway, split, emptied))


if __name__ == "__main__":
    main()
