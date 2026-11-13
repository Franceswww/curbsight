"""
safety.py - near-misses from the trajectories: time-to-collision, post-encroachment time, heat maps.

    python safety.py --tracks ../out/tracks.csv --out ../out/safety
                     [--ttc 1.5] [--pet veh-veh=1.5,veh-ped=1.0,veh-bike=1.5]

Two surrogate safety measures, the ones the traffic-conflict literature has used for fifty years
because they can be read off trajectories alone, without waiting for a crash:

  TTC   time to collision. At one instant, if both road users kept their present speed and
        direction, how many seconds until they touch. Computed every frame for every pair within
        20 m where at least one is a vehicle that is actually moving (a pedestrian walking up to a
        parked car is not a conflict). The vehicle is its body rectangle, L x W at its heading; the
        other party is a disc (0.4 m for a person, 0.6 m for a bicycle, half its width for a
        vehicle). The disc's relative motion is a straight line in the vehicle's frame, so the
        time it enters the rectangle, grown by the disc radius, is a ray-box intersection.
        An encounter is a stretch of consecutive frames with TTC under 5 s; its minimum is kept.

  PET   post-encroachment time. Two road users used the same patch of ground; how many seconds
        passed between the first leaving it and the second arriving. The ground is cut into 1 m
        cells; a vehicle occupies every cell its body covers, a person the cell under their feet.
        For each cell, consecutive occupants are paired. Where their directions differ by more
        than 30 degrees the paths crossed; where they agree it is one following the other, which
        is a headway, reported separately and not drawn as a near-miss.

A near-miss is an encounter at or below the threshold for its pair of road users (THRESHOLDS
below): TTC 1.5 s for every pair; PET 1.5 s between two vehicles, 1.0 s between a vehicle and a
pedestrian, 1.5 s between a vehicle and a bicycle. The sources are listed in the README.

The 5 s in this file (COLLECT_S) is not a threshold. It is the window inside which an encounter is
recorded at all, so that near_misses.csv holds the whole distribution and a different threshold
can be applied without running again.

Each encounter is written to near_misses.csv with its time, place, the two parties, their classes
and speeds, the value and the threshold that applied to it. The heat maps accumulate the
near-misses on the aerial, one for TTC and one for PET.

Read the top of the list before trusting the map. A fusion that has merged two road users into one
object, or split one into two, produces a pair that was never two bodies in the same place, and the
most severe "conflicts" are where such errors surface first.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "calibration")
LAT0, LON0 = 33.94760, -83.37630
KX, KY = 111320.0 * math.cos(math.radians(LAT0)), 111320.0

VEHICLE = {"Passenger Vehicle", "Motorcycle", "Bus", "Truck"}
SIZE = {"Passenger Vehicle": (4.5, 1.9), "Truck": (5.5, 2.1), "Bus": (10.5, 2.6), "Motorcycle": (2.0, 0.8)}
RADIUS = {"Pedestrian": 0.4, "Bicycle": 0.6}                      # a disc for the party that is not a box

# Near-miss thresholds in seconds, by pair of road users. Sources: README, "Thresholds and their sources".
THRESHOLDS = {"TTC": {"veh-veh": 1.5, "veh-ped": 1.5, "veh-bike": 1.5},
              "PET": {"veh-veh": 1.5, "veh-ped": 1.0, "veh-bike": 1.5}}
COLLECT_S = 5.0                                                   # an encounter under this is recorded; not a threshold


def kind(cls):
    return "ped" if cls == "Pedestrian" else "bike" if cls == "Bicycle" else "veh"


def load(path):
    """per object: frame-sorted arrays of t, E, N, heading (rad, nan where none); plus class"""
    rows = defaultdict(list)
    cls = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g = int(r["gid"]); cls[g] = r["cls"]
            h = math.radians(float(r["hdg"])) if r.get("hdg") not in (None, "") else float("nan")
            seen = 1.0 if r.get("n_cams") in (None, "") else float(r["n_cams"])
            rows[g].append((float(r["t_sec"]), float(r["E"]), float(r["N"]), h, seen))
    objs = {}
    for g, p in rows.items():
        p.sort()
        A = np.array(p, float)
        T, E, N, H = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
        OBS = A[:, 4] >= 1                                           # a camera saw the object in this row
        # velocity by central difference over +-0.33 s of real time
        V = np.zeros((len(T), 2))
        for i in range(len(T)):
            j0 = np.searchsorted(T, T[i] - 0.33); j1 = min(np.searchsorted(T, T[i] + 0.33, side="right"), len(T)) - 1
            dt = T[j1] - T[j0]
            if dt > 1e-6:
                V[i] = ((E[j1] - E[j0]) / dt, (N[j1] - N[j0]) / dt)
        # a vehicle's heading: the file's where it has one, else its direction of travel
        speed = np.hypot(*V.T)
        Hv = np.where(np.isnan(H), np.where(speed > 0.5, np.arctan2(V[:, 1], V[:, 0]), np.nan), H)
        # hold the last known heading through stands and gaps
        for i in range(1, len(Hv)):
            if np.isnan(Hv[i]): Hv[i] = Hv[i - 1]
        for i in range(len(Hv) - 2, -1, -1):
            if np.isnan(Hv[i]): Hv[i] = Hv[i + 1]
        objs[g] = dict(T=T, E=E, N=N, V=V, speed=speed, H=Hv, cls=cls[g], obs=OBS)
    return objs


def ray_box_entry(x, y, vx, vy, hx, hy):
    """time at which a point at (x, y) moving (vx, vy) enters the box |x|<=hx, |y|<=hy; inf if never"""
    if abs(x) <= hx and abs(y) <= hy:
        return 0.0
    t0, t1 = -math.inf, math.inf
    for p, v, h in ((x, vx, hx), (y, vy, hy)):
        if abs(v) < 1e-9:
            if abs(p) > h:
                return math.inf
            continue
        a, b = (-h - p) / v, (h - p) / v
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    if t0 > t1 or t1 < 0:
        return math.inf
    return max(t0, 0.0)


def ttc_pair(A, i, B, j, margin=0.0):
    """TTC between object A at its row i (the box) and object B at its row j (the disc), seconds.

    margin shrinks the box the disc has to enter. With margin 0 the geometry is exact: the body
    rectangle grown by the disc radius. The positions carry about a metre of error, and at that
    error a pedestrian waiting at the kerb while a bus passes a metre away is, as measured, under
    the bus. So the near-miss decision uses margin = the position error: the person has to be on
    course for the inner part of the vehicle, not its measured edge.
    """
    L, W = SIZE.get(A["cls"], (4.5, 1.9))
    r = RADIUS.get(B["cls"], SIZE.get(B["cls"], (4.5, 1.9))[1] / 2)
    th = A["H"][i]
    if np.isnan(th):
        return math.inf
    hx, hy = L / 2 + r - margin, W / 2 + r - margin
    if hx <= 0.2 or hy <= 0.2:
        return math.inf
    c, s = math.cos(-th), math.sin(-th)
    dx, dy = B["E"][j] - A["E"][i], B["N"][j] - A["N"][i]
    dvx, dvy = B["V"][j][0] - A["V"][i][0], B["V"][j][1] - A["V"][i][1]
    x, y = c * dx - s * dy, s * dx + c * dy
    vx, vy = c * dvx - s * dvy, s * dvx + c * dvy
    return ray_box_entry(x, y, vx, vy, hx, hy)


def same_object(A, B, tol_s=0.1, close_m=2.0):
    """Two tracks that are one road user seen twice: a rider also detected as a pedestrian, a bus
    and its driver through the windscreen, a car fused into two objects. They share at least 3 s
    and sit within close_m of each other for most of it. Such a pair is no encounter at all."""
    tb = B["T"]
    d = []
    for i, t in enumerate(A["T"]):
        k = np.searchsorted(tb, t)
        for kk in (k - 1, k):
            if 0 <= kk < len(tb) and abs(tb[kk] - t) <= tol_s:
                d.append(math.hypot(B["E"][kk] - A["E"][i], B["N"][kk] - A["N"][i])); break
    # one second of shared time is enough: the clearest case was a "pedestrian" that lived for 1.5 s,
    # entirely inside a passing bus, because the detector had found the people printed on its side
    if len(d) < 15:
        return False
    d = np.array(d)
    return (d < close_m).mean() > 0.6


def on_road(field, e, n, min_vehicles=3):
    """does the lane field (build_lane_field.py) know this ground as somewhere vehicles go?"""
    c = field["cell"]
    f = field["cells"].get((math.floor(e / c), math.floor(n / c)))
    return f is not None and f[0] >= min_vehicles


def ttc_events(objs, fps, field=None, near_m=20.0, collect_s=COLLECT_S, min_speed=1.0, approach_s=1.5, margin=0.8):
    """encounters: (ttc_min, t, E, N, gA, gB, flag) for every pair whose TTC dipped below collect_s.

    ttc_min is the smallest POSITIVE value in the encounter. The flag says why an encounter is not
    a near-miss even when that value is small:

      duplicate       the two tracks are one road user seen twice (same_object)
      overlap         TTC was zero throughout: the measured bodies intersect, which in traffic that
                      did not crash is position error, not the closest call of the hour
      unobserved      at the minimum one party's row is filled (n_cams 0), not detected: a near-miss
                      has to be seen, not interpolated
      contact         TTC reached zero at some point: as measured the bodies met. Nobody was hit,
                      so a position was wrong by at least the real gap, and the pair says nothing
                      about how close the call was. A pedestrian waiting at the kerb while a bus
                      passes, measured a metre into the lane, is the usual case
      no_approach     TTC was never seen above approach_s before it reached its minimum. A near-miss
                      is an approach we watched happen; a pair that is already close in its first
                      frame is two bodies a metre apart with a metre of position error, which is
                      what a pedestrian at the kerb beside a passing lorry looks like
      off_road_party  the person stands on ground no vehicle has ever used (lane field); a vehicle
                      that stays on the road cannot reach them, so the course is lateral error
      within_position_error
                      the other party is on course for the measured edge of the vehicle but not for
                      its inner part once the box is shrunk by the position error (margin, 0.8 m):
                      a pedestrian at the kerb as a bus passes a metre away

    The 71-minute recording had 68 zeros among its first 94 "near-misses", all three kinds.
    """
    by_frame = defaultdict(list)
    for g, o in objs.items():
        for i, t in enumerate(o["T"]):
            by_frame[int(round(t * fps))].append((g, i))
    enc = {}                                                         # (gA, gB) -> state dict
    done = []
    dup_cache = {}

    def flush(key, e):
        if key not in dup_cache:
            dup_cache[key] = same_object(objs[key[0]], objs[key[1]])
        B = objs[key[1]]
        if dup_cache[key]:
            flag = "duplicate"
        elif e["min"] == math.inf:
            flag = "overlap"
        elif e.get("touched"):
            # TTC reached zero at some frame: as measured, the two bodies met. Nobody was hit, so at
            # that moment one position was wrong by at least the gap that really remained, and how
            # close the call truly was cannot be read from this pair. A near-miss is a pair that
            # came close and, as measured, did not touch.
            flag = "contact"
        elif not objs[key[0]]["obs"][e["iA"]] or not B["obs"][e["iB"]]:
            # at the closest moment one party's position is a row the fusion filled through a gap,
            # not a detection. The two-minute sample had five "pedestrians" born from the faces
            # printed on a bus advertisement, carried on for three seconds after the detections
            # stopped; every one of their near-misses sat on filled rows.
            flag = "unobserved"
        elif e["max_before_min"] < approach_s or e.get("t_above_at_min") is None \
                or (e["t"] - e["t_above_at_min"]) < 0.5 * (approach_s - e["min"]):
            # Either TTC was never beyond the threshold before its minimum, or it got from the
            # threshold to the minimum in less than half the time a steady approach needs (TTC
            # falls about one second per second). A drop from 3 s to 0.1 s inside two frames is a
            # position jump of one of the parties, not a vehicle closing on a person.
            flag = "no_approach"
        elif B["cls"] not in VEHICLE and field is not None and not on_road(field, B["E"][e["iB"]], B["N"][e["iB"]]):
            flag = "off_road_party"
        elif e["min_inner"] > approach_s:
            flag = "within_position_error"
        else:
            flag = ""
        done.append((e["min"] if e["min"] < math.inf else 0.0, e["t"], e["E"], e["N"], key[0], key[1], flag))

    for fr in sorted(by_frame):
        L = by_frame[fr]
        if len(L) < 2:
            continue
        P = np.array([[objs[g]["E"][i], objs[g]["N"][i]] for g, i in L])
        D = np.hypot(*(P[:, None, :] - P[None, :, :]).transpose(2, 0, 1))
        for a in range(len(L)):
            gA, iA = L[a]; A = objs[gA]
            if A["cls"] not in VEHICLE or A["speed"][iA] < min_speed:
                continue                                             # only a moving vehicle can run into something
            for b in np.where(D[a] < near_m)[0]:
                if b == a:
                    continue
                gB, iB = L[b]; B = objs[gB]
                if gA == gB:
                    continue
                t = ttc_pair(A, iA, B, iB)
                if t > collect_s:
                    continue
                t_inner = ttc_pair(A, iA, B, iB, margin)            # the same course against the shrunk box
                key = (gA, gB)
                tt = A["T"][iA]
                e = enc.get(key)
                if e is None or fr - e["last"] > 6:                  # a new encounter after a gap
                    if e is not None:
                        flush(key, e)
                    e = enc[key] = dict(min=math.inf, t=tt, E=B["E"][iB], N=B["N"][iB], iB=iB, iA=iA, last=fr,
                                        run_max=0.0, max_before_min=0.0, min_inner=math.inf)
                if 0 < t_inner < e["min_inner"]:
                    e["min_inner"] = t_inner
                if 0 < t < e["min"]:
                    e["max_before_min"] = e["run_max"]               # the largest TTC seen before this new low
                    e["t_above_at_min"] = e.get("t_above")           # when TTC was last beyond the threshold
                    e.update(min=t, t=tt, E=B["E"][iB], N=B["N"][iB], iB=iB, iA=iA)
                elif t == 0 and e["min"] == math.inf:
                    e.update(t=tt, E=B["E"][iB], N=B["N"][iB], iB=iB, iA=iA)
                if t == 0:
                    e["touched"] = True
                if t >= approach_s:
                    e["t_above"] = tt
                e["run_max"] = max(e["run_max"], t)
                e["last"] = fr
    for key, e in enc.items():
        flush(key, e)
    return done


def footprint_cells(o, i, cell):
    """the 1 m cells a vehicle's body covers at row i; a person's single cell"""
    if o["cls"] not in VEHICLE or np.isnan(o["H"][i]):
        return {(math.floor(o["E"][i] / cell), math.floor(o["N"][i] / cell))}
    L, W = SIZE.get(o["cls"], (4.5, 1.9))
    th = o["H"][i]; ax = np.array([math.cos(th), math.sin(th)]); lat = np.array([-ax[1], ax[0]])
    out = set()
    for u in np.arange(-L / 2, L / 2 + 1e-9, 0.5):
        for w in np.arange(-W / 2, W / 2 + 1e-9, 0.5):
            p = np.array([o["E"][i], o["N"][i]]) + u * ax + w * lat
            out.add((math.floor(p[0] / cell), math.floor(p[1] / cell)))
    return out


def pet_events(objs, cell=1.0, collect_s=COLLECT_S, gap_s=0.5, min_speed=1.0):
    """(pet, t_second_arrives, E, N, gA_first, gB_second, crossing, flag) for consecutive occupants
    of a cell. flag is "" for a conflict worth counting, otherwise why it is not one."""
    occ = defaultdict(list)                                          # cell -> [(t, g)]
    for g, o in objs.items():
        for i in range(len(o["T"])):
            for c in footprint_cells(o, i, cell):
                occ[c].append((o["T"][i], g))
    out = []
    for c, L in occ.items():
        L.sort()
        # intervals per object in this cell
        iv = []                                                      # (t_in, t_out, g)
        cur = {}
        for t, g in L:
            if g in cur and t - cur[g][1] <= gap_s:
                cur[g][1] = t
            else:
                if g in cur:
                    iv.append((cur[g][0], cur[g][1], g))
                cur[g] = [t, t]
        for g, (a, b) in cur.items():
            iv.append((a, b, g))
        iv.sort()
        for k in range(1, len(iv)):
            a_in, a_out, gA = iv[k - 1]; b_in, b_out, gB = iv[k]
            if gA == gB:
                continue
            pet = b_in - a_out
            if pet < 0 or pet > collect_s:
                continue                                             # simultaneous occupancy is not an encroachment
            A, B = objs[gA], objs[gB]
            kA, kB = kind(A["cls"]), kind(B["cls"])
            if kA != "veh" and kB != "veh":
                continue                                             # two people sharing a kerb are not a conflict
            ia = min(int(np.searchsorted(A["T"], a_out)), len(A["T"]) - 1)
            ib = min(int(np.searchsorted(B["T"], b_in)), len(B["T"]) - 1)
            va, vb = A["V"][ia], B["V"][ib]
            flag = ""
            v_fast = max(np.hypot(*va), np.hypot(*vb), 1.0)
            if pet == 0:
                # the second arrived in the frame the first left: the two measured bodies touched.
                # With a metre of position error and 10 m bus footprints this is how two buses
                # passing a metre apart look, and how a passenger stepping out of a bus looks.
                flag = "overlap"
            elif not A["obs"][ia] or not B["obs"][ib]:
                flag = "unobserved"                                  # a filled row, not a detection, at the moment that matters
            elif pet < 1.0 / v_fast:
                # positions carry about a metre of error; at the faster party's speed that metre is
                # 1/v seconds, and a PET shorter than that cannot be told from touching
                flag = "within_position_error"
            elif (kA == "veh" and np.hypot(*va) < min_speed) or (kB == "veh" and np.hypot(*vb) < min_speed):
                flag = "vehicle_standing"                            # a parked car and a passer-by are not a conflict
            elif kA != "veh" and A["T"][-1] < a_out + 2.0:
                flag = "boarding"                                    # the person vanished into the vehicle's footprint
            elif kB != "veh" and B["T"][0] > b_in - 2.0:
                flag = "alighting"                                   # the person appeared out of the vehicle's footprint
            # crossing or following: directions of travel at the moments that matter
            if np.hypot(*va) < 0.3 or np.hypot(*vb) < 0.3 or kA != "veh" or kB != "veh":
                crossing = True                                      # a standing party or a person: not a headway
            else:
                ang = abs(math.degrees(math.atan2(va[0] * vb[1] - va[1] * vb[0], va[0] * vb[0] + va[1] * vb[1])))
                crossing = ang > 30.0
            out.append((pet, b_in, (c[0] + 0.5) * cell, (c[1] + 0.5) * cell, gA, gB, crossing, flag))
    # one event per pair per 5 s: the smallest PET among the unflagged, else the smallest flagged
    best = {}
    for ev in sorted(out, key=lambda e: e[1]):
        key = (ev[4], ev[5])
        prev = best.get(key)
        if prev and ev[1] - prev[-1][1] <= collect_s:
            cur = prev[-1]
            if (ev[7] == "" and cur[7] != "") or (ev[7] == cur[7] and ev[0] < cur[0]):
                prev[-1] = ev
        else:
            best.setdefault(key, []).append(ev)
    events = [ev for lst in best.values() for ev in lst]
    dup = {}
    out2 = []
    for ev in events:
        key = (ev[4], ev[5])
        if key not in dup:
            dup[key] = same_object(objs[key[0]], objs[key[1]])
        out2.append(ev[:7] + ("duplicate" if dup[key] else ev[7],))
    return out2


def heatmap(points, base, E0, N1, res, label, out_path, cell_m=1.0):
    """near-miss positions accumulated on the dimmed aerial, square-root scaled like density.py"""
    H, W = base.shape[:2]
    gw, gh = int(W * res / cell_m) + 1, int(H * res / cell_m) + 1
    grid = np.zeros((gh, gw), np.float32)
    for e, n in points:
        gx, gy = int((e - E0) / cell_m), int((N1 - n) / cell_m)
        if 0 <= gx < gw and 0 <= gy < gh:
            grid[gy, gx] += 1
    big = cv2.GaussianBlur(cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 1.5 / res)
    v = np.sqrt(big / big.max()) if big.max() > 0 else big
    heat = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = np.clip(v * 2.2, 0, 1)[..., None]
    img = (base * 0.55 * (1 - alpha) + base * 0.55 * alpha * 0.25 + heat * alpha * 0.75).astype(np.uint8)
    for e, n in points:                                              # the events themselves, so a single one still shows
        q = (int(round((e - E0) / res)), int(round((N1 - n) / res)))
        cv2.circle(img, q, 9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, label, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, label, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


FONT = cv2.FONT_HERSHEY_SIMPLEX
PAIR_COL = {"veh-ped": (0, 215, 255), "ped-veh": (0, 215, 255), "veh-veh": (255, 170, 70), "veh-bike": (90, 220, 90), "bike-veh": (90, 220, 90)}
PAIR_NAME = {"veh-ped": "vehicle - pedestrian", "veh-veh": "vehicle - vehicle", "veh-bike": "vehicle - bicycle"}


def pair_key(p):
    return {"ped-veh": "veh-ped", "bike-veh": "veh-bike"}.get(p, p)


def parse_thresholds(arg, default):
    """'1.5' applies to every pair; 'veh-veh=1.5,veh-ped=1.0' changes only the pairs it names"""
    thr = dict(default)
    if arg is None:
        return thr
    try:
        v = float(arg)
        return {k: v for k in thr}
    except ValueError:
        pass
    for kv in arg.split(","):
        k, v = kv.split("=")
        k = pair_key(k.strip())
        if k not in thr:
            raise SystemExit("unknown pair '%s': use veh-veh, veh-ped or veh-bike" % k)
        thr[k] = float(v)
    return thr


def event_map(events, base, E0, N1, res, thr, label, out_path):
    """every near-miss as one marker on the dimmed aerial: colour = who met whom, size = how close the call was
    (a marker at the threshold is small, one at 0 s is large); the seconds are written beside each marker"""
    img = (base * 0.55).astype(np.uint8)
    H, W = img.shape[:2]
    for d in sorted(events, key=lambda d: -d["seconds"]):                 # the closest calls drawn last, on top
        q = (int(round((d["E"] - E0) / res)), int(round((N1 - d["N"]) / res)))
        if not (0 <= q[0] < W and 0 <= q[1] < H):
            continue
        r = int(round(9 + 26 * (thr - min(d["seconds"], thr)) / thr))
        col = PAIR_COL.get(d["pair"], (255, 255, 255))
        cv2.circle(img, q, r, col, -1, cv2.LINE_AA); cv2.circle(img, q, r, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "%.1f" % d["seconds"], (q[0] + r + 3, q[1] + 6), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, "%.1f" % d["seconds"], (q[0] + r + 3, q[1] + 6), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, label, (24, 46), FONT, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, label, (24, 46), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    # legend, bottom left
    counts = defaultdict(int)
    for d in events: counts[pair_key(d["pair"])] += 1
    y = H - 30 - 34 * (len(PAIR_NAME) + 1)
    cv2.rectangle(img, (18, y - 30), (560, H - 18), (20, 20, 20), -1)
    for k in ("veh-ped", "veh-veh", "veh-bike"):
        cv2.circle(img, (46, y), 12, PAIR_COL[k], -1, cv2.LINE_AA); cv2.circle(img, (46, y), 12, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "%s: %d" % (PAIR_NAME[k], counts.get(k, 0)), (70, y + 8), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA); y += 34
    cv2.circle(img, (40, y), 9, (255, 255, 255), 1, cv2.LINE_AA); cv2.circle(img, (72, y), 35 // 2, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "size: %.1f s  ->  0 s" % thr, (100, y + 8), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def bar_chart(out_path, title, xticks, series, vline=None, width=1600, height=640, note=""):
    """grouped bars with OpenCV. xticks: one label per bin ("" for none); series: [(name, colour, values)];
    vline: a bin position (float, in bins) for a vertical threshold line"""
    img = np.full((height, width, 3), 255, np.uint8)
    L, R, T, B = 90, 40, 80, 110
    W, H = width - L - R, height - T - B
    n = max(1, len(xticks)); ymax = max(1, max(max(v) for _, _, v in series))
    step = max(1, int(math.ceil(ymax / 5.0))); ymax = int(math.ceil(ymax / float(step))) * step
    def Y(v): return T + H - int(round(v / float(ymax) * H))
    for k in range(0, ymax + 1, step):
        cv2.line(img, (L, Y(k)), (L + W, Y(k)), (228, 228, 228), 1)
        cv2.putText(img, str(k), (L - 16 - 13 * len(str(k)), Y(k) + 7), FONT, 0.6, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(img, (L, T), (L, T + H), (0, 0, 0), 2); cv2.line(img, (L, T + H), (L + W, T + H), (0, 0, 0), 2)
    gw = W / float(n); bw = gw / (len(series) + 1)
    for si, (name, col, vals) in enumerate(series):
        for i, v in enumerate(vals):
            x0 = int(round(L + i * gw + bw * (si + 0.5))); x1 = int(round(x0 + bw))
            if v > 0: cv2.rectangle(img, (x0, Y(v)), (x1, T + H), col, -1)
    for i, lab in enumerate(xticks):
        if lab: cv2.putText(img, lab, (int(L + i * gw) - 4, T + H + 28), FONT, 0.6, (70, 70, 70), 1, cv2.LINE_AA)
    if vline is not None:
        x = int(round(L + vline * gw)); cv2.line(img, (x, T), (x, T + H), (0, 0, 230), 2)
        cv2.putText(img, "threshold", (x + 8, T + 22), FONT, 0.6, (0, 0, 230), 1, cv2.LINE_AA)
    cv2.putText(img, title, (L, 48), FONT, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    if note: cv2.putText(img, note, (L, height - 40), FONT, 0.6, (70, 70, 70), 1, cv2.LINE_AA)
    for si, (name, col, _) in enumerate(series):
        x = L + W - 620; y = T + 12 + 30 * si
        cv2.rectangle(img, (x, y), (x + 24, y + 18), col, -1); cv2.putText(img, name, (x + 32, y + 16), FONT, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def distributions(events, thr_ttc, thr_pet, out_path):
    """the smallest TTC / PET of every encounter under 5 s, counted ones against the ones set aside"""
    edges = np.arange(0, 5.001, 0.25)
    panels = []
    for m, thr in (("TTC", thr_ttc), ("PET", thr_pet)):
        ev = [d for d in events if d["measure"] == m and (m == "TTC" or d["crossing"] == 1)]
        kept = np.zeros(len(edges) - 1, int); aside = np.zeros(len(edges) - 1, int)
        for d in ev:
            b = min(int(d["seconds"] / 0.25), len(edges) - 2)
            if d["flag"] == "": kept[b] += 1
            else: aside[b] += 1
        tmp = out_path + ".%s.png" % m.lower()
        bar_chart(tmp, "%s: smallest value of each encounter under 5 s  (%d encounters)" % (m, len(ev)),
                  ["%g" % e if abs(e - round(e)) < 1e-9 else "" for e in edges[:-1]],
                  [("counted", (200, 120, 40), list(kept)),
                   ("set aside (position error, duplicate, boarding, standing)", (190, 190, 190), list(aside))],
                  vline=thr / 0.25, note="seconds")
        panels.append(cv2.imread(tmp)); os.remove(tmp)
    cv2.imwrite(out_path, np.vstack(panels))


def over_time(events, clock, out_path, bin_s=300.0):
    """near-misses per bin of bin_s seconds, TTC and PET side by side; only worth drawing past ten minutes"""
    nm = [d for d in events if d["near_miss"]]
    t_all = [d["t_sec"] for d in events]
    if not t_all or max(t_all) - min(t_all) < 600:
        return False
    t0 = math.floor(min(t_all) / bin_s) * bin_s; nb = int(math.ceil((max(t_all) - t0) / bin_s)) + 1
    ttc = np.zeros(nb, int); pet = np.zeros(nb, int)
    for d in nm:
        b = min(int((d["t_sec"] - t0) / bin_s), nb - 1)
        if d["measure"] == "TTC": ttc[b] += 1
        else: pet[b] += 1
    labels = [clock(t0 + i * bin_s)[:5] if clock(t0) else "%d" % int((t0 + i * bin_s) // 60) for i in range(nb)]
    bar_chart(out_path, "near-misses per %d minutes" % int(bin_s // 60), labels,
              [("TTC", (60, 60, 220), list(ttc)), ("PET", (200, 120, 40), list(pet))], note="start of each bin")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="tracks.csv from export_tracks.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--ttc", default=None, help="s; a TTC at or below this is a near-miss. One number for every pair, or "
                                                "veh-veh=..,veh-ped=..,veh-bike=..  Default: 1.5 for every pair")
    ap.add_argument("--pet", default=None, help="s; a PET at or below this is a near-miss, same syntax. "
                                                "Default: veh-veh=1.5,veh-ped=1.0,veh-bike=1.5")
    ap.add_argument("--min-speed", type=float, default=1.0, help="m/s; a vehicle slower than this cannot run into anyone")
    ap.add_argument("--margin", type=float, default=0.8, help="m; position error the TTC geometry must beat before a near-miss counts")
    ap.add_argument("--start", default=None, help="wall-clock time of t=0, HH:MM:SS, for the event list")
    ap.add_argument("--lane-field", default=os.path.join(CAL, "lane_field_sanford_cedar.json"),
                    help="build_lane_field.py output; a person off it cannot be on a vehicle's path")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = None
    if a.start:
        hh, mm, ss = (float(v) for v in a.start.split(":")); t0 = hh * 3600 + mm * 60 + ss
    clock = lambda t: ("%02d:%02d:%05.2f" % (int((t0 + t) // 3600), int((t0 + t) % 3600 // 60), (t0 + t) % 60)) if t0 is not None else ""
    THR = {"TTC": parse_thresholds(a.ttc, THRESHOLDS["TTC"]), "PET": parse_thresholds(a.pet, THRESHOLDS["PET"])}
    thr_of = lambda m, pair: THR[m].get(pair_key(pair), max(THR[m].values()))
    top = {m: max(THR[m].values()) for m in THR}

    objs = load(a.tracks)
    print("%d objects loaded" % len(objs))
    field = None
    if a.lane_field and os.path.exists(a.lane_field):
        j = json.load(open(a.lane_field))
        field = dict(cell=float(j["cell_m"]), cells={tuple(int(x) for x in k.split(",")): (v["vehicles"],) for k, v in j["cells"].items()})
    ttc = ttc_events(objs, a.fps, field=field, min_speed=a.min_speed, approach_s=top["TTC"], margin=a.margin)
    pet = pet_events(objs, min_speed=a.min_speed)
    print("encounters: %d with TTC under 5 s, %d with PET under 5 s" % (len(ttc), len(pet)))

    events = []
    for t_min, t, e, n, gA, gB, flag in ttc:
        A, B = objs[gA], objs[gB]
        iA = min(int(np.searchsorted(A["T"], t)), len(A["T"]) - 1); iB = min(int(np.searchsorted(B["T"], t)), len(B["T"]) - 1)
        pr = "%s-%s" % (kind(A["cls"]), kind(B["cls"])); th = thr_of("TTC", pr)
        events.append(dict(measure="TTC", seconds=round(t_min, 2), threshold=th, near_miss=int(flag == "" and 0 < t_min <= th), flag=flag,
                           t_sec=round(t, 2), time=clock(t),
                           E=round(e, 2), N=round(n, 2), lat=round(LAT0 + n / KY, 7), lon=round(LON0 + e / KX, 7),
                           pair=pr, gid_a=gA, cls_a=A["cls"], speed_a=round(float(A["speed"][iA]), 1),
                           gid_b=gB, cls_b=B["cls"], speed_b=round(float(B["speed"][iB]), 1), crossing=""))
    for p, t, e, n, gA, gB, crossing, flag in pet:
        A, B = objs[gA], objs[gB]
        iA = min(int(np.searchsorted(A["T"], t)), len(A["T"]) - 1); iB = min(int(np.searchsorted(B["T"], t)), len(B["T"]) - 1)
        pr = "%s-%s" % (kind(A["cls"]), kind(B["cls"])); th = thr_of("PET", pr)
        events.append(dict(measure="PET", seconds=round(p, 2), threshold=th, near_miss=int(flag == "" and crossing and p <= th), flag=flag,
                           t_sec=round(t, 2), time=clock(t),
                           E=round(e, 2), N=round(n, 2), lat=round(LAT0 + n / KY, 7), lon=round(LON0 + e / KX, 7),
                           pair=pr, gid_a=gA, cls_a=A["cls"], speed_a=round(float(A["speed"][iA]), 1),
                           gid_b=gB, cls_b=B["cls"], speed_b=round(float(B["speed"][iB]), 1), crossing=int(crossing)))
    events.sort(key=lambda d: (d["measure"], d["near_miss"] == 0, d["seconds"]))
    with open(os.path.join(a.out, "near_misses.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0])); w.writeheader(); w.writerows(events)

    ext = json.load(open(os.path.join(CAL, "aerial_wide.json"))); E0, N1, RES = ext["E0"], ext["N1"], ext["res"]
    base = cv2.imread(os.path.join(CAL, "aerial_wide.png"))
    lines = []
    for m in ("TTC", "PET"):
        nm = [d for d in events if d["measure"] == m and d["near_miss"]]
        by_pair = defaultdict(int)
        for d in nm: by_pair[pair_key(d["pair"])] += 1
        parts = ", ".join("%s <= %.1f s: %d" % (PAIR_NAME[k], THR[m][k], by_pair.get(k, 0)) for k in ("veh-veh", "veh-ped", "veh-bike"))
        lines.append("%s: %d near-misses (%s)" % (m, len(nm), parts))
        heatmap([(d["E"], d["N"]) for d in nm], base, E0, N1, RES, "%s near-misses: %s" % (m, parts),
                os.path.join(a.out, "nearmiss_%s.png" % m.lower()))
    nm_all = [d for d in events if d["near_miss"]]
    event_map(nm_all, base, E0, N1, RES, max(top.values()),
              "near-misses, TTC or PET at or under the threshold of the pair: %d" % len(nm_all), os.path.join(a.out, "nearmiss_events.png"))
    distributions(events, top["TTC"], top["PET"], os.path.join(a.out, "nearmiss_distribution.png"))
    if over_time(events, clock, os.path.join(a.out, "nearmiss_over_time.png")):
        lines.append("near-misses over time: nearmiss_over_time.png")
    follow = sum(1 for d in events if d["measure"] == "PET" and d["crossing"] == 0 and d["flag"] == "" and d["seconds"] <= d["threshold"])
    lines.append("PET at or under the threshold between vehicles following in the same direction (headways, not drawn): %d" % follow)
    for m in ("TTC", "PET"):
        flags = defaultdict(int)
        for d in events:
            if d["measure"] == m and d["flag"] and d["seconds"] <= d["threshold"]:
                flags[d["flag"]] += 1
        lines.append("%s under threshold but not counted: %s" % (m, ", ".join("%s %d" % kv for kv in sorted(flags.items())) or "none"))
    txt = "\n".join(lines); print(txt)
    open(os.path.join(a.out, "safety_summary.txt"), "w").write(txt + "\n")
    print("most severe near-misses:")
    for d in [d for d in events if d["measure"] == "TTC" and d["near_miss"]][:6] + [d for d in events if d["measure"] == "PET" and d["near_miss"]][:6]:
        print("   %s %5.2f s  %s  %-8s  %s %d (%.1f m/s) vs %s %d (%.1f m/s)  at E %.1f N %.1f"
              % (d["measure"], d["seconds"], d["time"], d["pair"], d["cls_a"], d["gid_a"], d["speed_a"], d["cls_b"], d["gid_b"], d["speed_b"], d["E"], d["N"]))


if __name__ == "__main__":
    main()
