"""
safety.py - near-misses from the trajectories: time-to-collision, post-encroachment time, heat maps.

    python safety.py --tracks ../out/tracks.csv --out ../out/safety

Two surrogate safety measures, the ones the traffic-conflict literature has used for fifty years
because they can be read off trajectories alone, without waiting for a crash:

  TTC   time to collision. At one instant, if both road users went on along the paths they
        actually took, each at its present speed, how many seconds until they touch. This is the
        projection FHWA's SSAM uses. A straight line at the present velocity, the oldest form of
        the measure, points a turning vehicle at whatever stands outside its bend, and is not
        used. Computed every frame for every pair within 80 m where at least one is a vehicle that
        is actually moving (a pedestrian walking up to a parked car is not a conflict). The vehicle
        is its body rectangle, L x W at the heading it has at that point of its path; the other
        party is a disc (0.4 m for a person, 0.6 m for a bicycle, half its width for a vehicle).
        Both are stepped along their paths in 0.05 s steps; past the end of a recorded path a road
        user goes straight on. An encounter is a stretch of consecutive frames with TTC under 8 s;
        its minimum is kept.

  PET   post-encroachment time. Two road users used the same patch of ground; how many seconds
        passed between the first leaving it and the second arriving. The ground is cut into 1 m
        cells; a vehicle occupies every cell its body covers, a person the cell under their feet.
        For each cell, consecutive occupants are paired, and the pair is typed the way published
        tools type an encounter (encounter_type.py): following, oncoming, merging or crossing.
        PET is counted for merging and crossing only, as SUMO's safety device does. A follower in
        the same lane is a matter for TTC, and two vehicles that pass in opposite lanes only touch
        the same cell at the centre line.

Every counted encounter is given a severity level by its pair of road users (LEVELS below): the
upper edge of the band its smallest value falls in. Vehicle and vehicle: TTC 1.5 / 2.5 / 4.0 / 5.0 s,
PET 2 / 3 / 5 s. Vehicle and pedestrian: TTC 1.5 / 3.0 s, PET 1 / 3 / 5 s. Vehicle and bicycle: TTC
1.0 / 2.0 / 3.0 s, PET 1.5 / 3 / 5 s. The sources are listed in the README. The tests that tell an
approach from position error are made against the upper edge of that level, not against one fixed
value.

Each encounter is written to near_misses.csv with its time, place, the two parties, their classes
and speeds, the value and its severity level. The maps accumulate encounters on the aerial: every
counted one (conflicts_ttc.png, encounters_pet.png) and those of the most severe level
(nearmiss_ttc.png, nearmiss_pet.png).

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

import encounter_type

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "calibration")
LAT0, LON0 = 33.94760, -83.37630
KX, KY = 111320.0 * math.cos(math.radians(LAT0)), 111320.0

VEHICLE = {"Passenger Vehicle", "Motorcycle", "Bus", "Truck"}
SIZE = {"Passenger Vehicle": (4.5, 1.9), "Truck": (5.5, 2.1), "Bus": (10.5, 2.6), "Motorcycle": (2.0, 0.8)}
RADIUS = {"Pedestrian": 0.4, "Bicycle": 0.6}                      # a disc for the party that is not a box

# Severity levels in seconds, by pair of road users: the upper edges of the bands, most severe first.
# Sources: README, "Severity levels and their sources".
LEVELS = {"TTC": {"veh-veh": (1.5, 2.5, 4.0, 5.0), "veh-ped": (1.5, 3.0), "veh-bike": (1.0, 2.0, 3.0)},
          "PET": {"veh-veh": (2.0, 3.0, 5.0), "veh-ped": (1.0, 3.0, 5.0), "veh-bike": (1.5, 3.0, 5.0)}}
COLLECT_S = 5.0                                                   # an encounter under this is written to the file
WATCH_S = 8.0                                                     # TTC is followed from this far, to see an approach from above 5 s


COUNTED = ("crossing", "merging")                                # encounter types with a post-encroachment time


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


TAU = np.arange(0.0, 8.0 + 1e-9, 0.05)                           # s; the instants a path is followed to


def _path(o):
    """distance along the recorded path, unwrapped heading and the direction past the end, once per object"""
    if "S" in o:
        return
    d = np.hypot(np.diff(o["E"]), np.diff(o["N"]))
    o["S"] = np.concatenate([[0.0], np.cumsum(d)]) + 1e-7 * np.arange(len(o["T"]))      # strictly increasing
    o["Hu"] = np.unwrap(o["H"]) if not np.isnan(o["H"]).any() else None
    k = max(len(o["T"]) - 8, 0)
    de, dn = o["E"][-1] - o["E"][k], o["N"][-1] - o["N"][k]; L = math.hypot(de, dn)
    o["end_u"] = (de / L, dn / L) if L > 0.3 else (0.0, 0.0)


def _along(o, i):
    """where object o is, from its row i, at the instants TAU: s + v * tau along its own path, straight on past the end"""
    s = o["S"][i] + o["speed"][i] * TAU
    over = np.clip(s - o["S"][-1], 0, None)
    return np.interp(s, o["S"], o["E"]) + over * o["end_u"][0], np.interp(s, o["S"], o["N"]) + over * o["end_u"][1], s


def ttc_pair(A, i, B, j, margin=0.0):
    """TTC between object A at its row i (the box) and object B at its row j (the disc), seconds, with
    both carried on along their own recorded paths at their present speeds.

    margin shrinks the box the disc has to enter. With margin 0 the geometry is exact: the body
    rectangle grown by the disc radius. The positions carry about a metre of error, and at that
    error a pedestrian waiting at the kerb while a bus passes a metre away is, as measured, under
    the bus. So the decision uses margin = the position error: the person has to be on course for
    the inner part of the vehicle, not its measured edge.
    """
    _path(A); _path(B)
    if A["Hu"] is None:
        return math.inf
    L, W = SIZE.get(A["cls"], (4.5, 1.9))
    r = RADIUS.get(B["cls"], SIZE.get(B["cls"], (4.5, 1.9))[1] / 2)
    hx, hy = L / 2 + r - margin, W / 2 + r - margin
    if hx <= 0.2 or hy <= 0.2:
        return math.inf
    reach = (A["speed"][i] + B["speed"][j]) * TAU[-1] + hx + hy
    if math.hypot(B["E"][j] - A["E"][i], B["N"][j] - A["N"][i]) > reach:
        return math.inf                                              # too far apart to meet inside the horizon
    ea, na, sa = _along(A, i); eb, nb, _ = _along(B, j)
    ha = np.interp(sa, A["S"], A["Hu"])                              # the heading the vehicle has at that point of its path
    dx, dy = eb - ea, nb - na; c, s_ = np.cos(ha), np.sin(ha)
    x, y = dx * c + dy * s_, -dx * s_ + dy * c
    hit = (np.abs(x) <= hx) & (np.abs(y) <= hy)
    return float(TAU[int(np.argmax(hit))]) if hit.any() else math.inf


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


def ttc_events(objs, fps, edges_of, field=None, near_m=80.0, watch_s=WATCH_S, min_speed=1.0, margin=0.8):
    """encounters: one dict (ttc, t, E, N, gA, gB, iA, iB, flag, band) for every pair whose TTC dipped below
    watch_s. edges_of(other party) gives the upper edges of the severity levels for that pair; band is
    the edge of the level the minimum falls in (None above the last edge). The tests that depend on a
    threshold (approach seen from above it, steady descent, inner box) are made against that edge.

    The minimum is the smallest POSITIVE value in the encounter. The flag says why an encounter is
    not counted even when that value is small:

      duplicate       the two tracks are one road user seen twice (same_object)
      overlap         TTC was zero throughout: the measured bodies intersect, which in traffic that
                      did not crash is position error, not the closest call of the hour
      unobserved      at the minimum one party's row is filled (n_cams 0), not detected: a near-miss
                      has to be seen, not interpolated
      contact         TTC reached zero at some point: as measured the bodies met. Nobody was hit,
                      so a position was wrong by at least the real gap, and the pair says nothing
                      about how close the call was. A pedestrian waiting at the kerb while a bus
                      passes, measured a metre into the lane, is the usual case
      no_approach     no approach was watched: TTC came down from above the level's upper edge faster than a
                      steady approach can, or dropped into its minimum by more than a second between two frames
                      (a position jump); for a person or a bicycle also when it began inside the level and was
                      not followed for a second or more while it fell. A near-miss
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
    enc = {}; done = []; dup_cache = {}
    ed = {g: edges_of(o) for g, o in objs.items()}

    def flush(key, e):
        if key not in dup_cache:
            dup_cache[key] = same_object(objs[key[0]], objs[key[1]])
        A = objs[key[0]]; B = objs[key[1]]; edges = ed[key[1]]; m = e["min"]
        if dup_cache[key]: base = "duplicate"
        elif m == math.inf: base = "overlap"
        elif e.get("touched"): base = "contact"
        elif not A["obs"][e["iA"]] or not B["obs"][e["iB"]]: base = "unobserved"
        else: base = ""
        off = B["cls"] not in VEHICLE and field is not None and not on_road(field, B["E"][e["iB"]], B["N"][e["iB"]])
        own = next((h for h in edges if m <= h), None)               # the level of the measured minimum
        h = own if own is not None else edges[-1]
        flag = base
        if not flag:
            ta = e["t_above_at_min"].get(h)
            if e["max_before_min"] >= h:
                # seen coming from beyond the edge: it must not have fallen to the minimum in less than half the
                # time a steady approach needs (that is a position jump, not a vehicle closing in)
                seen = ta is not None and (e["t"] - ta) >= 0.5 * (h - m)
            elif B["cls"] in VEHICLE:
                # it began inside the level, as a vehicle does that sets off behind another in a queue. The
                # simulation tools count that (SUMO and SSAM set no condition on speed or on how an approach
                # begins), so it counts here too, unless TTC dropped into its minimum by more than a second from
                # one frame to the next: that is a position jump
                seen = not (e.get("drop") is not None and e["drop"] > 1.0)
            else:
                # a person or a bicycle already inside the level in the first frame is usually someone at the kerb
                # beside a passing vehicle: it counts only when it was followed for a second or more while TTC
                # fell steadily
                lasted = e["t"] - e["t0"]
                seen = (e["ttc0"] is not None and lasted >= 1.0 and lasted >= 0.5 * (e["ttc0"] - m)
                        and e.get("drop") is not None and e["drop"] <= 1.0)
            if not seen:
                flag = "no_approach"
            elif off:
                flag = "off_road_party"
            elif e["min_inner"] > h:
                flag = "within_position_error"
        done.append(dict(ttc=(m if m < math.inf else 0.0), t=e["t"], E=e["E"], N=e["N"], gA=key[0], gB=key[1],
                         iA=e["iA"], iB=e["iB"], flag=flag, band=own))

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
                if t > watch_s:
                    continue
                t_inner = ttc_pair(A, iA, B, iB, margin)            # the same course against the shrunk box
                key = (gA, gB); tt = A["T"][iA]
                e = enc.get(key)
                if e is None or fr - e["last"] > 6:                  # a new encounter after a gap
                    if e is not None:
                        flush(key, e)
                    e = enc[key] = dict(min=math.inf, t=tt, E=B["E"][iB], N=B["N"][iB], iB=iB, iA=iA, last=fr,
                                        run_max=0.0, max_before_min=0.0, min_inner=math.inf, t_above={}, t_above_at_min={},
                                        t0=tt, ttc0=None, prev=None)
                if 0 < t_inner < e["min_inner"]:
                    e["min_inner"] = t_inner
                if e["ttc0"] is None and t > 0:
                    e["ttc0"] = t                                    # the first value of the encounter
                if 0 < t < e["min"]:
                    e["drop"] = (e["prev"] - t) if e["prev"] is not None else None   # the step into this new low
                    e["max_before_min"] = e["run_max"]               # the largest TTC seen before this new low
                    e["t_above_at_min"] = dict(e["t_above"])         # when TTC was last beyond each edge
                    e.update(min=t, t=tt, E=B["E"][iB], N=B["N"][iB], iB=iB, iA=iA)
                elif t == 0 and e["min"] == math.inf:
                    e.update(t=tt, E=B["E"][iB], N=B["N"][iB], iB=iB, iA=iA)
                if t == 0:
                    e["touched"] = True
                for h in ed[gB]:
                    if t >= h:
                        e["t_above"][h] = tt
                e["run_max"] = max(e["run_max"], t)
                if t > 0:
                    e["prev"] = t
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


def level_of(measure, pair, seconds):
    """upper edge of the severity level a value falls in; None above the last edge"""
    for h in LEVELS[measure][pair_key(pair)]:
        if seconds <= h:
            return h
    return None


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
        ev = [d for d in events if d["measure"] == m and (m == "TTC" or d["type"] in COUNTED)]
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
    nm = [d for d in events if d["most_severe"]]
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
    ap.add_argument("--min-speed", type=float, default=1.0, help="m/s; a vehicle slower than this cannot run into anyone")
    ap.add_argument("--margin", type=float, default=0.8, help="m; position error the TTC geometry must beat before an encounter counts")
    ap.add_argument("--start", default=None, help="wall-clock time of t=0, HH:MM:SS, for the event list")
    ap.add_argument("--lane-field", default=os.path.join(CAL, "lane_field_sanford_cedar.json"),
                    help="build_lane_field.py output; a person off it cannot be on a vehicle's path")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = None
    if a.start:
        hh, mm, ss = (float(v) for v in a.start.split(":")); t0 = hh * 3600 + mm * 60 + ss
    clock = lambda t: ("%02d:%02d:%05.2f" % (int((t0 + t) // 3600), int((t0 + t) % 3600 // 60), (t0 + t) % 60)) if t0 is not None else ""

    objs = load(a.tracks)
    print("%d objects loaded" % len(objs))
    field = None
    if a.lane_field and os.path.exists(a.lane_field):
        j = json.load(open(a.lane_field))
        field = dict(cell=float(j["cell_m"]), cells={tuple(int(x) for x in k.split(",")): (v["vehicles"],) for k, v in j["cells"].items()})
    ttc = ttc_events(objs, a.fps, lambda o: LEVELS["TTC"]["veh-" + kind(o["cls"])], field=field, min_speed=a.min_speed, margin=a.margin)
    pet = pet_events(objs, min_speed=a.min_speed)

    def row(measure, seconds, level, flag, t, e, n, A, B, gA, gB, typ):
        iA = min(int(np.searchsorted(A["T"], t)), len(A["T"]) - 1); iB = min(int(np.searchsorted(B["T"], t)), len(B["T"]) - 1)
        return dict(measure=measure, seconds=round(seconds, 2), level=level if level is not None else "", most_severe=0, flag=flag,
                    t_sec=round(t, 2), time=clock(t), E=round(e, 2), N=round(n, 2), lat=round(LAT0 + n / KY, 7), lon=round(LON0 + e / KX, 7),
                    pair="%s-%s" % (kind(A["cls"]), kind(B["cls"])), gid_a=gA, cls_a=A["cls"], speed_a=round(float(A["speed"][iA]), 1),
                    gid_b=gB, cls_b=B["cls"], speed_b=round(float(B["speed"][iB]), 1), type=typ)

    events = []
    for d in ttc:
        if d["ttc"] > COLLECT_S:
            continue                                                 # watched from 8 s only to see the approach
        A, B = objs[d["gA"]], objs[d["gB"]]
        events.append(row("TTC", d["ttc"], d["band"] if d["flag"] == "" else None, d["flag"], d["t"], d["E"], d["N"], A, B, d["gA"], d["gB"], ""))
        events[-1]["_band"] = d["band"]
    # two vehicles are evaluated from both sides (each as the box in turn): one encounter per pair and 3 s
    sides = defaultdict(list)
    for d in events:
        if d["pair"] == "veh-veh" and d["_band"] is not None:
            sides[tuple(sorted((d["gid_a"], d["gid_b"])))].append(d)
    for L in sides.values():
        L.sort(key=lambda d: d["t_sec"]); clusters = []
        for d in L:
            if not clusters or d["t_sec"] - clusters[-1][-1]["t_sec"] > 3.0:
                clusters.append([])
            clusters[-1].append(d)
        for c in clusters:
            keep = min([d for d in c if d["flag"] == ""] or c, key=lambda d: d["seconds"])
            for d in c:
                if d is not keep and d["flag"] == "":
                    d["flag"] = "same_pair"; d["level"] = ""         # the same encounter seen from the other vehicle
    for d in events:
        del d["_band"]
    print("encounters: %d with TTC under 5 s, %d with PET under 5 s" % (len(events), len(pet)))
    for p, t, e, n, gA, gB, crossing, flag in pet:
        A, B = objs[gA], objs[gB]
        typ = encounter_type.encounter(objs, gA, gB, t, p, e, n)["type"]
        pr = "%s-%s" % (kind(A["cls"]), kind(B["cls"]))
        events.append(row("PET", p, level_of("PET", pr, round(p, 2)) if flag == "" and typ in COUNTED else None, flag, t, e, n, A, B, gA, gB, typ))
    for d in events:
        d["most_severe"] = int(d["level"] != "" and d["level"] == LEVELS[d["measure"]][pair_key(d["pair"])][0])
    events.sort(key=lambda d: (d["measure"], d["level"] == "", d["seconds"]))
    with open(os.path.join(a.out, "near_misses.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0])); w.writeheader(); w.writerows(events)

    ext = json.load(open(os.path.join(CAL, "aerial_wide.json"))); E0, N1, RES = ext["E0"], ext["N1"], ext["res"]
    base = cv2.imread(os.path.join(CAL, "aerial_wide.png"))
    lines = []
    for m in ("TTC", "PET"):
        tot = 0
        for k in ("veh-veh", "veh-ped", "veh-bike"):
            c = [sum(1 for d in events if d["measure"] == m and pair_key(d["pair"]) == k and d["level"] == h) for h in LEVELS[m][k]]
            tot += sum(c)
            lines.append("%s  %-22s levels up to %s s: %s  (%d)" % (m, PAIR_NAME[k], " / ".join("%g" % h for h in LEVELS[m][k]), " / ".join(str(x) for x in c), sum(c)))
        sev = [d for d in events if d["measure"] == m and d["most_severe"]]
        lines.append("%s  counted %d, of them at the most severe level %d" % (m, tot, len(sev)))
        heatmap([(d["E"], d["N"]) for d in sev], base, E0, N1, RES, "%s, most severe level: %d" % (m, len(sev)),
                os.path.join(a.out, "nearmiss_%s.png" % m.lower()))
    ttc_all = [d for d in events if d["measure"] == "TTC" and d["level"] != ""]
    heatmap([(d["E"], d["N"]) for d in ttc_all], base, E0, N1, RES, "TTC conflicts, all levels: %d" % len(ttc_all),
            os.path.join(a.out, "conflicts_ttc.png"))
    sev_all = [d for d in events if d["most_severe"]]
    event_map(sev_all, base, E0, N1, RES, max(LEVELS[m]["veh-veh"][0] for m in LEVELS),
              "most severe level, TTC or PET: %d" % len(sev_all), os.path.join(a.out, "nearmiss_events.png"))
    distributions(events, LEVELS["TTC"]["veh-veh"][0], LEVELS["PET"]["veh-veh"][0], os.path.join(a.out, "nearmiss_distribution.png"))
    if over_time(events, clock, os.path.join(a.out, "nearmiss_over_time.png")):
        lines.append("most severe level over time: nearmiss_over_time.png")
    enc = [d for d in events if d["measure"] == "PET" and d["flag"] == ""]
    by_type = defaultdict(int)
    for d in enc: by_type[d["type"]] += 1
    lines.append("PET encounters under 5 s by type: crossing %d and merging %d (counted, %d); following %d (same lane, a matter for TTC) "
                 "and oncoming %d (opposite lanes), not counted" % (by_type["crossing"], by_type["merging"], by_type["crossing"] + by_type["merging"],
                                                                   by_type["following"], by_type["oncoming"]))
    counted = [d for d in enc if d["type"] in COUNTED]
    heatmap([(d["E"], d["N"]) for d in counted], base, E0, N1, RES, "PET encounters under 5 s on crossing or merging paths: %d" % len(counted),
            os.path.join(a.out, "encounters_pet.png"))
    for m in ("TTC", "PET"):
        flags = defaultdict(int)
        for d in events:
            if d["measure"] == m and d["flag"] and d["seconds"] <= LEVELS[m][pair_key(d["pair"])][-1]:
                flags[d["flag"]] += 1
        lines.append("%s inside the last level but not counted: %s" % (m, ", ".join("%s %d" % kv for kv in sorted(flags.items())) or "none"))
    txt = "\n".join(lines); print(txt)
    open(os.path.join(a.out, "safety_summary.txt"), "w").write(txt + "\n")
    print("most severe level:")
    for d in [d for d in events if d["measure"] == "TTC" and d["most_severe"]][:6] + [d for d in events if d["measure"] == "PET" and d["most_severe"]][:6]:
        print("   %s %5.2f s  %s  %-8s  %s %d (%.1f m/s) vs %s %d (%.1f m/s)  at E %.1f N %.1f"
              % (d["measure"], d["seconds"], d["time"], d["pair"], d["cls_a"], d["gid_a"], d["speed_a"], d["cls_b"], d["gid_b"], d["speed_b"], d["E"], d["N"]))


if __name__ == "__main__":
    main()
