"""
stitch_tracks.py - one camera's tracks, after the tracker and before the fusion: mark what is hidden,
and give a road user that comes back from behind another one its old number.

    python stitch_tracks.py --in sample/cam03_1019/trajectories.csv --out sample/cam03_1019/trajectories.csv

Six things the online tracker cannot do, all in image space and all offline.

1. Occlusion flag (column occ). A car half hidden behind a bus still gets a mask, but only of the part
   that shows: the mask's bottom edge is then the bus's roof line, not the tyres, and the foot point
   the whole chain is built on sits metres too far back. The test is geometric: points just below the
   mask's bottom edge that fall inside another road user's mask mean something nearer is covering the
   foot (occ = 1); the same test along the left and right edges against a nearer mask marks a side cut
   (occ = 2, the foot is seen but the visible width is not the whole width). The detection is kept;
   the fusion trusts it less and never anchors a hand-over on it.

2. Observation-centric recovery. The tracker follows a lost track with its Kalman prediction, and the
   prediction of a car that was being covered runs away, because the shrinking box looked like motion.
   When the car reappears where it stood, nothing links it to its old track and it gets a new number
   (in the two-minute sample one waiting car became three objects, and the middle one, which never
   moved, was dropped as parked). Here a track that ends is compared with tracks that begin later by
   their last and first RELIABLE boxes (not hidden, not cut by the frame edge), the idea of OC-SORT's
   observation-centric recovery (Cao et al., CVPR 2023) applied offline: same kind of road user, the
   two boxes overlap (IoU >= 0.5), the old track was slow enough that it is still expected there, and
   for a gap longer than 5 s the place must have been covered by a nearer road user for most of the
   gap. One successor per track, closest in time first.

3. Figures on a vehicle. A vehicle carries people who are not road users: the faces printed on a
   bus's advertising, its passengers behind the windows, and the driver of a car seen through the
   windscreen. The detector returns them as persons, the tracker follows them along with the
   vehicle, and the fusion puts "pedestrians" in the carriageway at vehicle speed (the one
   vehicle-pedestrian near-miss by time to collision in 71 minutes was a car and its own driver). A
   person whose outline lies within a vehicle's silhouette (80 % of it; the convex hull of the
   vehicle's mask, because the mask itself has a hole where the figure is) with the feet above that
   vehicle's lower edge (by 5 % of its height: someone standing in front of it has the feet on the
   road, at or below that edge) is dropped, and a track that is such a figure in half its frames or
   more is dropped whole.
   A bicycle on a car's rack is the same kind of thing: detected as a bicycle, followed along with
   the car, and exported as a cyclist at car speed. A bicycle track is dropped when one and the same
   vehicle track holds it (the bicycle's centre within the vehicle's box, widened by a quarter) in
   80 % of its frames or more, that vehicle travels (more than its own box diagonal), the bicycle
   keeps its place on it (its offset from the vehicle's centre, in units of the vehicle's box, varies
   by under 0.15) and nobody rides it (a person's box over the bicycle in under 20 % of the frames).

4. Hops. Where a track jumps by most of its own size from one processed frame to the next and was
   not moving that way before or after, it has been handed to a different road user (the tracker's
   wider second match can do that where detections come and go, as on a row of parked cars 40 m
   away). It is split at the jump, and the rest gets a new number. This is done first.

5. Two kinds in one track. The tracker matches boxes, not classes: a pedestrian track that is lost for
   two seconds is carried on by its prediction and can be picked up by a car that drives in where the
   prediction went. The fusion gives a track one class, by majority, so the pedestrian's rows became
   vehicle positions and the car a path it never drove (in 71 minutes 117 tracks held both, the
   longest a parked car that walked off with a passer-by for 529 frames). A track is split wherever
   it changes between person (pedestrian, rider) and vehicle AND the box is another box. Where the box
   stays (IoU >= 0.5 across the change: someone seen from straight above, called a car for a few
   frames) it is the detector's label that slipped, not another road user: the track stays whole and
   the rows of the rarer kind take the commoner label. Done before the hops.

6. Relay behind a thin occluder. A car that passes behind a signal pole is, for two or three frames,
   two boxes: what is left of it on one side, shrinking, and what has come out on the other side,
   growing. The tracker keeps the old number on the shrinking box until it is gone and gives the
   growing one a new number, so the two tracks exist together and no gap rule can join them. A
   vehicle track that ends while another vehicle track, begun within the last three processed
   frames, holds its box (80 % of the smaller box inside the larger in every frame both exist) is
   that other track's beginning: the new track takes the old number, and of the frames both have,
   the larger box is kept. Only for a vehicle on the move: the old box must have shrunk to under
   half within its last eight frames and the new track must travel half its width within its first
   eight (parked cars in a row trade boxes with their neighbours without either). Vehicles only: in
   a group of pedestrians one box inside another is two people.

Writes the same columns plus occ; track_id of a recovered track is replaced by its predecessor's.
"""
import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

FRAME_W, FRAME_H = 1920, 1080
EDGE_PX = 15
OCC_SHARE = 0.4          # share of the test points along an edge that must lie in another mask
MIN_OCCLUDER = 0.3       # an occluder is at least this wide, as a share of the width of what it hides
GAP_FREE_S = 5.0         # up to here a return on the same spot needs no further evidence
GAP_MAX_S = 30.0         # beyond GAP_FREE_S the spot must have been covered; never further than this
IOU_SAME = 0.5
COVER_SHARE = 0.5        # of the gap's frames with a nearer road user over the spot
PED = {"Pedestrian"}
BIG_CLS = {"Bus", "Truck"}
ON_VEHICLE_SHARE = 0.8   # of a person's outline inside the vehicle's outline
ON_VEHICLE_LIFT = 0.05   # feet above the vehicle's lower edge by this share of the vehicle's height
HOP = 0.8                # a step between two processed frames longer than this share of the box's longer side is a jump


def kind(cls):
    return "ped" if cls in PED else ("bike" if cls in ("Bicycle", "Motorcycle") else "veh")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix, 1e-6)


def share_inside(a, b):
    """share of box a that lies inside box b"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def parse_poly(s):
    if not s:
        return None
    P = np.array([[int(v) for v in q.split(",")] for q in s.split(";")], np.int32)
    return P if len(P) >= 3 else None


def edge_profile(P, at, axis, pick):
    """Where the outline P crosses the line x = at (axis 0) or y = at (axis 1): the largest (pick = max)
    or smallest crossing coordinate on the other axis, None if the line misses the outline."""
    a = P[:, axis].astype(float); b = P[:, 1 - axis].astype(float)
    a2 = np.roll(a, -1); b2 = np.roll(b, -1)
    m = (np.minimum(a, a2) <= at) & (at <= np.maximum(a, a2)) & (a != a2)
    if not m.any():
        return None
    v = b[m] + (at - a[m]) * (b2[m] - b[m]) / (a2[m] - a[m])
    return float(pick(v))


def inside(d, x, y):
    """is the image point inside road user d: its mask outline where there is one, else its box"""
    b = d["box"]
    if not (b[0] <= x <= b[2] and b[1] <= y <= b[3]):
        return False
    if d["poly"] is None:
        return True
    return cv2.pointPolygonTest(d["poly"], (float(x), float(y)), False) >= 0


def occlusion(d, others):
    """0 = free, 1 = the foot is covered (another mask directly under the bottom edge), 2 = a side is
    covered by a nearer road user (the bottom edge is seen, the visible width is not all of it)"""
    x1, y1, x2, y2 = d["box"]; w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return 0
    # Only a road user of some size relative to this one can hide its foot: people boarding stand in front of a bus's
    # lower edge without moving its lowest point, and a bus at its stop was flagged covered for most of its dwell.
    near = [o for o in others if o is not d and o["box"][0] <= x2 + 8 and o["box"][2] >= x1 - 8 and o["box"][1] <= y2 + 0.2 * h + 8 and o["box"][3] >= y1
            and (o["box"][2] - o["box"][0]) >= MIN_OCCLUDER * w]
    if not near:
        return 0
    P = d["poly"]
    # bottom edge: 7 columns, two depths below the outline
    dy = max(3.0, 0.04 * h); hit = 0; n = 7
    for x in np.linspace(x1 + 0.15 * w, x2 - 0.15 * w, n):
        yb = edge_profile(P, x, 0, np.max) if P is not None else y2
        if yb is None:
            yb = y2
        if any(inside(o, x, yb + dy) or inside(o, x, yb + 2 * dy) for o in near):
            hit += 1
    if hit / n >= OCC_SHARE:
        return 1
    # sides: 5 rows on each side, only against road users whose own foot is clearly lower (nearer)
    front = [o for o in near if o["box"][3] >= y2 + max(4.0, 0.1 * h)]
    if front:
        dx = max(3.0, 0.04 * w); m = 5
        for side in (0, 1):
            hit = 0
            for y in np.linspace(y1 + 0.2 * h, y2 - 0.2 * h, m):
                xe = edge_profile(P, y, 1, np.max if side else np.min) if P is not None else (x2 if side else x1)
                if xe is None:
                    xe = x2 if side else x1
                xt = xe + dx if side else xe - dx
                if any(inside(o, xt, y) or inside(o, xt + (dx if side else -dx), y) for o in front):
                    hit += 1
            if hit / m >= 0.6:
                return 2
    return 0


def on_vehicle(d, others):
    """a person whose outline lies on a vehicle with the feet above its lower edge: printed on it, or riding in it"""
    if d["cls"] != "Pedestrian" or d["poly"] is None:
        return False
    x1, y1, x2, y2 = d["box"]
    for o in others:
        if o is d or kind(o["cls"]) != "veh" or o["poly"] is None:
            continue
        b = o["box"]
        if not (b[0] <= x1 and x2 <= b[2] and b[1] <= y1 and y2 <= b[3]):
            continue
        # the vehicle's convex hull, not its mask: the detector cuts the figure out of the bus's mask (the pixels go
        # to the "person"), so the mask has a hole exactly where the figure is
        if "hull" not in o:
            o["hull"] = cv2.convexHull(o["poly"]).reshape(-1, 2)
        H = o["hull"]
        pts = [(x, y) for x in np.linspace(x1, x2, 6)[1:-1] for y in np.linspace(y1, y2, 10)[1:-1]
               if cv2.pointPolygonTest(d["poly"], (float(x), float(y)), False) >= 0] or [((x1 + x2) / 2, (y1 + y2) / 2)]
        if np.mean([cv2.pointPolygonTest(H, (float(x), float(y)), False) >= 0 for x, y in pts]) < ON_VEHICLE_SHARE:
            continue
        yb = edge_profile(H, (x1 + x2) / 2, 0, np.max)
        if yb is None:
            yb = b[3]
        if y2 <= yb - ON_VEHICLE_LIFT * (b[3] - b[1]):
            return True
    return False


def cut_hops(tracks, stride, next_id):
    """A track that jumps further than HOP of its longer side between two consecutive processed frames, without
    moving that way just before or just after, has hopped onto another road user (the tracker's wider second match
    can do that where detections come and go): it is split there, and what follows gets a new number."""
    cuts = 0
    for tid in list(tracks):
        L = tracks[tid]; L.sort(key=lambda d: d["fr"])
        C = np.array([[(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2] for d in L])
        k = 1
        while k < len(L):
            if L[k]["fr"] - L[k - 1]["fr"] != stride:
                k += 1
                continue
            side = max(L[k - 1]["box"][2] - L[k - 1]["box"][0], L[k - 1]["box"][3] - L[k - 1]["box"][1])
            j = C[k] - C[k - 1]; mv = float(np.hypot(j[0], j[1]))
            if mv <= HOP * side:
                k += 1
                continue

            def agrees(i0, i1):
                # mean step over rows i0..i1, if they are consecutive processed frames: at least 0.3 of the jump, within 60 degrees of it
                if i0 < 0 or i1 >= len(L) or i1 - i0 < 1 or L[i1]["fr"] - L[i0]["fr"] != (i1 - i0) * stride:
                    return False
                m = (C[i1] - C[i0]) / (i1 - i0); n = float(np.hypot(m[0], m[1]))
                return n >= 0.3 * mv and float(m @ j) / (n * mv) > 0.5
            if agrees(max(k - 4, 0), k - 1) or agrees(k, min(k + 3, len(L) - 1)):
                k += 1
                continue
            rest = L[k:]; del L[k:]
            for d in rest:
                d["tid"] = next_id; d["row"]["track_id"] = next_id
            tracks[next_id] = rest; next_id += 1; cuts += 1
            break                                            # the remainder is examined as its own track
    return cuts


CARRIED_SHARE = 0.8      # of a bicycle track's frames in which one vehicle track holds it
CARRIED_GROW = 0.25      # the vehicle's box is widened by this share of its size on every side
CARRIED_STILL = 0.15     # the bicycle's offset on the vehicle, in vehicle-box units, varies by less than this
RIDER_SHARE = 0.2        # frames with a person over the bicycle: above this it is ridden, not carried


def carried_bicycles(tracks, frames):
    """Bicycle tracks that ride on a vehicle's rack: see point 3 of the header. Returns {bicycle track: vehicle track}."""
    out = {}
    for tid, L in tracks.items():
        if len(L) < 5 or sum(d["cls"] == "Bicycle" for d in L) < 0.5 * len(L):
            continue
        holder = defaultdict(list); ridden = 0
        for d in L:
            b = d["box"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            for o in frames[d["fr"]]:
                if o is d:
                    continue
                q = o["box"]
                if kind(o["cls"]) == "veh":
                    gw, gh = CARRIED_GROW * (q[2] - q[0]), CARRIED_GROW * (q[3] - q[1])
                    if q[0] - gw <= cx <= q[2] + gw and q[1] - gh <= cy <= q[3] + gh:
                        holder[o["tid"]].append((d["fr"], (cx - (q[0] + q[2]) / 2) / max(q[2] - q[0], 1.0), (cy - (q[1] + q[3]) / 2) / max(q[3] - q[1], 1.0),
                                                 (q[0] + q[2]) / 2, (q[1] + q[3]) / 2, float(np.hypot(q[2] - q[0], q[3] - q[1]))))
                elif o["cls"] == "Pedestrian":
                    ix = max(0.0, min(b[2], q[2]) - max(b[0], q[0])); iy = max(0.0, min(b[3], q[3]) - max(b[1], q[1]))
                    if ix * iy >= 0.3 * (b[2] - b[0]) * (b[3] - b[1]):
                        ridden += 1; break
        if not holder or ridden > RIDER_SHARE * len(L):
            continue
        vt, H = max(holder.items(), key=lambda kv: len(kv[1]))
        if len(H) < CARRIED_SHARE * len(L):
            continue
        H.sort(); A = np.array([h[1:] for h in H])
        travelled = float(np.hypot(A[-1, 2] - A[0, 2], A[-1, 3] - A[0, 3]))
        if travelled <= float(np.median(A[:, 4])):
            continue                                         # the vehicle stands: a bicycle parked beside a parked car
        if float(np.hypot(A[:, 0].std(), A[:, 1].std())) >= CARRIED_STILL:
            continue                                         # it moves along the vehicle: a rider passing, or being passed
        out[tid] = vt
    return out


def cut_kinds(tracks, next_id):
    """A track that is a person and then a vehicle (or the other way round) is two road users: split where the kind
    changes and the box is another box. Where the box stays (IoU >= IOU_SAME across the change) it is one road user
    whose label slipped: no split, and the rows of the rarer kind take the label of the commoner one.
    Returns (cuts, relabelled rows)."""
    cuts = relabelled = 0
    for tid in list(tracks):
        L = tracks[tid]; L.sort(key=lambda d: d["fr"])
        K = [kind(d["cls"]) == "veh" for d in L]
        if all(K) or not any(K):
            continue
        at = [k for k in range(1, len(L)) if K[k] != K[k - 1] and iou(L[k - 1]["box"], L[k]["box"]) < IOU_SAME]
        pieces = [L[a_:b_] for a_, b_ in zip([0] + at, at + [len(L)])]
        for n_, piece in enumerate(pieces):
            kv = [kind(d["cls"]) == "veh" for d in piece]
            if any(kv) and not all(kv):                         # one road user under two labels: the commoner kind names it
                major = sum(kv) * 2 >= len(kv)
                names = defaultdict(int)
                for d, v in zip(piece, kv):
                    if v == major:
                        names[d["cls"]] += 1
                name = max(names, key=names.get)
                for d, v in zip(piece, kv):
                    if v != major:
                        d["cls"] = name; d["row"]["class"] = name; relabelled += 1
            if n_ > 0:
                for d in piece:
                    d["tid"] = next_id; d["row"]["track_id"] = next_id
                tracks[next_id] = piece; next_id += 1; cuts += 1
        if len(pieces) > 1:
            del L[len(pieces[0]):]
    return cuts, relabelled


RELAY_FRAMES = 3         # two tracks of one vehicle may both exist for up to this many processed frames
RELAY_INSIDE = 0.8       # share of the smaller box inside the larger, in every frame both exist
RELAY_SHRUNK = 0.5       # the old track's last box against its largest of the last eight: it was disappearing


def relay(tracks, frames, stride):
    """Vehicle tracks that hand over behind a thin occluder (point 6 of the header) are joined: the track that begins
    takes the number of the one that ends. Returns (joins, set of id() of the dropped duplicate rows, [(old number, new track, frame)])."""
    joins = 0; dropped = set(); pairs = []
    changed = True
    while changed:
        changed = False
        for L in tracks.values():
            L.sort(key=lambda d: d["fr"])
        first = {t: L[0]["fr"] for t, L in tracks.items()}; last = {t: L[-1]["fr"] for t, L in tracks.items()}
        for a in sorted(tracks, key=lambda t: last[t]):
            if a not in tracks or kind(tracks[a][-1]["cls"]) != "veh":
                continue
            La = tracks[a]; fa = last[a]; at = {d["fr"]: d for d in La[-(RELAY_FRAMES + 1):]}
            best = None
            for d in frames[fa]:
                b = d["tid"]
                if b == a or b not in tracks or kind(d["cls"]) != "veh":
                    continue
                if not (fa - RELAY_FRAMES * stride < first[b] <= fa) or last[b] <= fa or first[a] >= first[b]:
                    continue
                common = [e for e in tracks[b] if e["fr"] <= fa]
                if not common or any(e["fr"] not in at for e in common):
                    continue
                share = []
                for e in common:
                    p_, q_ = at[e["fr"]]["box"], e["box"]
                    ix = max(0.0, min(p_[2], q_[2]) - max(p_[0], q_[0])); iy = max(0.0, min(p_[3], q_[3]) - max(p_[1], q_[1]))
                    share.append(ix * iy / max(min((p_[2] - p_[0]) * (p_[3] - p_[1]), (q_[2] - q_[0]) * (q_[3] - q_[1])), 1.0))
                if min(share) < RELAY_INSIDE:
                    continue
                # ...and it is a vehicle on the move going behind something: the old box has shrunk to under half of what
                # it was a few frames earlier, and the new track travels by half its own width within its first second.
                # Without this the rule fired on rows of parked cars 40 m away, where the tracker's boxes swap between
                # neighbours all the time (23 of 30 joins in 71 minutes, seen on the footage).
                area = lambda q: (q[2] - q[0]) * (q[3] - q[1])
                if area(La[-1]["box"]) > RELAY_SHRUNK * max(area(e["box"]) for e in La[-8:]):
                    continue
                Lb = tracks[b][:8]; b0, b1 = Lb[0]["box"], Lb[-1]["box"]
                if abs((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2 + abs((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2 < 0.5 * (b1[2] - b1[0]):
                    continue
                if best is None or min(share) > best[0]:
                    best = (min(share), b, common)
            if best is None:
                continue
            _, b, common = best
            for e in common:                                  # of the frames both have, the larger box stays
                pa = at[e["fr"]]
                small = pa if (pa["box"][2] - pa["box"][0]) * (pa["box"][3] - pa["box"][1]) <= (e["box"][2] - e["box"][0]) * (e["box"][3] - e["box"][1]) else e
                dropped.add(id(small["row"])); frames[e["fr"]].remove(small)
                (La if small is pa else tracks[b]).remove(small)
            for e in tracks[b]:
                e["tid"] = a; e["row"]["track_id"] = a
            La.extend(tracks[b]); del tracks[b]
            joins += 1; changed = True; pairs.append((a, b, fa))
            break
    return joins, dropped, pairs


def edge_cut(b):
    return b[3] >= FRAME_H - 10 or b[0] <= EDGE_PX or b[2] >= FRAME_W - EDGE_PX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="trajectories.csv of track_clip_seg.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=15.0, help="frame rate the frame column counts in")
    ap.add_argument("--no-recover", action="store_true", help="write the occ column only")
    ap.add_argument("--keep-figures", action="store_true", help="keep persons that lie on a bus or lorry (advertising, passengers)")
    ap.add_argument("--report", default=None, help="text file listing the recovered tracks")
    a = ap.parse_args()

    with open(a.src, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; raw = list(rd)
    frames = defaultdict(list); tracks = defaultdict(list)
    for r in raw:
        d = dict(fr=int(r["frame"]), tid=int(r["track_id"]), cls=r["class"], row=r,
                 box=(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])), poly=parse_poly(r.get("poly", "")))
        frames[d["fr"]].append(d); tracks[d["tid"]].append(d)
    # 0. two kinds in one track: split; then hops: a track that jumped onto another road user is split where it jumped
    n_kind, n_relab = cut_kinds(tracks, max(tracks) + 1)
    print("kinds: %d tracks split where they changed between person and vehicle, %d rows relabelled (same box, the label slipped)" % (n_kind, n_relab))
    all_fr = sorted(frames); stride0 = min((q - p for p, q in zip(all_fr[:-1], all_fr[1:])), default=1)
    n_hops = 0
    while True:
        c_ = cut_hops(tracks, stride0, max(tracks) + 1)
        n_hops += c_
        if not c_:
            break
    print("hops: %d tracks split where they jumped onto another road user" % n_hops)
    n_relay, gone_rows, relay_pairs = relay(tracks, frames, stride0)
    if gone_rows:
        raw = [r for r in raw if id(r) not in gone_rows]
    print("relay: %d vehicle tracks continued by the track that took over behind a thin occluder" % n_relay)
    for a_, b_, f_ in relay_pairs:
        print("  relay %d <- %d at frame %d" % (a_, b_, f_))
    # 3. figures on a vehicle, first: what is not a road user is no occluder and no track either
    if not a.keep_figures:
        n_fig = 0; fig_tracks = set()
        for fr, L in frames.items():
            for d in L:
                d["fig"] = on_vehicle(d, L); n_fig += d["fig"]
        for tid, L in tracks.items():
            if sum(d["fig"] for d in L) >= 0.5 * len(L) and any(d["fig"] for d in L):
                fig_tracks.add(tid)
        carried = carried_bicycles(tracks, frames)
        fig_tracks.update(carried)
        gone = sum(1 for L in frames.values() for d in L if d["fig"] or (d["tid"] in fig_tracks and d["tid"] not in carried))
        for fr in list(frames):
            frames[fr] = [d for d in frames[fr] if not (d["fig"] or d["tid"] in fig_tracks)]
        for tid in list(tracks):
            tracks[tid] = [d for d in tracks[tid] if not (d["fig"] or d["tid"] in fig_tracks)]
            if not tracks[tid]:
                del tracks[tid]
        kept_rows = set(id(d["row"]) for L in frames.values() for d in L)
        raw = [r for r in raw if id(r) in kept_rows]
        print("figures on a vehicle: %d person detections dropped (printed on a vehicle, or riding in it), %d tracks dropped whole" % (gone, len(fig_tracks) - len(carried)))
        print("bicycles carried by a vehicle: %d tracks dropped%s" % (len(carried), "".join("  [%d on %d]" % kv for kv in sorted(carried.items()))))
    # 1. occlusion flag
    n_occ = [0, 0, 0]
    for fr, L in frames.items():
        for d in L:
            d["occ"] = occlusion(d, L); n_occ[d["occ"]] += 1
    tot = max(sum(n_occ), 1)
    print("occlusion: %d detections, foot covered %d (%.1f %%), side covered %d (%.1f %%)" % (tot, n_occ[1], 100 * n_occ[1] / tot, n_occ[2], 100 * n_occ[2] / tot))

    links = []
    if not a.no_recover:
        all_fr = sorted(frames); stride = min((q - p for p, q in zip(all_fr[:-1], all_fr[1:])), default=1)
        info = {}
        for tid, L in tracks.items():
            L.sort(key=lambda d: d["fr"])
            if len(L) < 4:
                continue
            ok = [d for d in L if d["occ"] == 0 and not edge_cut(d["box"])]
            if len(ok) < 3:
                continue
            k = [kind(d["cls"]) for d in L]; kd = max(set(k), key=k.count)
            tail = [d for d in ok if d["fr"] >= ok[-1]["fr"] - a.fps]            # the last second of reliable boxes
            head = [d for d in ok if d["fr"] <= ok[0]["fr"] + a.fps]
            e_box = np.median(np.array([d["box"] for d in tail[-5:]]), axis=0); s_box = np.median(np.array([d["box"] for d in head[:5]]), axis=0)

            def speed(seg):
                if len(seg) < 2 or seg[-1]["fr"] == seg[0]["fr"]:
                    return 0.0
                c0 = np.array([(seg[0]["box"][0] + seg[0]["box"][2]) / 2, (seg[0]["box"][1] + seg[0]["box"][3]) / 2])
                c1 = np.array([(seg[-1]["box"][0] + seg[-1]["box"][2]) / 2, (seg[-1]["box"][1] + seg[-1]["box"][3]) / 2])
                return float(np.hypot(*(c1 - c0))) / ((seg[-1]["fr"] - seg[0]["fr"]) / a.fps)
            info[tid] = dict(kind=kd, f0=L[0]["fr"], f1=L[-1]["fr"], e_box=e_box, s_box=s_box, e_fr=ok[-1]["fr"], s_fr=ok[0]["fr"],
                             v_end=speed(tail), v_start=speed(head), n=len(L),
                             end_hidden=L[-1]["occ"] > 0 or L[-1]["fr"] > ok[-1]["fr"], start_hidden=L[0]["occ"] > 0 or L[0]["fr"] < ok[0]["fr"])
        cand = []
        for ta, A in info.items():
            if A["f1"] >= all_fr[-1] - stride:
                continue
            diag = float(np.hypot(A["e_box"][2] - A["e_box"][0], A["e_box"][3] - A["e_box"][1]))
            for tb, B in info.items():
                if tb == ta or B["kind"] != A["kind"] or B["f0"] <= A["f1"]:
                    continue
                gap = (B["f0"] - A["f1"] - stride) / a.fps                          # the missing time
                if gap > GAP_MAX_S:
                    continue
                dt = (B["s_fr"] - A["e_fr"]) / a.fps                                # between the two reliable observations
                if A["v_end"] * dt > 0.5 * diag or B["v_start"] * dt > 0.5 * diag:  # too fast to still be where it was last seen
                    continue
                io = iou(A["e_box"], B["s_box"])
                if io < IOU_SAME:
                    continue
                cover = None
                if gap > GAP_FREE_S:
                    fs = [fr for fr in all_fr if A["f1"] < fr < B["f0"]]; hid = 0
                    for fr in fs:
                        if any(share_inside(A["e_box"], o["box"]) >= 0.3 and o["box"][3] >= A["e_box"][3] - 0.1 * (A["e_box"][3] - A["e_box"][1])
                               and o["tid"] not in (ta, tb) for o in frames[fr]):
                            hid += 1
                    cover = hid / max(len(fs), 1)
                    if cover < COVER_SHARE:
                        continue
                # the spot must not have been held by a third track in between (that one is then the candidate, not this pair)
                taken = False
                for tc, Cc in info.items():
                    if tc in (ta, tb) or Cc["kind"] != A["kind"]:
                        continue
                    if Cc["f0"] < B["f0"] and Cc["f1"] > A["f1"] and (iou(A["e_box"], Cc["s_box"]) >= IOU_SAME or iou(A["e_box"], Cc["e_box"]) >= IOU_SAME):
                        taken = True; break
                if taken:
                    continue
                cand.append((gap, -io, ta, tb, cover))
        used_a, used_b = set(), set()
        for gap, nio, ta, tb, cover in sorted(cand):
            if ta in used_a or tb in used_b:
                continue
            used_a.add(ta); used_b.add(tb); links.append((ta, tb, gap, -nio, cover))
        # chains: b takes a's number, and a may itself have taken an earlier one
        new_id = {}
        for ta, tb, *_ in sorted(links, key=lambda x: info[x[1]]["f0"]):
            new_id[tb] = new_id.get(ta, ta)
        for r in raw:
            t = int(r["track_id"])
            if t in new_id:
                r["track_id"] = new_id[t]
        print("recovered: %d tracks continue an earlier one (%d vehicles, %d pedestrians, %d two-wheelers)" % (
            len(links), sum(info[x[0]]["kind"] == "veh" for x in links), sum(info[x[0]]["kind"] == "ped" for x in links), sum(info[x[0]]["kind"] == "bike" for x in links)))
        lines = ["  %s %d -> %d  gap %.1f s, boxes overlap %.2f%s" % (info[ta]["kind"], ta, tb, gap, io, "" if cover is None else ", spot covered %.0f %% of the gap" % (100 * cover))
                 for ta, tb, gap, io, cover in sorted(links, key=lambda x: info[x[1]]["f0"])]
        for ln in lines:
            print(ln)
        if a.report:
            with open(a.report, "w") as f:
                f.write("\n".join(lines) + "\n")

    occ_of = {id(d["row"]): d["occ"] for L in frames.values() for d in L}
    out_fields = fields + ([] if "occ" in fields else ["occ"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields); w.writeheader()
        for r in raw:
            r["occ"] = occ_of[id(r)]; w.writerow(r)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
