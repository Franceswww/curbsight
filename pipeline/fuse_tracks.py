"""
fuse_tracks.py - one object, one ID across the four roadside cameras.

    python fuse_tracks.py --clipdir <dir with camNN/trajectories.csv> --cams cam01,cam02,cam03,cam04

Every detection is put on the ground with its camera's model (out_refined/<cam>_fit_pnp/cam.json).
In each frame, detections of the same kind (pedestrian / vehicle) from different cameras that land
within a gate of each other are linked (greedy, closest first; gate 1.2 m for pedestrians, 3.0 m for
vehicles). Camera tracks that are linked in at least 3 frames are merged into one global ID
(union-find). A global object's position in a frame is the detection closest to its own camera
(the most accurate view); its class is the majority over all its detections, so a track that
flickers between pedestrian and motorcycle keeps one class.
Output: <clipdir>/fused.csv  (frame, gid, cls, E, N, n_cams, src = cam:id|cam:id, dist_m)
"""
import argparse
import csv
import os
from collections import Counter, defaultdict

import numpy as np

import render_demo as RD

GATE = {"Pedestrian": 1.2, "Bicycle": 1.5, "Motorcycle": 1.5, "Passenger Vehicle": 3.0, "Truck": 5.0, "Bus": 6.0}   # metres; two views of one
HEIGHT = {"Passenger Vehicle": 1.5, "Truck": 2.5, "Bus": 3.2, "Motorcycle": 1.2, "Bicycle": 1.2}                     # metres, for the 3-D box
MIN_LINK_FRAMES = 6                                                                                                # bus see different ends of it
MIN_LINK_RATIO = 0.5                   # ...and in at least half of the frames both tracks exist: two buses passing each other are close for a moment only
BIG = {"Bus", "Truck"}
GAP_S, GAP_BASE_M = 5.0, 3.0           # bridge a lost track if another of the same kind starts within GAP_S at a consistent position
FRAME_W, FRAME_H = 1920, 1080          # the view size the trajectories were made at (one quadrant of the 4K four-view frame)
# measurement noise of a camera's centre estimate, sigma = a + b * range (m), fitted to the RTK validation of
# 2026-10-27 (validate_rtk.py --cams, median error per range bin over both crossings): cam02 27-54 cm to 40 m then
# 132 cm, cam03 56-116 cm, cam04 55-66 cm mid-range but 1.2-2.6 m near and far. Unknown cameras: 0.5 + 0.01 r.
NOISE = {"cam02_1019": (0.25, 0.018), "cam03_1019": (0.55, 0.008), "cam04_1019": (0.45, 0.025)}
# optional learned noise schedule (learn_kf_noise.py): env KF_NOISE=<kf_noise.pt> makes a small GRU scale sigma_a and the
# measurement sigma per step from what the filter sees; unset = the fixed noises above (bit-identical to before)
KF_NOISE = os.environ.get("KF_NOISE"); _KF_NET = []


def kf_net():
    if not _KF_NET:
        import torch; from learn_kf_noise import NoiseNet, CAMS
        net = NoiseNet(); net.load_state_dict(torch.load(KF_NOISE, map_location="cpu")); net.eval(); _KF_NET.append((torch, net, CAMS))
        print(f"learned KF noise schedule from {KF_NOISE}")
    return _KF_NET[0]


def gate(c1, c2):
    return max(GATE.get(c1, 3.0), GATE.get(c2, 3.0))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix, 1e-6)


def kind(cls):
    return "ped" if cls == "Pedestrian" else "veh"


class UF:
    def __init__(self): self.p = {}
    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def smooth_heading(seq, half=8, sigma=4.0):
    """seq: list of (frame, heading or None) in frame order -> list of smoothed headings (None stays None where no
    neighbour within the window has one). Circular: unit vectors averaged with gaussian weights, zero-phase. A
    heading read off 0.5 s of noisy positions wobbles several degrees per frame and the drawn car snakes; +-1 s
    of samples (half = 8 processed frames) does not."""
    out = []; n = len(seq); w = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    for i in range(n):
        vx = vy = 0.0
        for k in range(max(0, i - half), min(n, i + half + 1)):
            th = seq[k][1]
            if th is not None and seq[k][0] - seq[i][0] <= 4 * (k - i) + 8 and abs(seq[k][0] - seq[i][0]) <= 2 * half + 8:
                vx += w[k - i + half] * np.cos(th); vy += w[k - i + half] * np.sin(th)
        out.append(None if vx == 0 and vy == 0 else float(np.arctan2(vy, vx)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clipdir", required=True); ap.add_argument("--cams", default="cam01,cam02,cam03,cam04"); ap.add_argument("--fit-suffix", default="_fit_pnp")
    ap.add_argument("--no-size-check", dest="size_check", action="store_false", help="do not scale the measurement noise by the box's inconsistency with the fixed vehicle size (on by default: crossing 2 moving 36 -> 20 cm)")
    ap.add_argument("--motion", default="cv", choices=["cv", "ctrv"], help="trajectory model: constant velocity (SORT/ByteTrack) or constant turn rate and velocity (vehicles turning in the junction)")
    ap.add_argument("--max-fill", type=int, default=75, help="a gap of up to this many MISSING raw frames (75 = 5 s at 15 fps) inside one object is filled with the smoother's estimate; 0 = leave gaps")
    ap.add_argument("--max-range", type=float, default=45.0,
                    help="drop detections further than this from their camera: the ground error per pixel grows as range^2 (0.3 m/px at 45 m from a 6 m camera); "
                         "the RTK check gave 2-3 m errors beyond 40 m and a far car's box hopped 3-11 m between frames, so 45 m (was 60)")
    a = ap.parse_args()
    cams = {c: RD.Cam(c, c + a.fit_suffix) for c in a.cams.split(",")}
    tracks = {c: RD.load_tracks(os.path.join(a.clipdir, c, "trajectories.csv")) for c in cams}
    # ground positions and distance to the own camera, per detection
    det = defaultdict(list); dropped = [0]; dropped_edge = [0]; kept_edge = [0]   # frame -> [(cam, id, cls, E, N, dist, box, edge_cut)]
    for c, T in tracks.items():
        for fr, ds in T.items():
            if not ds: continue
            g = cams[c].ground([[d["u"], d["v"]] for d in ds])
            for d, (e, n) in zip(ds, g):
                dist = float(np.hypot(e - cams[c].pos[0], n - cams[c].pos[1])) if cams[c].pos is not None else 20.0
                if dist > a.max_range: dropped[0] += 1; continue
                b = d.get("box")
                # a box cut by the frame edge: the centre-from-box solve is biased (the visible width and centre shift
                # as the cut grows). A vehicle box cut at a SIDE edge but still mostly visible is kept, flagged so its
                # size_ok is forced low below (trusted less by the Kalman update, not erased), so a bus cut in two
                # cameras at once keeps a smoothed-through position instead of a hole. Dropped outright: a cut at
                # the bottom (no foot point), a pedestrian (foot point only, nothing to down-weight), and a sliver
                # narrower than half its height - cam04 tracked a car stopped beside the frame edge as a 17 px box
                # for 36 s (RTK check 2026-11-03): keeping that sliver put the car 2.6 m off and stopped cam03 and
                # cam04 linking; with this rule crossing 2 scores 19 / 72 cm (was 20 / 90), crossing 1 unchanged.
                edge_cut = b is not None and (b[3] >= FRAME_H - 10 or b[0] <= 15 or b[2] >= FRAME_W - 15)   # 15 px
                if edge_cut:
                    if d["cls"] == "Pedestrian" or b[3] >= FRAME_H - 10 or (b[2] - b[0]) < 0.5 * (b[3] - b[1]): dropped_edge[0] += 1; continue
                    kept_edge[0] += 1
                det[fr].append((c, d["id"], d["cls"], float(e), float(n), dist, b, edge_cut))
    print(f"{dropped[0]} detections dropped beyond {a.max_range:.0f} m, {dropped_edge[0]} cut by the frame edge dropped (bottom / pedestrian / sliver), {kept_edge[0]} side-cut vehicle boxes kept down-weighted")
    # LINK positions: the solved vehicle centre wherever the camera track's own heading is known, else the foot
    # point. Two cameras see opposite ends of a bus: foot points 10 m apart (beyond any sane gate - a bus seen by
    # cam02 and cam03 used to stay two objects), centres 1-2 m apart.
    by_track = defaultdict(list)
    for fr, L in det.items():
        for k, d in enumerate(L): by_track[(d[0], d[1])].append((fr, d[3], d[4], k))
    link_pos = {}; track_hdg = {}
    for key, h in by_track.items():
        h.sort(); prev = None
        raw_th = []
        for i, (fr, e, n, k) in enumerate(h):
            th = None
            for span in (4, 8, 16):
                p0, p1 = h[max(i - span, 0)], h[min(i + span, len(h) - 1)]
                de, dn = p1[1] - p0[1], p1[2] - p0[2]; dd = np.hypot(de, dn)
                if dd > 2.5:
                    th = np.arctan2(dn, de)
                    if prev is not None and dd < 5.0 and abs((th - prev + np.pi) % (2 * np.pi) - np.pi) > np.radians(60): th = prev
                    break
            if th is None: th = prev
            prev = th; raw_th.append((fr, th))
        sm_th = smooth_heading(raw_th)
        for i, (fr, e, n, k) in enumerate(h):
            th = sm_th[i]; d = det[fr][k]; pos = (e, n); track_hdg[(fr, key)] = th
            if th is not None and d[2] != "Pedestrian" and d[6] is not None:
                Lv, Wv = RD.SIZE.get(d[2], (4.5, 1.9)); Hv = HEIGHT.get(d[2], 1.5)
                cc = cams[d[0]].centre_from_box((d[6][0] + d[6][2]) / 2, d[6][3], th, Lv, Wv, Hv)
                if cc is not None: pos = (float(cc[0]), float(cc[1]))
            link_pos[(fr, key)] = pos
        first = next((track_hdg[(fr, key)] for fr, _, _, _ in h if track_hdg[(fr, key)] is not None), None)   # first seen stopped: faces the way it leaves
        for fr, _, _, _ in h:
            if track_hdg[(fr, key)] is None: track_hdg[(fr, key)] = first
            else: break
    for fr in det:
        det[fr] = [d + (link_pos.get((fr, (d[0], d[1])), (d[3], d[4])),) for d in det[fr]]
    # per-frame links, counted per track pair
    links = Counter()
    for fr, L in det.items():
        cand = []
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                if kind(L[i][2]) != kind(L[j][2]): continue
                if L[i][0] == L[j][0]:                                       # same camera: only a large vehicle split into two boxes
                    if L[i][2] in BIG and L[j][2] in BIG and L[i][6] and L[j][6] and iou(L[i][6], L[j][6]) > 0.1:
                        links[((L[i][0], L[i][1]), (L[j][0], L[j][1]))] += 1
                    continue
                if L[i][7] or L[j][7]:
                    # one side's box is cut by the frame edge: its position alone is not trusted evidence (that is
                    # why it is down-weighted above) - a real continuing object keeps the SAME heading across
                    # cameras, so require that too before accepting a match on position proximity alone; this is
                    # what stops an edge-cut box from coincidentally landing close to a different, unrelated object
                    hi = track_hdg.get((fr, (L[i][0], L[i][1]))); hj = track_hdg.get((fr, (L[j][0], L[j][1])))
                    if hi is not None and hj is not None and abs((hi - hj + np.pi) % (2 * np.pi) - np.pi) > np.radians(60): continue
                dd = np.hypot(L[i][8][0] - L[j][8][0], L[i][8][1] - L[j][8][1])          # solved centres, see above
                # the gate grows with range: a camera's position error is ~1 m at 20 m and 2-3 m at 50 m (RTK check),
                # so a car 55 m from cam04 and 28 m from cam03 needs ~6 m, two near views ~4 m
                if dd < gate(L[i][2], L[j][2]) * (1 + (L[i][5] + L[j][5]) / 80.0): cand.append((dd, i, j))
        used = set()
        for dd, i, j in sorted(cand):
            if (i, L[j][0]) in used or (j, L[i][0]) in used: continue    # one partner per camera per detection
            used.add((i, L[j][0])); used.add((j, L[i][0]))
            links[((L[i][0], L[i][1]), (L[j][0], L[j][1]))] += 1
    frames_of = defaultdict(set)
    for fr, L in det.items():
        for c, i, *_ in L: frames_of[(c, i)].add(fr)
    uf = UF(); rejected = 0; handovers = 0
    for (p, q), n in links.items():
        both = frames_of[p] & frames_of[q]; common = len(both)
        if n >= MIN_LINK_FRAMES and n >= MIN_LINK_RATIO * common: uf.union(p, q)
        elif n >= MIN_LINK_FRAMES: rejected += 1
        elif n >= 2 and n == common:
            # a short overlap in which the two tracks agree in EVERY frame, one track ending there and the other
            # starting there: a camera handover (a bus leaving cam03's view as cam02 picks it up overlapped by 3 frames
            # and became two objects drawn on top of each other). Fewer frames than a link needs, but unambiguous.
            lo, hi = min(both), max(both); ends = {max(frames_of[p]), max(frames_of[q])}; starts = {min(frames_of[p]), min(frames_of[q])}
            if any(abs(e - hi) <= 4 for e in ends) and any(abs(st - lo) <= 4 for st in starts) and (max(frames_of[p]) - hi <= 4) != (max(frames_of[q]) - hi <= 4):
                uf.union(p, q); handovers += 1
    print(f"{handovers} short-overlap camera handovers merged")
    print(f"{rejected} track pairs close for a few frames only (passing each other), not merged")
    # bridge occlusion gaps: a track that ends, and one of the same kind that starts soon after where the first would be
    tr = defaultdict(list)
    for fr in sorted(det):
        for c, i, cls, e, n, dist, box, ec, *_ in det[fr]: tr[(c, i)].append((fr, e, n, cls, ec))
    # frame numbers are RAW frame indices (15/s, every 2nd processed); a 2026-10-28 bug divided them by the processed
    # rate, so every gap looked 2.2x longer than real and no handover over 2.3 s was ever bridged
    FPS_RAW = 15.0; ends = []; starts = []
    for k, L in tr.items():
        L.sort()
        # exit/entry position, time and velocity come from the last/first NON-edge-cut detection: an edge-cut box's
        # position is already not trusted (down-weighted above), so anchoring a handover prediction on it would just
        # reintroduce that same bias into the cross-camera match
        rel = [x for x in L if not x[4]] or L
        f0, f1 = rel[0][0], rel[-1][0]
        tail = [x for x in rel if x[0] >= f1 - FPS_RAW]                             # velocity VECTOR over the last second
        dt_t = max((tail[-1][0] - tail[0][0]) / FPS_RAW, 0.5)
        ve, vn = ((tail[-1][1] - tail[0][1]) / dt_t, (tail[-1][2] - tail[0][2]) / dt_t) if len(tail) > 1 else (0.0, 0.0)
        head = [x for x in rel if x[0] <= f0 + FPS_RAW]                             # and over the first second, for the constant-acceleration hypothesis below
        dt_h = max((head[-1][0] - head[0][0]) / FPS_RAW, 0.5)
        ve0, vn0 = ((head[-1][1] - head[0][1]) / dt_h, (head[-1][2] - head[0][2]) / dt_h) if len(head) > 1 else (0.0, 0.0)
        ends.append((f1, rel[-1][1], rel[-1][2], kind(L[-1][3]), (ve, vn), k)); starts.append((f0, rel[0][1], rel[0][2], kind(L[0][3]), (ve0, vn0), k))
    all_f = sorted(det); stride = min((b - c for c, b in zip(all_f[:-1], all_f[1:])), default=2)   # raw frames between processed frames
    pairs = []
    for f1, e1, n1, k1, v1, ka in ends:
        for f0, e0, n0, k0, v0, kb in starts:
            if kb == ka or k0 != k1 or uf.find(ka) == uf.find(kb): continue
            dt = (f0 - f1) / FPS_RAW
            if not (0 <= (f0 - f1 - stride) / FPS_RAW <= GAP_S): continue        # the MISSING time is limited to GAP_S (a 5 s hole has 5 s + one stride between its two edge measurements); dt = 0: one camera's track ends in the frame the next camera's begins (a bus turning out of cam03 into cam02)
            # two kinematic hypotheses for where the lost track is now: constant velocity (the exit velocity carried on) and
            # constant acceleration between the exit and entry velocities (displacement = mean velocity x dt), which is what a
            # vehicle pulling away from a stop or braking into one does; the better fitting one is scored against the gate
            dd_cv = np.hypot(e0 - (e1 + v1[0] * dt), n0 - (n1 + v1[1] * dt))
            dd_ca = np.hypot(e0 - (e1 + 0.5 * (v1[0] + v0[0]) * dt), n0 - (n1 + 0.5 * (v1[1] + v0[1]) * dt))
            dd = min(dd_cv, dd_ca)
            if dd <= GAP_BASE_M + 0.5 * dt: pairs.append((dd, ka, kb, dt, np.hypot(e0 - e1, n0 - n1), "cv" if dd_cv <= dd_ca else "ca"))
    used_end = set(); used_start = set(); bridged = 0; bridged_ca = 0                 # separate roles: a track can end into one and be the start of another
    for dd, ka, kb, dt, jump, hyp in sorted(pairs):
        if ka in used_end or kb in used_start: continue
        uf.union(ka, kb); used_end.add(ka); used_start.add(kb); bridged += 1; bridged_ca += hyp == "ca"
        print(f"  bridged {ka[0]}:{ka[1]} -> {kb[0]}:{kb[1]}  gap {dt:.1f} s, moved {jump:.1f} m, {dd:.1f} m off the {hyp} prediction")
    print(f"{bridged} occlusion gaps bridged ({bridged_ca} only under the constant-acceleration hypothesis)")
    # global ids
    keys = sorted({(c, d["id"]) for c, T in tracks.items() for ds in T.values() for d in ds})
    roots = {k: uf.find(k) for k in keys}; gid = {r: i + 1 for i, r in enumerate(sorted(set(roots.values())))}
    merged = Counter(roots.values()); print(f"{len(keys)} camera tracks -> {len(gid)} objects; {sum(1 for v in merged.values() if v > 1)} objects seen by more than one camera")
    # class per object: majority over all its detections
    votes = defaultdict(Counter)
    for fr, L in det.items():
        for c, i, cls, e, n, dist, box, *_ in L: votes[gid[roots[(c, i)]]][cls] += 1
    gcls = {g: v.most_common(1)[0][0] for g, v in votes.items()}
    # per frame, per object: all cameras that see it, averaged with weight 1/dist^2 (near views count more);
    # pe, pn = weighted unit direction camera -> object, for pushing a vehicle box away from the cameras
    by_frame = {}
    for fr in sorted(det):
        by = defaultdict(list)
        for c, i, cls, e, n, dist, box, *_ in det[fr]: by[gid[roots[(c, i)]]].append((c, i, cls, e, n, dist))
        by_frame[fr] = by
    # vehicle heading per object and frame, from the raw fused track (same rule as the renderer: believed only
    # after 2.5 m of travel over the window; a >60 deg flip is jitter unless the vehicle clearly moved)
    raw = defaultdict(list)
    for fr, by in by_frame.items():
        for g, L in by.items():
            w = np.array([1.0 / max(d[5], 5.0) ** 2 for d in L]); w /= w.sum()
            raw[g].append((fr, float(sum(wi * d[3] for wi, d in zip(w, L))), float(sum(wi * d[4] for wi, d in zip(w, L)))))
    hdg = {}
    for g, h in raw.items():
        h.sort(); prev = None
        for i, (fr, _, _) in enumerate(h):
            th = None
            for span in (4, 8, 16):
                p0, p1 = h[max(i - span, 0)], h[min(i + span, len(h) - 1)]
                de, dn = p1[1] - p0[1], p1[2] - p0[2]; d = np.hypot(de, dn)
                if d > 2.5:
                    th = np.arctan2(dn, de)
                    if prev is not None and d < 5.0 and abs((th - prev + np.pi) % (2 * np.pi) - np.pi) > np.radians(60): th = prev
                    break
            if th is None: th = prev                                                   # a stopped vehicle keeps its last heading
            hdg[(g, fr)] = th; prev = th
        # a vehicle first seen waiting at the light faces the way it will drive off: back-fill its first heading
        first = next((hdg[(g, fr)] for fr, _, _ in h if hdg[(g, fr)] is not None), None)
        for fr, _, _ in h:
            if hdg[(g, fr)] is None: hdg[(g, fr)] = first
            else: break
        for (fr, _, _), th in zip(h, smooth_heading([(fr, hdg[(g, fr)]) for fr, _, _ in h])): hdg[(g, fr)] = th
    # vehicle centre: the foot point (bottom centre of the box) is the vehicle's NEAR END as seen from that camera, so
    # each camera's point is pushed toward the far end BEFORE averaging - two cameras see different ends of one car.
    # The push is along the vehicle's own axes, not along the camera ray: checked against the LiDAR car's RTK track
    # (validate_rtk.py), the raw foot point of a car approaching a camera 20 deg off its axis lay ON the car's axis,
    # 2.1 m ahead of the roof antenna, with no lateral offset; pushing along the ray put it 0.6 m off to the side.
    # With c = cos, s = sin of the angle between the view direction and the heading: half a length along the axis
    # weighted c|c|, half a width sideways weighted s|s| (end-on: a full half length; side-on: a full half width).
    # Heading unknown (never moved): (L+W)/4 along the ray.
    # With the heading known and the camera model available, the centre is solved exactly from the observed box
    # (Cam.centre_from_box: the L x W x H box whose projection has that bottom-centre); the closed formula above is
    # the fallback and the starting point. An oblique view used to be 1.1 m short with the formula alone.
    det_box = {(c, i, fr): box for fr, L_ in det.items() for c, i, cls, e, n, dist, box, *_ in L_}
    det_edge = {(c, i, fr): ec for fr, L_ in det.items() for c, i, cls, e, n, dist, box, ec, *_ in L_}
    rows = []
    for fr, by in by_frame.items():
        for g, L in by.items():
            cls_g = gcls[g]; Lv, Wv = RD.SIZE.get(cls_g, (4.5, 1.9)); Hv = HEIGHT.get(cls_g, 1.5)
            # heading from the NEAREST camera's own track (smooth); the fused raw track mixes cameras whose positions
            # differ by a metre or two and gave a car waiting at the light a heading 40 deg off
            near0 = min(L, key=lambda d: d[5]); th = track_hdg.get((fr, (near0[0], near0[1])))
            if th is None: th = hdg.get((g, fr))
            w = np.array([1.0 / max(d[5], 5.0) ** 2 for d in L]); w /= w.sum()
            E = sum(wi * d[3] for wi, d in zip(w, L)); N = sum(wi * d[4] for wi, d in zip(w, L))
            pv = np.zeros(2); Ec, Nc = 0.0, 0.0; size_ok = []
            for wi, d in zip(w, L):
                cp = cams[d[0]].pos; v = np.zeros(2); push_e, push_n = 0.0, 0.0; solved = None; pre_len = len(size_ok)
                if cp is not None:
                    v = np.array([d[3] - cp[0], d[4] - cp[1]]); v = v / max(np.linalg.norm(v), 1e-6); pv += wi * v
                if cls_g == "Pedestrian" or cp is None: pass
                elif th is None: push_e, push_n = v * ((Lv + Wv) / 4)
                else:
                    box = det_box.get((d[0], d[1], fr))
                    if box is not None:
                        solved = cams[d[0]].centre_from_box((box[0] + box[2]) / 2, box[3], th, Lv, Wv, Hv)
                        if solved is not None: size_ok.append(cams[d[0]].box_consistency(solved, th, Lv, Wv, Hv, box))
                    if solved is None:
                        ax, lat = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
                        c, s = float(v @ ax), float(v @ lat)
                        push_e, push_n = ax * (Lv / 2 * c * abs(c)) + lat * (Wv / 2 * s * abs(s))
                if det_edge.get((d[0], d[1], fr)):             # edge-cut: the box geometry assumption is broken, trust this contribution less
                    if len(size_ok) > pre_len: size_ok[-1] = min(size_ok[-1], 0.2)
                    else: size_ok.append(0.2)
                if solved is not None: Ec += wi * solved[0]; Nc += wi * solved[1]
                else: Ec += wi * (d[3] + push_e); Nc += wi * (d[4] + push_n)
            pn_ = np.linalg.norm(pv); pv = pv / pn_ if pn_ > 1e-6 else pv
            near = min(L, key=lambda d: d[5])
            rows.append(dict(frame=fr, gid=g, cls=cls_g, E=round(E, 3), N=round(N, 3), Ec=round(Ec, 3), Nc=round(Nc, 3), size_ok=round(float(np.mean(size_ok)), 2) if size_ok else 1.0,
                             hdg=("" if th is None else round(float(np.degrees(th)), 1)), pe=round(pv[0], 3), pn=round(pv[1], 3),
                             cam=near[0], n_cams=len(L), src="|".join(f"{d[0]}:{d[1]}" for d in L), dist_m=round(near[5], 1), filled=0))
    # second-level merge: two VEHICLE objects that sit on top of each other for most of their common frames are one
    # vehicle (a bus whose front survived as its own track, two views that never linked because of the one-partner
    # rule). Criterion: coexist >= 6 frames and the centres are within 0.6 L of the larger class in >= 70 % of them.
    by_obj = defaultdict(dict)
    for k, r in enumerate(rows):
        if r["cls"] != "Pedestrian": by_obj[r["gid"]][r["frame"]] = k
    objs = sorted(by_obj); dup_merged = 0; alias = {}
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a_, b_ = objs[i], objs[j]
            if alias.get(a_) or alias.get(b_): continue
            common = sorted(set(by_obj[a_]) & set(by_obj[b_]))
            if len(common) < 6: continue
            La = RD.SIZE.get(rows[by_obj[a_][common[0]]]["cls"], (4.5, 1.9))[0]; Lb = RD.SIZE.get(rows[by_obj[b_][common[0]]]["cls"], (4.5, 1.9))[0]
            tol = 0.6 * max(La, Lb)
            close = [np.hypot(rows[by_obj[a_][f]]["Ec"] - rows[by_obj[b_][f]]["Ec"], rows[by_obj[a_][f]]["Nc"] - rows[by_obj[b_][f]]["Nc"]) < tol for f in common]
            if np.mean(close) >= 0.7:
                big, small = (a_, b_) if len(by_obj[a_]) >= len(by_obj[b_]) else (b_, a_); alias[small] = big; dup_merged += 1
    if alias:
        keep = []
        for k, r in enumerate(rows):
            g = r["gid"]
            if g in alias:
                tgt = alias[g]
                if r["frame"] in by_obj[tgt]:                                           # both have this frame: keep the target's row
                    continue
                r["gid"] = tgt
            keep.append(r)
        rows = keep
    print(f"duplicate check: {dup_merged} vehicle objects merged into the object they overlapped")
    # ---- trajectory estimation: Kalman filter + RTS smoother per object (replaces the ad-hoc blend / clamp / gaussian) ----
    # Constant-velocity model (SORT / ByteTrack use the same for online tracking), chi-square innovation gating
    # (Bar-Shalom) so a measurement hop - a box cut by the frame edge, a far camera joining, a wrong link - is not
    # followed, and the Rauch-Tung-Striebel backward pass because this is offline: zero lag. Four consecutive gated
    # measurements that the model cannot absorb are another road user: the object is cut there into a new id.
    # Heading = direction of the smoothed velocity while moving (> 0.5 m/s), held while stopped, back-filled at the start.
    seq = defaultdict(list)
    for k, r in enumerate(rows): seq[r["gid"]].append((r["frame"], k))
    # A road user the cameras stop positioning for a moment (a box cut by the frame edge in both views at once, a miss)
    # keeps its id but leaves a hole. The hole is filled with placeholder rows that the filter predicts through without
    # a measurement; the RTS pass then fixes them from both ends. Rows carry filled=1 so a validation can leave them out.
    filled_n = 0; fill_len = {}                                                       # row index -> hole length in s, for the filter's weak measurements
    if a.max_fill > 0:
        for g in list(seq):
            sq = sorted(seq[g])
            for (f0, k0), (f1, k1) in zip(sq[:-1], sq[1:]):
                if stride < f1 - f0 <= a.max_fill + stride:                       # the MISSING frames are limited to max_fill (a 5 s hole has 75 missing frames and 77 between its edges)
                    r0, r1 = rows[k0], rows[k1]; T = (f1 - f0) / 15.0
                    # Hermite fill: the cubic through both edge positions with both edge velocities (velocity over up to 4 measured
                    # rows on each side). On holes cut into the LiDAR car's track it stays at the no-hole floor (2/3/5 s: 18/19/31 cm
                    # vs 52/92/243 cm linear), because a hole that covers a turn or an acceleration is not a straight line at constant
                    # speed. The filled positions enter the filter below as weak measurements, so the real measurement after the hole
                    # is no longer 10-16 m from a straight-ahead prediction and gated (which cut the object in two).
                    bef = [rows[k] for f, k in sq if f0 - 4 * stride <= f < f0]; aft = [rows[k] for f, k in sq if f1 < f <= f1 + 4 * stride]
                    def vel(ra, rb):
                        dt_ = (int(rb["frame"]) - int(ra["frame"])) / 15.0
                        v = {key: (float(rb[key]) - float(ra[key])) / dt_ for key in ("E", "N", "Ec", "Nc")} if dt_ > 0 else None
                        return None if v is None or np.hypot(v["Ec"], v["Nc"]) > 25.0 else v
                    v0 = vel(bef[0], r0) if bef else None; v1 = vel(r1, aft[-1]) if aft else None
                    z = {key: 0.0 for key in ("E", "N", "Ec", "Nc")}; v0 = v0 or z; v1 = v1 or z
                    for f in range(f0 + stride, f1, stride):
                        t = (f - f0) / (f1 - f0); nr = dict(r0)
                        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
                        for key in ("E", "N", "Ec", "Nc"): nr[key] = round(h00 * float(r0[key]) + h10 * v0[key] * T + h01 * float(r1[key]) + h11 * v1[key] * T, 3)
                        nr.update(frame=f, n_cams=0, filled=1, size_ok=1.0); rows.append(nr); fill_len[len(rows) - 1] = T; seq[g].append((f, len(rows) - 1)); filled_n += 1
        print(f"gap filling: {filled_n} placeholder rows inside objects (gaps up to {a.max_fill} frames)")
    # keep the measurement the filter is about to consume (Em, Nm; blank on placeholder rows): Ec/Nc below become the smoothed
    # estimate, and the learned-filter work (learn_kf_noise.py) needs the raw multi-camera measurement sequence to train on
    for r in rows: r["Em"], r["Nm"] = ("", "") if r.get("filled") == 1 else (r["Ec"], r["Nc"])
    next_gid = max(seq) + 1; cuts = 0; gated_total = 0; cut_list = []; manoeuvres = 0
    NS = 4 if a.motion == "cv" else 5
    H = np.zeros((2, NS)); H[0, 0] = H[1, 1] = 1.0
    def ctrv_predict(x, dt, sig_a, sig_w):
        """CTRV (Schubert et al. 2008): state x, y, v, psi, omega -> predicted state, Jacobian, process noise"""
        px, py, v, psi, w = x; F = np.eye(5)
        if abs(w) > 1e-3:
            s1, c1, s2, c2 = np.sin(psi), np.cos(psi), np.sin(psi + w * dt), np.cos(psi + w * dt)
            nx = px + v / w * (s2 - s1); ny = py + v / w * (-c2 + c1)
            F[0, 2] = (s2 - s1) / w; F[0, 3] = v / w * (c2 - c1); F[0, 4] = v * dt * c2 / w - v * (s2 - s1) / w ** 2
            F[1, 2] = (-c2 + c1) / w; F[1, 3] = v / w * (s2 - s1); F[1, 4] = v * dt * s2 / w - v * (-c2 + c1) / w ** 2
        else:
            nx = px + v * np.cos(psi) * dt; ny = py + v * np.sin(psi) * dt
            F[0, 2] = np.cos(psi) * dt; F[0, 3] = -v * np.sin(psi) * dt; F[1, 2] = np.sin(psi) * dt; F[1, 3] = v * np.cos(psi) * dt
        F[3, 4] = dt
        Q = np.zeros((5, 5)); Q[2, 2] = (sig_a * dt) ** 2; Q[4, 4] = (sig_w * dt) ** 2; Q[3, 3] = (sig_w * dt ** 2 / 2) ** 2
        Q[0, 0] = Q[1, 1] = (0.5 * sig_a * dt ** 2) ** 2
        return np.array([nx, ny, v, psi + w * dt, w]), F, Q
    def run_filter(ks, sig_a, sig_m, reseat=4.0, a_hole=None):
        """forward KF (cv) or EKF (ctrv) with gating over rows ks; returns (xf, Pf, xp, Pp, Fs, used), a cut, or
        ("manoeuvre", ...) when the first measurements after a filled hole reject the constant-velocity fill but are
        kinematically reachable at a_hole - the caller then re-runs with a_hole set (the hole's noise inflated)"""
        n = len(ks); xf = np.zeros((n, NS)); Pf = np.zeros((n, NS, NS)); xp = np.zeros((n, NS)); Pp = np.zeros((n, NS, NS)); Fs = np.zeros((n, NS, NS)); used = np.ones(n, bool)
        z0 = np.array([rows[ks[0]]["Ec"], rows[ks[0]]["Nc"]])
        if a.motion == "cv": x = np.r_[z0, 0.0, 0.0]; P = np.diag([sig_m ** 2, sig_m ** 2, 25.0, 25.0])
        else:
            # initial speed and heading from the first clear displacement (>= 1.5 m within the first 2 s), else at rest
            x = np.r_[z0, 0.0, 0.0, 0.0]; P = np.diag([sig_m ** 2, sig_m ** 2, 25.0, (2 * np.pi) ** 2, 1.0])
            for j in range(1, min(n, 16)):
                dz = np.array([rows[ks[j]]["Ec"], rows[ks[j]]["Nc"]]) - z0; ddt = (rows[ks[j]]["frame"] - rows[ks[0]]["frame"]) / 15.0
                if np.hypot(*dz) >= 1.5 and ddt > 0: x[2] = np.hypot(*dz) / ddt; x[3] = np.arctan2(dz[1], dz[0]); P[3, 3] = 0.3 ** 2; break
        gated_run = []
        net = kf_net() if (KF_NOISE and a.motion == "cv") else None; s_q = s_r = 0.0
        if net: torch, model, CAMS = net; h = torch.zeros(1, model.hidden); prev_innov = np.zeros(2); prev_corr = 0.0
        for i, k in enumerate(ks):
            if net:                                                                   # the same 12 features as in training, then two log-scales
                r_ = rows[k]; f = np.zeros(12, np.float32); f[:3] = [r_.get("cam", "") == c for c in CAMS]; f[3] = min(int(r_.get("n_cams", 1) or 1), 3) / 3
                f[4] = float(r_.get("size_ok", 1.0)); f[5] = float(r_.get("dist_m", 20.0)) / 50.0; f[6] = r_["cls"] == "Pedestrian"
                f[7:9] = np.clip(prev_innov / 2.0, -5, 5); f[9] = min(prev_corr, 5.0); f[10] = r_.get("filled") != 1; f[11] = i == 0
                with torch.no_grad(): s, h = model(torch.from_numpy(f)[None], h)
                s_q, s_r = float(s[0, 0]), float(s[0, 1]); prev_innov = np.zeros(2); prev_corr = 0.0
            if i == 0: xp[0], Pp[0], Fs[0] = x, P, np.eye(NS)
            else:
                dt = (rows[k]["frame"] - rows[ks[i - 1]]["frame"]) / 15.0
                if a.motion == "cv":
                    F = np.eye(4); F[0, 2] = F[1, 3] = dt; q = (sig_a * np.exp(s_q)) ** 2; Q = np.zeros((4, 4))
                    if a_hole is not None and (rows[k].get("filled") == 1 or rows[ks[i - 1]].get("filled") == 1):
                        # manoeuvre model for an unobserved stretch: the road user may accelerate, brake or turn at up to a_hole
                        # the whole time, so the uncertainty grows as continuous white-noise acceleration (position variance
                        # a_hole^2 T^3 / 3: 6 m after 3 s) instead of the per-step noise tuned for observed frames (2.3 m).
                        # Only used when the plain fill was rejected: with it always on, straight holes got worse (crossing 1
                        # 3 s holes 67 -> 140 cm) because the far, noisy edge measurements then outweigh the motion model.
                        qc = a_hole ** 2
                        for a_, v_ in ((0, 2), (1, 3)): Q[a_, a_] = qc * dt ** 3 / 3; Q[a_, v_] = Q[v_, a_] = qc * dt ** 2 / 2; Q[v_, v_] = qc * dt
                    else:
                        for a_, v_ in ((0, 2), (1, 3)): Q[a_, a_] = q * dt ** 4 / 4; Q[a_, v_] = Q[v_, a_] = q * dt ** 3 / 2; Q[v_, v_] = q * dt ** 2
                    x = F @ x
                else: x, F, Q = ctrv_predict(x, dt, sig_a, 0.5)
                P = F @ P @ F.T + Q; xp[i], Pp[i], Fs[i] = x, P, F
            z = np.array([rows[k]["Ec"], rows[k]["Nc"]]); rng = float(rows[k].get("dist_m", 20.0))
            if rows[k].get("filled") == 1: xf[i], Pf[i] = x, P; continue            # no measurement here: the prediction stands
            a_n, b_n = NOISE.get(rows[k].get("cam", ""), (0.5, 0.01)); sig = (a_n + b_n * rng) * (0.8 if rows[k]["cls"] == "Pedestrian" else 1.0)
            if int(rows[k].get("n_cams", 1)) > 1: sig *= 0.8                          # two views averaged
            if a.size_check: sig *= 1.0 + 3.0 * (1.0 - float(rows[k].get("size_ok", 1.0)))   # a box inconsistent with the fixed vehicle size is trusted less
            R = np.eye(2) * sig ** 2
            y = z - H @ x; S = H @ P @ H.T + R; d2 = float(y @ np.linalg.solve(S, y))          # the outlier gate uses the fixed sigma, so the learned scale changes the weighting, not what is gated
            if net: R = np.eye(2) * (sig * np.exp(s_r)) ** 2; S = H @ P @ H.T + R
            if i > 0 and d2 > 9.21:                                                   # chi-square 2 dof, 99 %
                used[i] = False; gated_run.append(i)
                if os.environ.get("FUSE_DEBUG"): print(f"    gated row {i} frame {rows[k]['frame']} d2 {d2:.1f} |y| {float(np.hypot(*y)):.1f} m sigma_pred {float(np.sqrt(P[0, 0])):.2f} m sig_meas {sig:.2f} filled_before {int(rows[ks[i - 1]].get('filled', 0) == 1)}")
                if len(gated_run) >= 4 and gated_run[-1] - gated_run[0] == 3:
                    # four in a row the model will not absorb: a camera handover with a 1-3 m step (same road user:
                    # re-seat the position, keep the velocity, no new id) or another road user (> 4 m: cut)
                    off = float(np.hypot(*y))
                    if off < reseat:                                                  # 4 m, or 0.6 of a bus's length: two cameras see its two ends
                        x = np.r_[z, x[2:]]; P[:2, :2] = R * 4; gated_run = []; used[i] = True
                    else:
                        j0 = gated_run[0]; T_hole = fill_len.get(ks[j0 - 1]) if j0 > 0 else None
                        if T_hole is not None:
                            T_eff = T_hole + (rows[k]["frame"] - rows[ks[j0]]["frame"]) / 15.0      # the hole plus the rows gated since it ended
                            if a_hole is None and off <= 0.5 * A_HOLE * T_eff ** 2 + 3.0: return ("manoeuvre", j0, off)   # reachable at A_HOLE: try the manoeuvre model before cutting
                        return ("cut", gated_run[0], off)
            else:
                gated_run = []; K = P @ H.T @ np.linalg.inv(S); x = x + K @ y; P = (np.eye(NS) - K @ H) @ P
                if net: prev_innov = y.copy(); prev_corr = float(np.hypot(*(K @ y)[:2]))
            xf[i], Pf[i] = x, P
        return (xf, Pf, xp, Pp, Fs, used)
    for g in list(seq):
        ks = [k for _, k in sorted(seq[g])]; cls_g = rows[ks[0]]["cls"]
        sig_a, sig_m = (1.0, 0.4) if cls_g == "Pedestrian" else (2.0, 0.5); reseat = max(4.0, 0.6 * RD.SIZE.get(cls_g, (4.5, 1.9))[0])
        A_HOLE = 1.0 if cls_g == "Pedestrian" else 2.5                                 # m/s^2 bound on what happens inside a hole
        queue = [(g, ks)]                                                                # (gid, rows): a cut puts both parts back on the queue, so a part that
        while queue:                                                                     # itself needs the manoeuvre model or another cut is handled the same way
            cur, ks = queue.pop(0)
            if len(ks) < 3: continue
            res = run_filter(ks, sig_a, sig_m, reseat)
            if isinstance(res[0], str) and res[0] == "manoeuvre":
                manoeuvres += 1; res = run_filter(ks, sig_a, sig_m, reseat, a_hole=A_HOLE)
            if isinstance(res[0], str):
                c = res[1]; cuts += 1; cut_list.append((cur, cls_g, len(ks), c, round(res[2], 1)))
                head, tail = ks[:c], ks[c:]; next_gid += 1
                for k in tail: rows[k]["gid"] = next_gid
                queue += [(cur, head), (next_gid, tail)]; continue
            ks_done = ks
            xf, Pf, xp, Pp, Fs, used = res; n = len(ks_done); gated_total += int((~used).sum())
            xs = xf.copy()
            for i in range(n - 2, -1, -1):                                             # RTS backward pass
                C = Pf[i] @ Fs[i + 1].T @ np.linalg.pinv(Pp[i + 1]); xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
            prev_h = None; held = None; hs = []
            for i, k in enumerate(ks_done):
                rows[k]["Ec"], rows[k]["Nc"] = round(float(xs[i, 0]), 3), round(float(xs[i, 1]), 3)
                if a.motion == "cv": sp = float(np.hypot(xs[i, 2], xs[i, 3])); hv = float(np.arctan2(xs[i, 3], xs[i, 2]))
                else: sp = abs(float(xs[i, 2])); hv = float(xs[i, 3] + (np.pi if xs[i, 2] < 0 else 0.0))
                # while clearly moving the heading is the velocity direction and is remembered; below 2 m/s (braking,
                # creeping, stopped) the remembered one is held: the deceleration tail's velocity is small and sideways
                # noise turned a car waiting at the light 22 deg across its lane
                if sp > 2.0: held = hv; h = hv
                elif held is not None: h = held
                elif sp > 0.5: h = hv
                else: h = prev_h
                hs.append(h); prev_h = h
            first = next((h for h in hs if h is not None), None)
            for i, k in enumerate(ks_done):
                h = hs[i] if hs[i] is not None else first
                if cls_g != "Pedestrian": rows[k]["hdg"] = "" if h is None else round(float(np.degrees(h)), 1)
    print(f"trajectory estimation ({a.motion}): Kalman + RTS per object, {gated_total} measurements gated as outliers, {manoeuvres} holes re-filled with the manoeuvre model, {cuts} objects cut at persistent disagreement"
          + ("" if not cut_list else ": " + ", ".join(f"obj {g} ({c}, {n} rows, at row {i}, {o} m)" for g, c, n, i, o in cut_list)))
    # written only now: the duplicate merge and the trajectory estimator above edit rows in place (a 2026-10-28
    # bug: the file used to be written before them, so none of it reached fused.csv)
    out = os.path.join(a.clipdir, "fused.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    multi = [(g, [k for k, r in roots.items() if gid[r] == g]) for g, n in merged.items() if n > 1]
    for g, ks in sorted((gid[r], ks) for r, ks in [(r, [k for k in keys if roots[k] == r]) for r in set(roots.values())] if len(ks) > 1):
        print(f"  object {g} ({gcls[g]}): " + ", ".join(f"{c}:{i}" for c, i in ks))
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
