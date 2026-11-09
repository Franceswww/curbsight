"""
render_fused.py - the four camera views and the fused objects on the aerial photo.

    python render_fused.py --clipdir <dir with camNN_*.mp4, camNN/trajectories.csv, fused.csv>

Same drawing rules as render_demo.py (parked vehicles hidden, heading from the track, box pushed
half a width away from the camera that saw it, bus 10.5 m), but one mark per fused object.
Outputs <clipdir>/fused_demo/demo.mp4 and still_*.png
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np

import fit_cam_H as F
import export_tracks as ET
import render_demo as RD

RES = F.RES


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clipdir", required=True); ap.add_argument("--cams", default="cam01,cam02,cam03,cam04"); ap.add_argument("--fit-suffix", default="_fit_pnp")
    ap.add_argument("--fps-out", type=float, default=None, help="default = video fps / 2 (every 2nd frame was processed), i.e. real time")
    ap.add_argument("--trail", type=int, default=12); ap.add_argument("--zoom", default=None, help="E0,N0,E1,N1 metres; default = the whole base image")
    ap.add_argument("--start", default="13:30:00", help="wall-clock time of the clip's first frame")
    ap.add_argument("--site", default="Sanford Dr at Cedar St"); ap.add_argument("--date", default="2026-03-23")
    ap.add_argument("--base", default="wide", choices=["wide", "corner"], help="wide = out_refined/aerial_wide.png (make_wide_base.py), corner = the 50 x 38 m corner grid")
    ap.add_argument("--truth", default=None, help="rtk_enu_lidar.npz: draw the LiDAR car's true outline (magenta) on the map")
    ap.add_argument("--truth-objects", default=None, help="lidar_objects.csv: draw the road users the LiDAR saw (white = pedestrian, grey = vehicle cluster)")
    ap.add_argument("--rec-start", type=float, default=None, help="recording offset of the clip's first frame, s (needed with --truth)")
    ap.add_argument("--centre-offset", type=float, default=1.46)
    ap.add_argument("--centre-clamp", type=float, default=1.0,
                    help="same as export_tracks.py: reject a solved centre further than this many "
                         "body-reaches from its own foot point; 0 disables the check")
    ap.add_argument("--bay-dirs", action="store_true",
                    help="a vehicle that never drives takes its direction from the parked cars around it in the aerial")
    ap.add_argument("--lot-file", default=None, help="a parking area drawn by hand (see "
                    "lot_sanford_cedar.json): bays and a dead end where nothing is of interest, and "
                    "the aisle between them, where a vehicle is traffic while it is moving")
    a = ap.parse_args()
    truth = None
    if a.truth:
        R_ = np.load(a.truth); t_wall = float(R_["bag0"]) + R_["t"]; rec0_ = float(R_["rec0"]); E_, N_ = R_["E"].copy(), R_["N"].copy()
        vE_, vN_ = np.gradient(E_, R_["t"]), np.gradient(N_, R_["t"]); sp_ = np.hypot(vE_, vN_); hd_ = np.where(sp_ > 0.5, np.arctan2(vN_, vE_), np.nan)
        idx_ = np.where(~np.isnan(hd_))[0]; pos_ = np.clip(np.searchsorted(idx_, np.arange(len(hd_)), side="right") - 1, 0, len(idx_) - 1); hd_ = hd_[idx_[pos_]]
        E_ += a.centre_offset * np.cos(hd_); N_ += a.centre_offset * np.sin(hd_); hdu_ = np.unwrap(hd_)
        truth = dict(wall=t_wall, rec0=rec0_, E=E_, N=N_, hd=hdu_)
    tobj = defaultdict(list)
    if a.truth_objects:
        for r in csv.DictReader(open(a.truth_objects)):
            if r["size_class"] in ("Pedestrian", "Vehicle", "Bus/Truck"): tobj[int(round(float(r["t_rec"]) * 2) / 2 * 100)].append((float(r["E"]), float(r["N"]), r["size_class"]))
    out = os.path.join(a.clipdir, "fused_demo"); os.makedirs(out, exist_ok=True)
    cam_names = a.cams.split(","); cams = {c: RD.Cam(c, c + a.fit_suffix) for c in cam_names}
    tracks = {c: RD.load_tracks(os.path.join(a.clipdir, c, "trajectories.csv")) for c in cams}
    caps = {c: cv2.VideoCapture([os.path.join(a.clipdir, f) for f in os.listdir(a.clipdir) if f.startswith(c) and f.endswith(".mp4")][0]) for c in cams}
    fps = caps[cam_names[0]].get(cv2.CAP_PROP_FPS); fps_out = a.fps_out or fps / 2
    fused = defaultdict(list); hist = defaultdict(list); push = {}; gid_of = {}; centred = False; hdg_csv = {}; clamped = 0
    with open(os.path.join(a.clipdir, "fused.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # Ec/Nc = vehicle centre (fuse_tracks.py pushes each camera's foot point before averaging); older
            # fused.csv files carry the foot point only and the push is done here at draw time
            centred = "Ec" in r
            fr, g = int(r["frame"]), int(r["gid"])
            # the same guard export_tracks.py applies: a smoothed centre that has drifted further from
            # its own foot point than a body is wide has lost its object, so the measurement wins
            e, n, fell_back = ET.centre_or_foot(r, r["cls"], a.centre_clamp) if centred else (float(r["E"]), float(r["N"]), False)
            clamped += fell_back
            fused[fr].append(dict(gid=g, cls=r["cls"], e=e, n=n, cam=r["cam"], n_cams=int(r["n_cams"]))); hist[g].append((fr, e, n))
            for src in r.get("src", "").split("|"):                      # camera track -> global id, for the labels in the views
                if ":" in src: cam_, tid_ = src.rsplit(":", 1); gid_of[(cam_, int(tid_))] = g
            push[(g, fr)] = (0.0, 0.0) if (centred and not fell_back) else (float(r.get("pe", 0)), float(r.get("pn", 0)))
            if r.get("hdg", "") != "": hdg_csv[(g, fr)] = np.radians(float(r["hdg"]))   # smoothed in fuse_tracks; used instead of re-estimating
    for g in hist: hist[g].sort()
    # smoothing: exponential average over consecutive processed frames (new sample weight 0.35), reset on a jump
    sm = {}
    for g, h in hist.items():
        prev = None; out_h = []
        for fr, e, n in h:
            if prev is not None and fr - prev[0] <= 4 and np.hypot(e - prev[1], n - prev[2]) < 3: e, n = 0.35 * e + 0.65 * prev[1], 0.35 * n + 0.65 * prev[2]
            prev = (fr, e, n); out_h.append(prev); sm[(g, fr)] = (e, n)
        hist[g] = out_h
    # an object hidden for up to 5 s (behind a bus, say) is still there: interpolate its position and draw it hollow
    inferred = set(); fps_proc = fps / 2
    for g, h in list(hist.items()):
        cls_g = next(d["cls"] for fr in fused for d in fused[fr] if d["gid"] == g)
        extra = []
        for (fa, ea, na), (fb, eb, nb) in zip(h[:-1], h[1:]):
            if 2 < fb - fa <= 5 * fps_proc:
                for fr in range(fa + 2, fb, 2):
                    t = (fr - fa) / (fb - fa); e, n = ea + t * (eb - ea), na + t * (nb - na)
                    extra.append((fr, e, n)); sm[(g, fr)] = (e, n); inferred.add((g, fr))
                    fused[fr].append(dict(gid=g, cls=cls_g, e=e, n=n, cam=None, n_cams=0)); push[(g, fr)] = (0.0, 0.0)
        if extra: hist[g] = sorted(h + extra)
    parked = {g: (max(np.hypot(e - h[0][1], n - h[0][2]) for _, e, n in h) < 1.5) for g, h in hist.items()}

    # What happens inside a parking area is not traffic at the junction: only the drive in and the
    # drive out are. The bays and the dead end at the back are drawn by hand on the aerial; what is
    # left between them is the aisle, which is a road like any other.
    lot = json.load(open(a.lot_file)) if a.lot_file else None

    def zone(e, n):
        if lot is None:
            return "aisle"
        R = lot["rect"]
        if not (R[0] <= e <= R[2] and R[1] <= n <= R[3]):
            return "aisle"                                  # outside the lot: ordinary road
        if e < lot["ignore_west_of"]:
            return "ignore"
        for b in lot["bays"]:
            lo, hi = b["e_range"]
            if not lo <= e <= hi:
                continue                                    # the row does not reach this far east
            (e1, n1_), (e2, n2_) = b["line"]
            edge = n1_ + (n2_ - n1_) * (e - e1) / (e2 - e1)
            back = edge + b["depth"] if b["side"] == "north" else edge - b["depth"]
            if (edge <= n <= back) if b["side"] == "north" else (back <= n <= edge):
                return "bay"
        return "aisle"

    # Stretches where a vehicle has come to rest: its solved centre still wanders a metre or two
    # while it stands, because the cameras see more or less of it from frame to frame. Each such
    # stretch is pinned to one position so the box does not drift, and inside the lot it is where
    # the car has parked, so it stops being drawn there.
    REST_R, REST_S = 1.5, 4.0
    rest = {}
    parked_from = {}        # gid -> the frame it parked inside the lot and never drove again
    for g, h in hist.items():
        i = 0
        while i < len(h):
            j = i
            while j + 1 < len(h) and np.hypot(h[j + 1][1] - h[i][1], h[j + 1][2] - h[i][2]) <= REST_R:
                j += 1
            if (h[j][0] - h[i][0]) / fps >= REST_S:
                pe_ = float(np.median([p[1] for p in h[i:j + 1]]))
                pn_ = float(np.median([p[2] for p in h[i:j + 1]]))
                for p in h[i:j + 1]:
                    rest[(g, p[0])] = (pe_, pn_)
                # Parking is the end of the story: if the car never gets 5 m away again, it stays
                # hidden to the end of its track. Otherwise the box blinks off while it rests and
                # back on the moment its solved centre wanders past the rest radius.
                if lot is not None and g not in parked_from \
                        and lot["rect"][0] <= pe_ <= lot["rect"][2] \
                        and lot["rect"][1] <= pn_ <= lot["rect"][3] \
                        and all(np.hypot(p[1] - pe_, p[2] - pn_) < 5.0 for p in h[j + 1:]):
                    parked_from[g] = h[i][0]
                i = j + 1
            else:
                i += 1

    cls_of = {d["gid"]: d["cls"] for frl in fused.values() for d in frl}

    def hidden(g, fr):
        """one rule for the map and for the camera views, so the two never disagree"""
        if cls_of.get(g) == "Pedestrian":
            return False
        if fr >= parked_from.get(g, 1 << 30):
            return True
        p = sm.get((g, fr))
        return bool(parked[g] or (p is not None and zone(*p) != "aisle"))

    def heading(g, fr):
        # a direction is believed only once the track has moved 2.5 m over the window: a stopped car's
        # box jitter is well under that, and used to turn the box sideways while it waited at the light
        h = hist[g]; i = next(k for k, p in enumerate(h) if p[0] == fr)
        reach = 0.0
        for span in (4, 8, 16):
            p0, p1 = h[max(i - span, 0)], h[min(i + span, len(h) - 1)]
            reach = max(reach, float(np.hypot(p1[1] - p0[1], p1[2] - p0[2])))
        # Below that distance no direction can be read from the track, and fuse_tracks' own smoothed
        # heading is just as unfounded there - on a car crawling into a bay it swings through a full
        # turn. Keep the last direction the car was actually seen driving in, or draw no direction.
        if reach <= 2.5: return last_heading.get(g)
        if (g, fr) in hdg_csv: return hdg_csv[(g, fr)]
        for span in (4, 8, 16):
            p0, p1 = h[max(i - span, 0)], h[min(i + span, len(h) - 1)]
            de, dn = p1[1] - p0[1], p1[2] - p0[2]; d = np.hypot(de, dn)
            if d > 2.5:
                th = np.arctan2(dn, de); prev = last_heading.get(g)
                # a sudden flip against the previous believed heading is jitter unless the car is clearly moving
                if prev is not None and d < 5.0 and abs((th - prev + np.pi) % (2 * np.pi) - np.pi) > np.radians(60): return prev
                return th
        return None

    bay_field = bay_ext = None
    if a.base == "wide":
        base_full = cv2.imread(os.path.join(RD.OUTR, "aerial_wide.png")); ext = json.load(open(os.path.join(RD.OUTR, "aerial_wide.json")))
        BE0, BN1, BRES = ext["E0"], ext["N1"], ext["res"]
        if a.bay_dirs:
            import bay_dirs as BD
            bay_field = BD.build(base_full, ext["res"]); bay_ext = ext
        e0, n0, e1, n1 = (ext["E0"], ext["N0"], ext["E1"], ext["N1"]) if a.zoom is None else [float(v) for v in a.zoom.split(",")]
    else:
        base_full = cv2.imread(os.path.join(RD.OUTR, "cam02_fit_ortho", "ortho_corner_grid.png")); BE0, BN1, BRES = F.E0, F.N1, RES
        e0, n0, e1, n1 = (F.E0, F.N0, F.E1, F.N1) if a.zoom is None else [float(v) for v in a.zoom.split(",")]
    x0, y0 = int((e0 - BE0) / BRES), int((BN1 - n1) / BRES); x1, y1 = int((e1 - BE0) / BRES), int((BN1 - n0) / BRES)
    up = 2 if (x1 - x0) < 1600 else 1
    base = cv2.resize(base_full[y0:y1, x0:x1], None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else base_full[y0:y1, x0:x1].copy()
    base = cv2.convertScaleAbs(base, alpha=0.8, beta=0)
    mpx = lambda e, n: ((e - e0) / BRES * up, (n1 - n) / BRES * up)
    SC = up / BRES * RES                                                   # pixels per 4 cm, for box sizes below
    Wv = 1600; ncam = len(cam_names)
    tiles_across = 2 if ncam == 4 else ncam
    tw = Wv // tiles_across; th_ = int(tw * 9 / 16); Hc = th_ * (2 if ncam == 4 else 1)
    Hm = int(base.shape[0] * Wv / base.shape[1])
    size = (Wv, Hc + Hm)
    vw = cv2.VideoWriter(os.path.join(out, "demo.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps_out, size)   # H.264; mp4v is not conformant at this frame size
    if not vw.isOpened():
        print("avc1 not available in this OpenCV build, falling back to mp4v"); vw = cv2.VideoWriter(os.path.join(out, "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, size)
    hh_, mm_, ss_ = (float(v) for v in a.start.split(":")); t_start = hh_ * 3600 + mm_ * 60 + ss_
    frames = sorted(fused); stills = set(frames[len(frames) // 6 * k] for k in range(1, 6)); last_heading = {}
    for fr in frames:
        tops = []; fmap = {d["gid"]: d for d in fused[fr]}            # this frame's objects, for the view labels
        for c in cam_names:
            caps[c].set(cv2.CAP_PROP_POS_FRAMES, fr - 1); ok, im = caps[c].read()
            if not ok: im = np.zeros((2160, 3840, 3), np.uint8)
            im_w = im.shape[1]
            im = cv2.resize(im, (tw, th_)); k = tw / im_w
            for d in tracks[c].get(fr, []):                                # boxes drawn after the resize so lines stay crisp
                g = gid_of.get((c, d["id"])); fo = fmap.get(g) if g is not None else None
                if g is not None and hidden(g, fr): continue                 # dropped on the map, so dropped here too
                g = (g, fo["n_cams"]) if fo else None                         # the view shows the GLOBAL class and this frame's camera count
                col = RD.CLS_COL.get(fo["cls"], (200, 200, 200)) if fo else (110, 110, 110)
                if "box" in d:
                    x1, y1, x2, y2 = [int(v * k) for v in d["box"]]
                    if "poly" in d:                                            # segmentation run: draw the mask itself, filled and outlined
                        P = (d["poly"] * k).astype(np.int32)
                        if g:
                            ov = im.copy(); cv2.fillPoly(ov, [P], col); im = cv2.addWeighted(im, 0.65, ov, 0.35, 0)
                        cv2.polylines(im, [P], True, col, 2 if g else 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2 if g else 1)
                    if g:
                        lab = f"{g[0]}" + ("*" if g[1] > 1 else "")
                        (lw, lh), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(im, (x1, max(y1 - lh - 6, 0)), (x1 + lw + 6, max(y1, lh + 6)), col, -1)
                        cv2.putText(im, lab, (x1 + 3, max(y1 - 4, lh + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(im, (int(d["u"] * k), int(d["v"] * k)), 4, col, -1)
            cv2.rectangle(im, (0, 0), (150, 34), (0, 0, 0), -1); cv2.putText(im, c, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); tops.append(im)
        top = np.vstack([np.hstack(tops[:2]), np.hstack(tops[2:])]) if ncam == 4 else np.hstack(tops)
        if top.shape[1] != Wv: top = cv2.copyMakeBorder(top, 0, 0, 0, Wv - top.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        bev = base.copy()
        for d in fused[fr]:
            g, cls = d["gid"], d["cls"]; col = RD.CLS_COL.get(cls, (200, 200, 200))
            if hidden(g, fr): continue
            e, n = sm[(g, fr)]
            resting = cls != "Pedestrian" and (g, fr) in rest
            if resting: e, n = rest[(g, fr)]                                # standing still: one position, no drift
            tr = [(ee, nn) for (ff, ee, nn) in hist[g] if fr - a.trail * 2 <= ff <= fr]
            if len(tr) > 1: cv2.polylines(bev, [np.array([mpx(ee, nn) for ee, nn in tr], np.int32)], False, col, 2, cv2.LINE_AA)
            x, y = mpx(e, n)
            ghost = False                                                   # interpolated frames drawn like the rest (hollow boxes confused viewers)
            if cls == "Pedestrian":
                cv2.circle(bev, (int(x), int(y)), 9, col, 2 if ghost else -1, cv2.LINE_AA)
                if not ghost: cv2.circle(bev, (int(x), int(y)), 9, (30, 30, 30), 2, cv2.LINE_AA)
            else:
                L, Wd = RD.SIZE.get(cls, (4.5, 1.9))
                th, from_bay = None, False
                if resting and bay_field is not None:
                    # It is standing. Whatever the track says is the aisle it arrived along, not the
                    # bay it is in, so ask the paint instead - but only where the lines around it
                    # agree. The paint gives the axis, never which end is the front.
                    aa, cc = BD.at(bay_field, bay_ext, e, n)
                    if aa is not None and cc >= 0.8: th, from_bay = np.radians(aa), True
                if th is None: th = heading(g, fr)
                if th is None: th = last_heading.get(g)
                if th is None:
                    cv2.circle(bev, (int(x), int(y)), 11, col, 2 if ghost else -1, cv2.LINE_AA)
                    if not ghost: cv2.circle(bev, (int(x), int(y)), 11, (30, 30, 30), 2, cv2.LINE_AA)
                else:
                    if not from_bay: last_heading[g] = th        # a bay axis is not a driving heading
                    pe, pn = push[(g, fr)]
                    if not centred and abs(pe) + abs(pn) < 1e-6 and d["cam"] is not None:
                        cp = cams[d["cam"]].pos; ve, vn = e - cp[0], n - cp[1]; dist = max(np.hypot(ve, vn), 1e-6); pe, pn = ve / dist, vn / dist
                    # the foot point is the vehicle's NEAR edge as seen from the camera: push the box centre away by the
                    # rectangle's half-extent along that view direction (half a length for an end-on bus, half a width side-on)
                    ca = abs(np.cos(th) * pe + np.sin(th) * pn); sa = abs(-np.sin(th) * pe + np.cos(th) * pn)
                    half = (L / 2) * ca + (Wd / 2) * sa
                    x, y = mpx(e + pe * half, n + pn * half)
                    box = cv2.boxPoints(((float(x), float(y)), (L / BRES * up, Wd / BRES * up), -np.degrees(th)))
                    if ghost:
                        cv2.polylines(bev, [box.astype(np.int32)], True, col, 3, cv2.LINE_AA)        # hollow: hidden, position inferred
                    else:
                        ov = bev.copy(); cv2.fillPoly(ov, [box.astype(np.int32)], col); bev = cv2.addWeighted(bev, 0.45, ov, 0.55, 0)
                        cv2.polylines(bev, [box.astype(np.int32)], True, (30, 30, 30), 2, cv2.LINE_AA)
                    if not from_bay:                              # the bay gives the axis, not the nose
                        nose = (x + np.cos(th) * L / 2 / BRES * up, y - np.sin(th) * L / 2 / BRES * up); cv2.circle(bev, (int(nose[0]), int(nose[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
            lab = f"{g}" + ("*" if d["n_cams"] > 1 else "") + ("?" if ghost else "")
            cv2.putText(bev, lab, (int(x) + 12, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA); cv2.putText(bev, lab, (int(x) + 12, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        if truth is not None and a.rec_start is not None:
            w = truth["rec0"] + a.rec_start + fr / fps
            if truth["wall"][0] <= w <= truth["wall"][-1]:
                te, tn, th_t = np.interp(w, truth["wall"], truth["E"]), np.interp(w, truth["wall"], truth["N"]), np.interp(w, truth["wall"], truth["hd"])
                tx, ty = mpx(te, tn); box_t = cv2.boxPoints(((float(tx), float(ty)), (4.7 / BRES * up, 1.9 / BRES * up), -np.degrees(th_t)))
                cv2.polylines(bev, [box_t.astype(np.int32)], True, (255, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(bev, "truth", (int(tx) + 12, int(ty) + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
            if tobj:
                key = int(round((a.rec_start + fr / fps) * 2) / 2 * 100)
                for te, tn, kind in tobj.get(key, []):
                    if e0 < te < e1 and n0 < tn < n1:
                        tx, ty = mpx(te, tn)
                        if kind == "Pedestrian": cv2.circle(bev, (int(tx), int(ty)), 6, (255, 255, 255), -1, cv2.LINE_AA); cv2.circle(bev, (int(tx), int(ty)), 6, (255, 0, 255), 1, cv2.LINE_AA)
                        else: cv2.circle(bev, (int(tx), int(ty)), 4, (200, 200, 200), 1, cv2.LINE_AA)
        bev = cv2.resize(bev, (Wv, Hm), interpolation=cv2.INTER_AREA)
        t = t_start + fr / fps
        cv2.rectangle(bev, (0, 0), (Wv, 36), (0, 0, 0), -1)
        cv2.putText(bev, f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}   {a.site}, {a.date}   {ncam} cameras   * = multi-camera" + ("   magenta = LiDAR truth" if truth is not None else ""), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        x = Wv - 620
        for name, col in (("pedestrian", RD.CLS_COL["Pedestrian"]), ("car", RD.CLS_COL["Passenger Vehicle"]), ("truck", RD.CLS_COL["Truck"]), ("bus", RD.CLS_COL["Bus"]), ("bike", RD.CLS_COL["Bicycle"])):
            cv2.circle(bev, (x, 18), 7, col, -1); cv2.putText(bev, name, (x + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); x += 120
        frame = np.vstack([top, bev]); vw.write(frame)
        if fr in stills: cv2.imwrite(os.path.join(out, f"still_{fr:05d}.png"), frame)
    vw.release(); print("wrote", out, len(frames), "frames; %d diverged centres replaced by their foot point" % clamped)


if __name__ == "__main__":
    main()
