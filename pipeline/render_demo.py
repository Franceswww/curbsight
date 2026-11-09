"""
render_demo.py - detections of two roadside cameras on the aerial photo, the way a live
bird's-eye dashboard shows them: pedestrians as dots, vehicles as rectangles along their heading,
IDs, short trails, the two camera views on top.

    python render_demo.py --clipdir <dir with cam02_*.mp4, cam03_*.mp4, cam02/, cam03/> --fit02 cam02_fit_ortho --fit03 cam03_fit_ortho

Positions: foot point -> camera->aerial homography (out_refined/<fit>/H.json, aerial frame in
LiDAR-origin metres, optional division-model lens term) -> aerial pixels (ortho_corner_grid.png,
4 cm). Per-track smoothing: exponential average over the last 3 processed frames.
Outputs in <clipdir>/demo/: demo.mp4, still_*.png
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np

import fit_cam_H as F

OUTR = F.OUTR
RES = F.RES
CLS_COL = {"Pedestrian": (40, 220, 255), "Passenger Vehicle": (255, 170, 60), "Truck": (80, 80, 240), "Bus": (60, 150, 255), "Bicycle": (120, 230, 120), "Motorcycle": (120, 230, 120)}
SIZE = {"Passenger Vehicle": (4.5, 1.9), "Truck": (5.5, 2.1), "Bus": (10.5, 2.6), "Motorcycle": (2.0, 0.8), "Bicycle": (1.8, 0.6)}   # Truck 6.5 -> 5.5 (2026-11-03): the detector's trucks here are pickups (LiDAR-visible 5.1 m); near-point check crossing 1 90 -> 70 cm, nothing else moved
CAM_COL = {"cam02": (255, 160, 40), "cam03": (40, 140, 255)}
CAM_POS = (-11.0, 6.0)   # both cameras share the NE-corner pole (LiDAR frame); used to push a box away from the camera


class Cam:
    """either a flat-plane homography (H.json) or a camera model on the LiDAR ground surface (cam.json)"""
    def __init__(self, name, fit):
        self.name = name; self.pnp = None
        cj = os.path.join(OUTR, fit, "cam.json")
        if os.path.exists(cj):
            import fit_pnp
            j = json.load(open(cj))
            self.pnp = fit_pnp.Camera(np.array(j["K"]), np.array(j["dist"]), np.array(j["rvec"]), np.array(j["tvec"]))
            self.surface = fit_pnp.Ground(j.get("ground_src", "usgs")); self.pos = np.array(j["centre"][:2])
            return
        j = json.load(open(os.path.join(OUTR, fit, "H.json")))
        self.H = np.array(j["H"]); self.lam = float(j.get("lambda_", 0)); self.c = np.array(j.get("centre", [1920, 1080])); self.sc = float(j.get("scale", np.hypot(3840, 2160)))
        self.pos = None

    def centre_from_box(self, u, v, th, L, W, H, iters=4):
        """the vehicle centre whose L x W x H box with heading th, projected through this camera, has the observed
        2-D box bottom-centre (u, v): the tight-box constraint of Mousavian et al. (CVPR 2017) reduced to the bottom
        edge and the horizontal centre. Start from the analytic push, then correct by the ground-plane difference
        between the observed and the modelled foot point; converges in 3-4 steps. Handles end-on (the foot point is
        the near face's centre), oblique (the near corner) and side-on views alike, which no closed formula did
        (checked against the LiDAR car's track, 2026-10-27)."""
        if self.pnp is None: return None
        (e, n), = self.ground([[u, v]]); ax = np.array([np.cos(th), np.sin(th)]); lat = np.array([-np.sin(th), np.cos(th)])
        foot = np.array([e, n]); c = foot.copy(); vd = c - self.pos; vd = vd / max(np.linalg.norm(vd), 1e-6); cc, ss = float(vd @ ax), float(vd @ lat)
        c = c + ax * (L / 2 * cc * abs(cc)) + lat * (W / 2 * ss * abs(ss)); c0 = c.copy()
        # the centre can never be further from the foot point than half the box diagonal; a solve that wanders
        # past that (flat viewing angles, a box cut by the frame edge) is discarded for the analytic start value
        reach = 0.5 * np.hypot(L, W) + 0.3; last = None
        for _ in range(iters):
            z0 = float(self.surface(np.array([c[0]]), np.array([c[1]]))[0])
            # horizontal extent from the full body; the LOWEST visible points are the tyre contacts, i.e. the wheel
            # rectangle (wheelbase x track ~ 0.65 L x 0.85 W), not the bumper's ground footprint - with the full
            # cuboid the modelled bottom row sat 20-45 px too low within 15 m of the camera
            P = [[q[0], q[1], z0 + dz] for sx in (1, -1) for sy in (1, -1) for q in [c + sx * L / 2 * ax + sy * W / 2 * lat] for dz in (0.0, H)]
            Pw = [[q[0], q[1], z0] for sx in (1, -1) for sy in (1, -1) for q in [c + sx * 0.65 * L / 2 * ax + sy * 0.85 * W / 2 * lat]]
            uv = self.pnp.project(P); um, vb = (uv[:, 0].min() + uv[:, 0].max()) / 2, self.pnp.project(Pw)[:, 1].max()
            (e1, n1), (e2, n2) = self.ground([[u, v], [um, vb]])
            step = np.array([e1 - e2, n1 - n2]); sn = float(np.hypot(*step))
            if sn > reach or (last is not None and sn > last): break                   # diverging
            c = c + step; last = sn
            if sn < 0.02: break
        return c if np.hypot(*(c - foot)) <= reach else c0

    def box_consistency(self, c, th, L, W, H, box):
        """how well the observed 2-D box (x1, y1, x2, y2) matches the L x W x H cuboid at centre c, heading th, seen
        through this camera: min over width and height of min(obs / model, model / obs), 0..1. A bus cut by a lamp
        post or a box that swallowed a neighbour scores low; the fusion then trusts that frame less."""
        if self.pnp is None or box is None: return 1.0
        ax, lat = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        z0 = float(self.surface(np.array([c[0]]), np.array([c[1]]))[0])
        P = [[q[0], q[1], z0 + dz] for sx in (1, -1) for sy in (1, -1) for q in [c + sx * L / 2 * ax + sy * W / 2 * lat] for dz in (0.0, H)]
        uv = self.pnp.project(P); mw, mh = uv[:, 0].max() - uv[:, 0].min(), uv[:, 1].max() - uv[:, 1].min()
        ow, oh = box[2] - box[0], box[3] - box[1]
        if min(mw, mh, ow, oh) <= 1: return 1.0
        return float(min(min(ow / mw, mw / ow), min(oh / mh, mh / oh)))

    def ground(self, pts):
        if self.pnp is not None:
            g = self.pnp.to_ground(pts, self.surface)[:, :2]
        else:
            p = np.asarray(pts, np.float32)
            if abs(self.lam) > 1e-9:
                q = (p - self.c) / self.sc; r2 = np.sum(q ** 2, 1, keepdims=True); p = (q / (1 + self.lam * r2) * self.sc + self.c).astype(np.float32)
            g = cv2.perspectiveTransform(p.reshape(1, -1, 2), self.H)[0]
        return g + self.learned_correction(g)

    def learned_correction(self, g):
        """GROUND_CORR=<dir> with <dir>/<cam>.npz (learn_ground_correction.py): a smooth per-camera correction of the
        ground position, kernel ridge on truth pairs. Unset: zero, the mapping is the fitted camera model alone."""
        if not hasattr(self, "_corr"):
            self._corr = None; d = os.environ.get("GROUND_CORR")
            f = os.path.join(d, f"{self.name}.npz") if d else None
            if f and os.path.exists(f):
                z = np.load(f); self._corr = (z["X"], z["alpha"], float(z["l"]))
        if self._corr is None: return np.zeros_like(g, dtype=np.float64)
        X, alpha, l = self._corr; G = np.asarray(g, np.float64)
        d2 = ((G[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / (l * l)) @ alpha


def load_tracks(path):
    T = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = dict(id=int(r["track_id"]), cls=r["class"], u=float(r["foot_x"]), v=float(r["foot_y"]))
            if "x1" in r: d["box"] = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            if r.get("poly"): d["poly"] = np.array([[int(v) for v in q.split(",")] for q in r["poly"].split(";")], np.int32)   # mask outline (track_clip_seg.py)
            T[int(r["frame"])].append(d)
    return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clipdir", required=True); ap.add_argument("--fit02", default="cam02_fit_ortho"); ap.add_argument("--fit03", default="cam03_fit_ortho")
    ap.add_argument("--fps-out", type=float, default=13.7); ap.add_argument("--trail", type=int, default=12); ap.add_argument("--zoom", default="-32,-22,10,12", help="E0,N0,E1,N1 metres of the map window")
    a = ap.parse_args()
    out = os.path.join(a.clipdir, "demo"); os.makedirs(out, exist_ok=True)
    cams = {"cam02": Cam("cam02", a.fit02), "cam03": Cam("cam03", a.fit03)}
    if cams["cam02"].pos is not None:
        cam_pos = cams["cam02"].pos                                                   # solved camera centre
    else:
        lo = json.load(open(os.path.join(OUTR, a.fit02, "H.json")))["lidar_to_ortho"]  # camera pole from the LiDAR frame into the aerial frame
        th_, t_, c_ = np.radians(lo["rotation_deg"]), np.array([lo["shift_east_m"], lo["shift_north_m"]]), np.array(lo["centre"])
        cam_pos = (np.array(CAM_POS) - c_) @ np.array([[np.cos(th_), -np.sin(th_)], [np.sin(th_), np.cos(th_)]]).T + c_ + t_
    tracks = {c: load_tracks(os.path.join(a.clipdir, c, "trajectories.csv")) for c in cams}
    caps = {c: cv2.VideoCapture([os.path.join(a.clipdir, f) for f in os.listdir(a.clipdir) if f.startswith(c) and f.endswith(".mp4")][0]) for c in cams}
    fps = caps["cam02"].get(cv2.CAP_PROP_FPS)
    base_full = cv2.imread(os.path.join(OUTR, "cam02_fit_ortho", "ortho_corner_grid.png"))
    e0, n0, e1, n1 = [float(v) for v in a.zoom.split(",")]
    x0, y0 = F.m2px([[e0, n1]])[0].astype(int); x1, y1 = F.m2px([[e1, n0]])[0].astype(int)
    base = base_full[y0:y1, x0:x1]; up = 2                                   # render at 2x and downsample for smooth edges
    base = cv2.resize(base, (base.shape[1] * up, base.shape[0] * up), interpolation=cv2.INTER_CUBIC)
    base = cv2.convertScaleAbs(base, alpha=0.8, beta=0)                       # darker so the marks stand out
    mpx = lambda e, n: ((e - e0) / RES * up, (n1 - n) / RES * up)

    # smoothed positions per camera/track/frame
    frames = sorted(set(tracks["cam02"]) | set(tracks["cam03"]))
    pos = {c: {} for c in cams}; state = {c: {} for c in cams}
    for fr in frames:
        for c in cams:
            dets = tracks[c].get(fr, [])
            if not dets: continue
            g = cams[c].ground([[d["u"], d["v"]] for d in dets])
            for d, (e, n) in zip(dets, g):
                prev = state[c].get(d["id"])
                if prev is not None and prev[0] == fr - 2 and np.hypot(e - prev[1], n - prev[2]) < 3:
                    e, n = 0.5 * e + 0.5 * prev[1], 0.5 * n + 0.5 * prev[2]
                state[c][d["id"]] = (fr, e, n); pos[c][(fr, d["id"])] = (e, n, d["cls"])
    hist = {c: defaultdict(list) for c in cams}
    for c in cams:
        for (fr, tid), (e, n, cls) in sorted(pos[c].items()): hist[c][tid].append((fr, e, n))

    # parked = the track never moves more than 1.5 m in the whole clip: not traffic, not drawn
    parked = {c: {tid: (max(np.hypot(e - h[0][1], n - h[0][2]) for _, e, n in h) < 1.5) for tid, h in hist[c].items()} for c in cams}

    def heading(c, tid, fr):
        h = hist[c][tid]; i = next(k for k, p in enumerate(h) if p[0] == fr)
        for span in (4, 8, 16):                                   # widen the window until the track has moved enough
            p0, p1 = h[max(i - span, 0)], h[min(i + span, len(h) - 1)]
            de, dn = p1[1] - p0[1], p1[2] - p0[2]
            if np.hypot(de, dn) > 0.6: return np.arctan2(dn, de)
        return None

    Wv = 1600; Hc = int(Wv / 2 * 9 / 16); Hm = int(base.shape[0] * Wv / base.shape[1])
    vw = cv2.VideoWriter(os.path.join(out, "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps_out, (Wv, Hc + Hm))
    stills = set(frames[len(frames) // 6 * k] for k in range(1, 6))
    last_heading = {}
    for fr in frames:
        tops = []
        for c in cams:
            caps[c].set(cv2.CAP_PROP_POS_FRAMES, fr - 1); ok, im = caps[c].read()
            if not ok: im = np.zeros((2160, 3840, 3), np.uint8)
            for d in tracks[c].get(fr, []):
                col = CLS_COL.get(d["cls"], (200, 200, 200))
                cv2.circle(im, (int(d["u"]), int(d["v"])), 12, col, 3); cv2.putText(im, str(d["id"]), (int(d["u"]) + 14, int(d["v"]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
            im = cv2.resize(im, (Wv // 2, Hc)); cv2.rectangle(im, (0, 0), (110, 34), (0, 0, 0), -1); cv2.putText(im, c, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CAM_COL[c], 2); tops.append(im)
        bev = base.copy()
        for c in cams:
            for d in tracks[c].get(fr, []):
                e, n, cls = pos[c][(fr, d["id"])]; col = CLS_COL.get(cls, (200, 200, 200))
                if cls != "Pedestrian" and parked[c][d["id"]]: continue
                tr = [(ee, nn) for (ff, ee, nn) in hist[c][d["id"]] if fr - a.trail * 2 <= ff <= fr]
                if len(tr) > 1: cv2.polylines(bev, [np.array([mpx(ee, nn) for ee, nn in tr], np.int32)], False, col, 2, cv2.LINE_AA)
                x, y = mpx(e, n)
                if cls == "Pedestrian":
                    cv2.circle(bev, (int(x), int(y)), 9, col, -1, cv2.LINE_AA); cv2.circle(bev, (int(x), int(y)), 9, (30, 30, 30), 2, cv2.LINE_AA)
                else:
                    L, Wd = SIZE.get(cls, (4.5, 1.9)); th = heading(c, d["id"], fr)
                    if th is None: th = last_heading.get((c, d["id"]))
                    if th is None:                                   # moving later, but no direction yet: a dot, not a box
                        cv2.circle(bev, (int(x), int(y)), 11, col, -1, cv2.LINE_AA); cv2.circle(bev, (int(x), int(y)), 11, (30, 30, 30), 2, cv2.LINE_AA)
                    else:
                        last_heading[(c, d["id"])] = th
                        # the foot point is under the vehicle's near side, not its centre: push the box half a width away from the camera
                        ve, vn = e - cam_pos[0], n - cam_pos[1]; dist = max(np.hypot(ve, vn), 1e-6)
                        x, y = mpx(e + ve / dist * Wd / 2, n + vn / dist * Wd / 2)
                        box = cv2.boxPoints(((float(x), float(y)), (L / RES * up, Wd / RES * up), -np.degrees(th)))
                        ov = bev.copy(); cv2.fillPoly(ov, [box.astype(np.int32)], col); bev = cv2.addWeighted(bev, 0.45, ov, 0.55, 0)
                        cv2.polylines(bev, [box.astype(np.int32)], True, (30, 30, 30), 2, cv2.LINE_AA)
                        nose = (x + np.cos(th) * L / 2 / RES * up, y - np.sin(th) * L / 2 / RES * up); cv2.circle(bev, (int(nose[0]), int(nose[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(bev, str(d["id"]), (int(x) + 12, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA); cv2.putText(bev, str(d["id"]), (int(x) + 12, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        bev = cv2.resize(bev, (Wv, Hm), interpolation=cv2.INTER_AREA)
        t = 13 * 3600 + 30 * 60 + fr / fps
        cv2.rectangle(bev, (0, 0), (Wv, 36), (0, 0, 0), -1)
        cv2.putText(bev, f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}   Sanford Dr at Cedar St, 2026-03-23   aerial: Athens-Clarke County 2025", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        x = Wv - 520
        x = Wv - 620
        for name, col in (("pedestrian", CLS_COL["Pedestrian"]), ("car", CLS_COL["Passenger Vehicle"]), ("truck", CLS_COL["Truck"]), ("bus", CLS_COL["Bus"]), ("bike", CLS_COL["Bicycle"])):
            cv2.circle(bev, (x, 18), 7, col, -1); cv2.putText(bev, name, (x + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); x += 120
        frame = np.vstack([np.hstack(tops), bev]); vw.write(frame)
        if fr in stills: cv2.imwrite(os.path.join(out, f"still_{fr:05d}.png"), frame)
    vw.release(); print("wrote", out, len(frames), "frames")


if __name__ == "__main__":
    main()
