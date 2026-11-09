"""
fit_cam_H.py - ground homography camera -> LiDAR map from the road paint itself.

The 4-corner solution from picked crosswalk corners is only the starting point;
the homography (plus one radial-distortion term) is then refined by registering
ALL the white-paint pixels of the camera frame (inside hand-drawn road polygons)
against the bright paint of the LiDAR map with a two-sided chamfer distance.

    python fit_cam_H.py --spec specs/cam02.json

spec json:
  image        camera frame
  rois         list of image polygons [[u,v],...] around road paint (crosswalks, stop bars)
  map_ladders  names in out_refined/ladders_tight.json used as map paint ROI
  map_boxes    extra map ROIs [[e0,n0,e1,n1],...] in metres (stop bars etc.)
  init_img / init_map   >= 4 starting correspondences (pixels / metres)
Outputs in out_refined/<cam>_fit/: H.json, overlay.png, control_points.jpg, map_points.png
"""
import argparse
import json
import os

import cv2
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUTR = os.path.join(HERE, "..", "calibration")      # solved camera models and the base map
E0, E1, N0, N1, RES = -34.0, 16.0, -24.0, 14.0, 0.04      # extent of corner_img.npy
LADDERS = json.load(open(os.path.join(OUTR, "ladders_tight.json")))


def m2px(p):
    p = np.asarray(p, float); return np.stack([(p[:, 0] - E0) / RES, (N1 - p[:, 1]) / RES], 1)


def px2m(q):
    q = np.asarray(q, float); return np.stack([q[:, 0] * RES + E0, N1 - q[:, 1] * RES], 1)


def paint_mask_image(im, rois, vmin=200, smax=55):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] > vmin) & (hsv[:, :, 1] < smax)).astype(np.uint8)
    roi = np.zeros(white.shape, np.uint8)
    for poly in rois:
        cv2.fillPoly(roi, [np.array(poly, np.int32)], 1)
    m = white * roi
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def paint_mask_map(img, ladders, boxes, pad_m=0.8, pct=88):
    roi = np.zeros(img.shape, np.uint8)
    for name in ladders:
        cv2.fillPoly(roi, [m2px(LADDERS[name]["corners"]).astype(np.int32)], 1)
    for e0, n0, e1, n1 in boxes:
        cv2.rectangle(roi, tuple(m2px([[e0, n1]])[0].astype(int)), tuple(m2px([[e1, n0]])[0].astype(int)), 1, -1)
    k = int(pad_m / RES) | 1
    roi = cv2.dilate(roi, np.ones((k, k), np.uint8))
    vals = img[(roi > 0) & (img > 0)]
    thr = np.percentile(vals, pct)
    return ((img > thr) & (roi > 0)).astype(np.uint8), roi


def area_of(q):
    return 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])


class Model:
    """H is parametrised by the map positions of 4 anchor image points, plus lambda."""
    def __init__(self, img_anchor, cx, cy, sc):
        self.a = np.asarray(img_anchor, np.float32); self.cx, self.cy, self.sc = cx, cy, sc

    def undist(self, p, lam):
        q = (p - [self.cx, self.cy]) / self.sc; r2 = np.sum(q ** 2, 1, keepdims=True)
        return (q / (1 + lam * r2) * self.sc + [self.cx, self.cy]).astype(np.float32)

    def H(self, params):
        dst = np.asarray(params[:8], np.float32).reshape(4, 2); lam = params[8]
        H, _ = cv2.findHomography(self.undist(self.a, lam), dst, 0)
        return H, lam

    def warp(self, pts, params):
        H, lam = self.H(params)
        return cv2.perspectiveTransform(self.undist(np.asarray(pts, np.float32), lam).reshape(1, -1, 2), H)[0]

    def unwarp(self, pts_m, params):
        H, lam = self.H(params)
        p = cv2.perspectiveTransform(np.asarray(pts_m, np.float32).reshape(1, -1, 2), np.linalg.inv(H))[0]
        q = (p - [self.cx, self.cy]) / self.sc; r2 = np.sum(q ** 2, 1, keepdims=True)
        # invert the division model (fixed point, 4 rounds)
        s = np.ones_like(r2)
        for _ in range(4):
            s = 1 + lam * r2 * s * s
        return (q * s * self.sc + [self.cx, self.cy]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--no-lambda", action="store_true"); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    spec = json.load(open(args.spec)); cam = spec["cam"]
    out = os.path.join(OUTR, f"{cam}_fit{args.tag}"); os.makedirs(out, exist_ok=True)
    im = cv2.imread(os.path.join(HERE, spec["image"])); Hh, Ww = im.shape[:2]
    corner = np.load(os.path.join(OUTR, "corner_img.npy"))

    mi = paint_mask_image(im, spec["rois"], spec.get("vmin", 200), spec.get("smax", 55))
    mm, roi_m = paint_mask_map(corner, spec["map_ladders"], spec.get("map_boxes", []), spec.get("map_pad", 0.8), spec.get("map_pct", 88))
    print(f"image paint px {int(mi.sum())}, map paint px {int(mm.sum())}")
    dt_map = cv2.distanceTransform(1 - mm, cv2.DIST_L2, 5) * RES              # metres
    dt_img = cv2.distanceTransform(1 - mi, cv2.DIST_L2, 5)                    # pixels
    rng = np.random.default_rng(0)
    pi = np.column_stack(np.where(mi > 0))[:, ::-1].astype(np.float32); pi = pi[rng.choice(len(pi), min(len(pi), 8000), replace=False)]
    pm = np.column_stack(np.where(mm > 0))[:, ::-1].astype(np.float32); pm = px2m(pm[rng.choice(len(pm), min(len(pm), 8000), replace=False)]).astype(np.float32)

    sets = spec.get("init_sets") or [dict(init_img=spec["init_img"], init_map=spec["init_map"])]
    init_img = np.array(sets[0]["init_img"], np.float32)
    model = Model(init_img[:4], Ww / 2, Hh / 2, np.hypot(Ww, Hh))

    def cost(params, verbose=False):
        if args.no_lambda: params = np.concatenate([params[:8], [0.0]])
        if abs(params[8]) > 0.8: return 10.0
        w = model.warp(pi, params); q = m2px(w)
        okq = (q[:, 0] >= 0) & (q[:, 0] < mm.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < mm.shape[0] - 1)
        d1 = np.full(len(q), 1.0); d1[okq] = np.minimum(dt_map[q[okq, 1].astype(int), q[okq, 0].astype(int)], 1.0)
        u = model.unwarp(pm, params)
        oku = (u[:, 0] >= 0) & (u[:, 0] < Ww - 1) & (u[:, 1] >= 0) & (u[:, 1] < Hh - 1)
        d2 = np.full(len(u), 150.0); d2[oku] = np.minimum(dt_img[u[oku, 1].astype(int), u[oku, 0].astype(int)], 150.0)
        if verbose:
            print(f"  image paint -> map: median {np.median(d1):.3f} m, mean(capped) {d1.mean():.3f} m, within 15 cm {np.mean(d1 < 0.15) * 100:.0f}%")
            print(f"  map paint -> image: median {np.median(d2):.1f} px, within 20 px {np.mean(d2 < 20) * 100:.0f}%")
        # keep the anchors within `anchor_radius` m of their starting positions (prevents collapse)
        dev = np.hypot(*(params[:8].reshape(4, 2) - p_ref).T); pen = np.sum(np.maximum(dev - R_ANCH, 0.0) ** 2)
        # the anchor quadrilateral must stay convex with the same orientation as at the start (no fold, no collapse)
        qd = params[:8].reshape(4, 2)[[0, 1, 3, 2]]; cr = [np.cross(qd[(k + 1) % 4] - qd[k], qd[(k + 2) % 4] - qd[(k + 1) % 4]) for k in range(4)]
        if np.sign(cr) .tolist() != ref_sign or abs(area_of(qd)) < 0.4 * ref_area or abs(area_of(qd)) > 2.5 * ref_area: return 10.0
        return d1.mean() / 1.0 + d2.mean() / 150.0 + pen

    R_ANCH = spec.get("anchor_radius", 3.0); p_ref = None
    best = (1e9, None)
    for si, st in enumerate(sets):
        ii = np.array(st["init_img"], np.float32); mm_ = np.array(st["init_map"], np.float32)
        H0, _ = cv2.findHomography(ii, mm_, 0)
        p0 = np.concatenate([cv2.perspectiveTransform(init_img[:4].reshape(1, -1, 2), H0)[0].ravel(), [0.0]])
        p_ref = p0[:8].reshape(4, 2).copy()
        pq = p_ref[[0, 1, 3, 2]]; ref_sign = np.sign([np.cross(pq[(k + 1) % 4] - pq[k], pq[(k + 2) % 4] - pq[(k + 1) % 4]) for k in range(4)]).tolist(); ref_area = abs(area_of(pq))
        if len(set(ref_sign)) != 1:
            print(f"init set {si}: anchor quad not convex, skipped"); continue
        print(f"init set {si}: cost {cost(p0):.4f}")
        bs = (cost(p0), p0)
        for rnd in range(2):
            r = minimize(cost, bs[1], method="Powell", options=dict(xtol=1e-3, ftol=1e-5, maxfev=6000))
            r = minimize(cost, r.x, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxfev=6000))
            if r.fun < bs[0] - 1e-6: bs = (r.fun, r.x)
        print(f"   refined: cost {bs[0]:.4f}  lambda {bs[1][8]:+.3f}")
        if bs[0] < best[0]: best = bs; best_ref = (p_ref, ref_sign, ref_area)
    p = best[1]; p_ref, ref_sign, ref_area = best_ref; init_map = np.array(sets[0]["init_map"], np.float32)
    if args.no_lambda: p[8] = 0.0
    print("final:"); cost(p, True)
    H, lam = model.H(p)
    w_init = model.warp(init_img, p); err = np.hypot(*(w_init - init_map).T)
    print("starting correspondences after refinement (m):", np.round(err, 2))
    json.dump(dict(cam=cam, H=H.tolist(), lambda_=float(lam), centre=[Ww / 2, Hh / 2], scale=float(np.hypot(Ww, Hh)),
                   anchors_img=init_img[:4].tolist(), anchors_map=p[:8].reshape(4, 2).tolist(),
                   note="map = undist(pixel) via division model, then H; map frame east/north metres, origin 33.94760,-83.37630"),
              open(os.path.join(out, "H.json"), "w"), indent=1)

    # ---- figures
    lo, hi = np.percentile(corner[corner > 0], [20, 96]); v = (np.clip((corner - lo) / (hi - lo), 0, 1) ** 0.5 * 255).astype(np.uint8)
    base = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    # warp the (undistorted) camera frame into the map pixel grid
    map_from_m = np.array([[1 / RES, 0, -E0 / RES], [0, -1 / RES, N1 / RES], [0, 0, 1]])
    if abs(lam) > 1e-6:
        xs, ys = np.meshgrid(np.arange(Ww, dtype=np.float32), np.arange(Hh, dtype=np.float32))
        src = model.unwarp(np.zeros((0, 2)), p) if False else None
        # undistorted image: for each undistorted pixel find the distorted source pixel
        grid = np.stack([xs.ravel(), ys.ravel()], 1)
        q = (grid - [Ww / 2, Hh / 2]) / model.sc; r2 = np.sum(q ** 2, 1, keepdims=True); s = np.ones_like(r2)
        for _ in range(4): s = 1 + lam * r2 * s * s
        srcp = q * s * model.sc + [Ww / 2, Hh / 2]
        und = cv2.remap(im, srcp[:, 0].reshape(Hh, Ww).astype(np.float32), srcp[:, 1].reshape(Hh, Ww).astype(np.float32), cv2.INTER_LINEAR)
    else:
        und = im
    warped = cv2.warpPerspective(und, map_from_m @ H, (base.shape[1], base.shape[0]))
    ov = cv2.addWeighted(base, 0.45, warped, 0.75, 0)
    for name, L in LADDERS.items():
        cv2.polylines(ov, [m2px(L["corners"]).astype(np.int32)], True, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(out, "overlay.png"), ov)

    # control-point image: ladder outlines and anchor points projected into the frame
    ci = im.copy()
    for name, L in LADDERS.items():
        c = np.array(L["corners"], np.float32)
        u = model.unwarp(c, p)
        if np.all((u[:, 0] > -500) & (u[:, 0] < Ww + 500) & (u[:, 1] > -500) & (u[:, 1] < Hh + 500)):
            cv2.polylines(ci, [u.astype(np.int32)], True, (0, 255, 255), 3)
            for k, (uu, vv) in enumerate(u):
                cv2.circle(ci, (int(uu), int(vv)), 12, (0, 0, 255), 3)
                cv2.putText(ci, f"{name}{k + 1} ({c[k, 0]:.1f},{c[k, 1]:.1f})", (int(uu) + 14, int(vv) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    # the image paint used, warped positions coloured by residual
    w = model.warp(pi, p); q = m2px(w); okq = (q[:, 0] >= 0) & (q[:, 0] < mm.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < mm.shape[0] - 1)
    d1 = np.full(len(q), 1.0); d1[okq] = dt_map[q[okq, 1].astype(int), q[okq, 0].astype(int)]
    for (uu, vv), d in zip(pi[::4], d1[::4]):
        col = (0, 200, 0) if d < 0.15 else ((0, 165, 255) if d < 0.4 else (255, 0, 255))
        cv2.circle(ci, (int(uu), int(vv)), 2, col, -1)
    cv2.putText(ci, f"{cam}: yellow = LiDAR crosswalk outlines projected with the fitted H; paint dots: green <15 cm, orange <40 cm, magenta worse", (30, Hh - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out, "control_points.jpg"), ci, [cv2.IMWRITE_JPEG_QUALITY, 88])

    mp = base.copy()
    for (e, n), d in zip(w, d1):
        col = (0, 200, 0) if d < 0.15 else ((0, 165, 255) if d < 0.4 else (255, 0, 255))
        x, y = m2px([[e, n]])[0]
        if 0 <= x < mp.shape[1] and 0 <= y < mp.shape[0]: mp[int(y), int(x)] = col
    for name, L in LADDERS.items():
        cv2.polylines(mp, [m2px(L["corners"]).astype(np.int32)], True, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(out, "map_points.png"), mp)
    print("wrote", out)


if __name__ == "__main__":
    main()
