"""
fit_cam_H_ortho.py - the same paint registration as fit_cam_H.py, but against the county's
2025 orthophoto instead of the LiDAR map, and a comparison of the two homographies.

    python fit_cam_H_ortho.py --spec specs/cam02.json --lidar-fit cam02_fit_nolam
    python fit_cam_H_ortho.py --spec specs/cam03_all.json --lidar-fit cam03_fit_D --offset-from out_refined/cam02_fit_ortho/H.json --lens --tag _all

Steps
  1. resample the orthophoto (maps/ortho, EPSG:3857) into the LiDAR corner grid
     (east/north metres from 33.94760,-83.37630, 4 cm) so both maps share one frame
  2. rigid 2-D transform LiDAR map -> orthophoto (the georeference offset), or reuse one
  3. fit camera -> orthophoto with the chamfer registration of fit_cam_H.py; the map paint ROI is
     the LiDAR crosswalk outlines moved by that transform, plus optional boxes in aerial metres and
     the green bike box found in the aerial; --lens frees the one-parameter division distortion
  4. score both homographies against both maps, and against each other on the road pixels
Outputs in out_refined/<cam>_fit_ortho<tag>/: H.json (frame = aerial, in LiDAR-origin metres),
overlay_ortho.png, overlay_lidar_on_ortho.png, control_points.jpg, comparison.json, comparison.txt
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
from scipy.optimize import minimize

import fit_cam_H as F

HERE = F.HERE
OUTR = F.OUTR
E0, E1, N0, N1, RES = F.E0, F.E1, F.N0, F.N1, F.RES
LAT0, LON0 = 33.94760, -83.37630
KX, KY = 111320.0 * np.cos(np.radians(LAT0)), 111320.0
RADIUS = 6378137.0
ORTHO_DIR = os.path.join(HERE, "..", "calibration", "ortho")   # county aerial tiles + ortho_index.csv


def load_ortho(no=11):
    with open(os.path.join(ORTHO_DIR, "ortho_index.csv"), encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if int(r["no"]) == no and "junction" not in r["file"])
    img = cv2.imread(os.path.join(ORTHO_DIR, row["file"]))
    return img, float(row["merc_x_left"]), float(row["merc_y_top"]), float(row["merc_m_per_px"])


def local_to_ortho_px(E, N, left, top, res):
    lat = LAT0 + N / KY; lon = LON0 + E / KX
    x = RADIUS * np.radians(lon); y = RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return (x - left) / res, (top - y) / res


def ortho_in_corner_grid(img, left, top, res):
    H, W = int((N1 - N0) / RES), int((E1 - E0) / RES)
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    E = E0 + cols * RES; N = N1 - rows * RES
    mx, my = local_to_ortho_px(E, N, left, top, res)   # float64: absolute Mercator metres lose ~1 m in float32
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)


def rigid(p, th, t, c):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (np.asarray(p, float) - c) @ R.T + c + t


def register_lidar_to_ortho(pts_l, dt_o):
    c = pts_l.mean(0)

    def score(th, t):
        q = F.m2px(rigid(pts_l, th, t, c))
        ok = (q[:, 0] >= 0) & (q[:, 0] < dt_o.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < dt_o.shape[0] - 1)
        d = np.full(len(q), 1.0); d[ok] = np.minimum(dt_o[q[ok, 1].astype(int), q[ok, 0].astype(int)], 1.0)
        return d.mean(), d

    best = (9, 0, np.zeros(2))
    for th in np.radians(np.arange(-3, 3.01, 0.25)):
        for dx in np.arange(-2, 2.01, 0.1):
            for dy in np.arange(-2, 2.01, 0.1):
                s, _ = score(th, np.array([dx, dy]))
                if s < best[0]: best = (s, th, np.array([dx, dy]))
    s0, th, t = best
    for step in (0.02, 0.005):
        for _ in range(3):
            for dth in np.radians([-0.1, 0, 0.1]):
                for dx in (-step, 0, step):
                    for dy in (-step, 0, step):
                        s, _ = score(th + dth, t + [dx, dy])
                        if s < best[0]: best = (s, th + dth, t + np.array([dx, dy]))
            s0, th, t = best
    s, d = score(th, t)
    return th, t, c, d


def green_box_aerial(oc, region=((-30, 0), (-10, 14))):
    hsv = cv2.cvtColor(oc, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60)).astype(np.uint8)
    (e0, n0), (e1, n1) = region
    r = np.zeros_like(m); cv2.fillPoly(r, [F.m2px([[e0, n1], [e1, n1], [e1, n0], [e0, n0]]).astype(np.int32)], 1); m *= r
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return F.px2m(cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea))))


def paint_mask_aerial(og, polys_m, pad_m=0.8, pct=88):
    """bright paint inside the given map polygons (metres), same percentile rule as the LiDAR mask"""
    roi = np.zeros(og.shape, np.uint8)
    for poly in polys_m:
        cv2.fillPoly(roi, [F.m2px(poly).astype(np.int32)], 1)
    k = int(pad_m / RES) | 1
    roi = cv2.dilate(roi, np.ones((k, k), np.uint8))
    vals = og[(roi > 0) & (og > 0)]
    thr = np.percentile(vals, pct)
    return ((og > thr) & (roi > 0)).astype(np.uint8), roi


def residuals(warp_fn, pi, dt_map, pm, unwarp_fn, dt_img, Ww, Hh):
    q = F.m2px(warp_fn(pi)); ok = (q[:, 0] >= 0) & (q[:, 0] < dt_map.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < dt_map.shape[0] - 1)
    d1 = np.full(len(q), 1.0); d1[ok] = np.minimum(dt_map[q[ok, 1].astype(int), q[ok, 0].astype(int)], 1.0)
    u = unwarp_fn(pm); oku = (u[:, 0] >= 0) & (u[:, 0] < Ww - 1) & (u[:, 1] >= 0) & (u[:, 1] < Hh - 1)
    d2 = np.full(len(u), 150.0); d2[oku] = np.minimum(dt_img[u[oku, 1].astype(int), u[oku, 0].astype(int)], 150.0)
    return dict(img_to_map_median_m=float(np.median(d1)), img_to_map_within15cm=float(np.mean(d1 < 0.15)),
                map_to_img_median_px=float(np.median(d2)), map_to_img_within20px=float(np.mean(d2 < 20)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--lidar-fit", default="cam02_fit_nolam"); ap.add_argument("--site", type=int, default=11)
    ap.add_argument("--offset-from", default=None, help="reuse lidar_to_ortho from this H.json instead of re-registering (one frame for all cameras)")
    ap.add_argument("--lens", action="store_true", help="free the one-parameter division distortion"); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    spec = json.load(open(args.spec)); cam = spec["cam"]
    out = os.path.join(OUTR, f"{cam}_fit_ortho{args.tag}"); os.makedirs(out, exist_ok=True)
    im = cv2.imread(os.path.join(HERE, spec["image"])); Hh, Ww = im.shape[:2]
    corner = np.load(os.path.join(OUTR, "corner_img.npy"))
    ortho, left, top, res = load_ortho(args.site)
    oc = ortho_in_corner_grid(ortho, left, top, res)
    cv2.imwrite(os.path.join(out, "ortho_corner_grid.png"), oc)
    og = cv2.cvtColor(oc, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rng = np.random.default_rng(0)

    # camera paint and LiDAR paint (LiDAR frame)
    mi = F.paint_mask_image(im, spec["rois"], spec.get("vmin", 200), spec.get("smax", 55))
    mm_l, _ = F.paint_mask_map(corner, spec["map_ladders"], spec.get("map_boxes", []), spec.get("map_pad", 0.8), spec.get("map_pct", 88))
    dt_l = cv2.distanceTransform(1 - mm_l, cv2.DIST_L2, 5) * RES
    dt_img = cv2.distanceTransform(1 - mi, cv2.DIST_L2, 5)
    pi = np.column_stack(np.where(mi > 0))[:, ::-1].astype(np.float32); pi = pi[rng.choice(len(pi), min(len(pi), 8000), replace=False)]
    pm_l = F.px2m(np.column_stack(np.where(mm_l > 0))[:, ::-1]); pm_l = pm_l[rng.choice(len(pm_l), min(len(pm_l), 8000), replace=False)].astype(np.float32)

    # 2. georeference offset LiDAR -> aerial
    if args.offset_from:
        lo_ = json.load(open(args.offset_from))["lidar_to_ortho"]
        th, t, c = np.radians(lo_["rotation_deg"]), np.array([lo_["shift_east_m"], lo_["shift_north_m"]]), np.array(lo_["centre"])
        print(f"map offset reused from {args.offset_from}")
    else:
        mm_o0, _ = F.paint_mask_map(og, spec["map_ladders"], spec.get("map_boxes", []), spec.get("map_pad", 0.8), spec.get("map_pct", 88))
        th, t, c, _ = register_lidar_to_ortho(pm_l, cv2.distanceTransform(1 - mm_o0, cv2.DIST_L2, 5) * RES)
    T = lambda p: rigid(p, th, t, c)

    # aerial paint ROI: LiDAR crosswalk outlines moved by T, boxes given in aerial metres, the green bike box
    polys = [T(F.LADDERS[n]["corners"]) for n in spec["map_ladders"]]
    for e0, n0, e1, n1 in spec.get("map_boxes_aerial", []): polys.append([[e0, n1], [e1, n1], [e1, n0], [e0, n0]])
    if spec.get("aerial_green_box"): polys.append(green_box_aerial(oc))
    mm_o, _ = paint_mask_aerial(og, polys, spec.get("map_pad", 0.8), spec.get("map_pct", 88))
    dt_o = cv2.distanceTransform(1 - mm_o, cv2.DIST_L2, 5) * RES
    pm_o = F.px2m(np.column_stack(np.where(mm_o > 0))[:, ::-1]); pm_o = pm_o[rng.choice(len(pm_o), min(len(pm_o), 8000), replace=False)].astype(np.float32)
    q = F.m2px(T(pm_l)); ok = (q[:, 0] >= 0) & (q[:, 0] < dt_o.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < dt_o.shape[0] - 1)
    d_reg = np.full(len(q), 1.0); d_reg[ok] = np.minimum(dt_o[q[ok, 1].astype(int), q[ok, 0].astype(int)], 1.0)
    print(f"paint px: image {int(mi.sum())}, lidar map {int(mm_l.sum())}, aerial map {int(mm_o.sum())} in {len(polys)} ROIs")
    print(f"LiDAR -> aerial rigid: rotation {np.degrees(th):+.2f} deg, shift east {t[0]:+.2f} m north {t[1]:+.2f} m, "
          f"LiDAR paint on aerial paint: median {np.median(d_reg):.3f} m, within 15 cm {np.mean(d_reg < 0.15) * 100:.0f}%")

    # 3. fit camera -> aerial (anchors initialised from the LiDAR-frame init carried over by T)
    init_img = np.array(spec["init_img"], np.float32); init_map = T(np.array(spec["init_map"], np.float32)).astype(np.float32)
    model = F.Model(init_img[:4], Ww / 2, Hh / 2, np.hypot(Ww, Hh))
    H0, _ = cv2.findHomography(init_img, init_map, 0)
    p_ref = cv2.perspectiveTransform(init_img[:4].reshape(1, -1, 2), H0)[0]
    pq = p_ref[[0, 1, 3, 2]]; cr = lambda a, b: a[0] * b[1] - a[1] * b[0]
    ref_sign = np.sign([cr(pq[(k + 1) % 4] - pq[k], pq[(k + 2) % 4] - pq[(k + 1) % 4]) for k in range(4)]).tolist(); ref_area = abs(F.area_of(pq))
    R_ANCH = spec.get("anchor_radius", 3.0)

    def cost(params):
        if not args.lens: params = np.concatenate([params[:8], [0.0]])
        if abs(params[8]) > 0.8: return 10.0
        w = model.warp(pi, params); q = F.m2px(w)
        okq = (q[:, 0] >= 0) & (q[:, 0] < mm_o.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < mm_o.shape[0] - 1)
        d1 = np.full(len(q), 1.0); d1[okq] = np.minimum(dt_o[q[okq, 1].astype(int), q[okq, 0].astype(int)], 1.0)
        u = model.unwarp(pm_o, params)
        oku = (u[:, 0] >= 0) & (u[:, 0] < Ww - 1) & (u[:, 1] >= 0) & (u[:, 1] < Hh - 1)
        d2 = np.full(len(u), 150.0); d2[oku] = np.minimum(dt_img[u[oku, 1].astype(int), u[oku, 0].astype(int)], 150.0)
        dev = np.hypot(*(params[:8].reshape(4, 2) - p_ref).T); pen = np.sum(np.maximum(dev - R_ANCH, 0.0) ** 2)
        qd = params[:8].reshape(4, 2)[[0, 1, 3, 2]]; crs = [cr(qd[(k + 1) % 4] - qd[k], qd[(k + 2) % 4] - qd[(k + 1) % 4]) for k in range(4)]
        if np.sign(crs).tolist() != ref_sign or abs(F.area_of(qd)) < 0.4 * ref_area or abs(F.area_of(qd)) > 2.5 * ref_area: return 10.0
        return d1.mean() + d2.mean() / 150.0 + pen

    p0 = np.concatenate([p_ref.ravel(), [0.0]]); bs = (cost(p0), p0); print(f"init cost {bs[0]:.4f}")
    for rnd in range(3):
        r = minimize(cost, bs[1], method="Powell", options=dict(xtol=1e-3, ftol=1e-6, maxfev=8000))
        r = minimize(cost, r.x, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-7, maxfev=8000))
        if r.fun < bs[0] - 1e-7: bs = (r.fun, r.x)
    p = bs[1].copy()
    if not args.lens: p[8] = 0.0
    H_o, lam = model.H(p); print(f"refined cost {bs[0]:.4f}  lambda {lam:+.4f}")
    json.dump(dict(cam=cam, H=H_o.tolist(), lambda_=float(lam), centre=[Ww / 2, Hh / 2], scale=float(np.hypot(Ww, Hh)),
                   anchors_img=init_img[:4].tolist(), anchors_map=p[:8].reshape(4, 2).tolist(),
                   note="camera -> aerial paint; map = undist(pixel) via division model (lambda_, centre, scale), then H; frame = aerial, "
                        "east/north metres from 33.94760,-83.37630; the LiDAR map sits lidar_to_ortho away from it",
                   lidar_to_ortho=dict(rotation_deg=float(np.degrees(th)), shift_east_m=float(t[0]), shift_north_m=float(t[1]), centre=c.tolist())),
              open(os.path.join(out, "H.json"), "w"), indent=1)

    # 4. scores (distortion-aware warps for the new fit)
    warpL = lambda pts: model.warp(pts, p); unwarpL = lambda pts: model.unwarp(pts, p)
    Hl = np.array(json.load(open(os.path.join(OUTR, args.lidar_fit, "H.json")))["H"])
    warp = lambda Hm: (lambda pts: cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(1, -1, 2), Hm)[0])
    unwarp = lambda Hm: (lambda pts: cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(1, -1, 2), np.linalg.inv(Hm))[0])
    Rm = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]); Cm = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]]); Tm = np.array([[1, 0, c[0] + t[0]], [0, 1, c[1] + t[1]], [0, 0, 1]])
    Hl_o = Tm @ Rm @ Cm @ Hl
    Tinv = np.linalg.inv(Tm @ Rm @ Cm)
    res_ = {
        "lidar_H_vs_lidar_paint": residuals(warp(Hl), pi, dt_l, pm_l, unwarp(Hl), dt_img, Ww, Hh),
        "lidar_H_vs_aerial_paint_after_offset": residuals(warp(Hl_o), pi, dt_o, pm_o, unwarp(Hl_o), dt_img, Ww, Hh),
        "aerial_H_vs_aerial_paint": residuals(warpL, pi, dt_o, pm_o, unwarpL, dt_img, Ww, Hh),
        "aerial_H_vs_lidar_paint_after_offset": residuals(lambda pts: (Tinv @ np.c_[warpL(pts), np.ones(len(pts))].T).T[:, :2], pi, dt_l, pm_l,
                                                          lambda pts: unwarpL((Tm @ Rm @ Cm @ np.c_[np.asarray(pts, float), np.ones(len(pts))].T).T[:, :2]), dt_img, Ww, Hh),
    }
    # per-ROI residual of the new fit (map paint -> image), so the far crosswalk can be judged on its own
    per_roi = {}
    for name, poly in zip(spec["map_ladders"] + [f"box{k}" for k in range(len(polys) - len(spec["map_ladders"]))], polys):
        mk = np.zeros(og.shape, np.uint8); cv2.fillPoly(mk, [F.m2px(poly).astype(np.int32)], 1); mk = cv2.dilate(mk, np.ones((21, 21), np.uint8)) * mm_o
        pts = F.px2m(np.column_stack(np.where(mk > 0))[:, ::-1]).astype(np.float32)
        if len(pts) < 20: continue
        u = unwarpL(pts); oku = (u[:, 0] >= 0) & (u[:, 0] < Ww - 1) & (u[:, 1] >= 0) & (u[:, 1] < Hh - 1)
        d2 = np.full(len(u), 150.0); d2[oku] = np.minimum(dt_img[u[oku, 1].astype(int), u[oku, 0].astype(int)], 150.0)
        # express in metres: pixel -> metre scale at that ROI from the homography jacobian (median over points)
        q = F.m2px(warpL(u + [1, 0])); q0 = F.m2px(warpL(u)); mpp = np.median(np.hypot(*(q - q0).T)) * RES
        per_roi[name] = dict(n=int(len(pts)), median_px=float(np.median(d2)), median_m=float(np.median(d2) * mpp), within20px=float(np.mean(d2 < 20)))
    roi = np.zeros((Hh, Ww), np.uint8)
    for poly in spec["rois"]: cv2.fillPoly(roi, [np.array(poly, np.int32)], 1)
    gy, gx = np.mgrid[0:Hh:25, 0:Ww:25]; grid = np.stack([gx.ravel(), gy.ravel()], 1); grid = grid[roi[grid[:, 1], grid[:, 0]] > 0].astype(np.float32)
    a_ = warp(Hl_o)(grid); b_ = warpL(grid); dd = np.hypot(*(a_ - b_).T)
    bands = {"near (v>1400)": grid[:, 1] > 1400, "mid (900-1400)": (grid[:, 1] > 900) & (grid[:, 1] <= 1400), "far (v<=900)": grid[:, 1] <= 900}
    disagree = {k: dict(n=int(m.sum()), median_m=float(np.median(dd[m])), p90_m=float(np.percentile(dd[m], 90))) for k, m in bands.items() if m.sum()}
    disagree["all"] = dict(n=int(len(dd)), median_m=float(np.median(dd)), p90_m=float(np.percentile(dd, 90)))
    rep = dict(lidar_to_ortho=dict(rotation_deg=float(np.degrees(th)), shift_east_m=float(t[0]), shift_north_m=float(t[1]),
                                   paint_residual_median_m=float(np.median(d_reg)), paint_within15cm=float(np.mean(d_reg < 0.15))),
               lens_lambda=float(lam), residuals=res_, per_roi_map_to_image=per_roi, homography_disagreement_m=disagree)
    json.dump(rep, open(os.path.join(out, "comparison.json"), "w"), indent=1)
    lines = [f"{cam}: camera -> aerial (2025 EagleView, 6 cm) vs camera -> LiDAR; lens lambda {lam:+.4f}", "",
             f"LiDAR map -> aerial rigid offset: {np.degrees(th):+.2f} deg, east {t[0]:+.2f} m, north {t[1]:+.2f} m; LiDAR paint on aerial paint: median {np.median(d_reg) * 100:.0f} cm, {np.mean(d_reg < 0.15) * 100:.0f}% within 15 cm", ""]
    for k, v in res_.items():
        lines.append(f"{k:40s} image->map median {v['img_to_map_median_m'] * 100:4.0f} cm, {v['img_to_map_within15cm'] * 100:3.0f}% <15 cm | map->image median {v['map_to_img_median_px']:4.1f} px, {v['map_to_img_within20px'] * 100:3.0f}% <20 px")
    lines.append(""); lines.append("new fit, aerial paint -> image, per ROI:")
    for k, v in per_roi.items(): lines.append(f"  {k:8s} n={v['n']:5d}  median {v['median_px']:5.1f} px = {v['median_m'] * 100:4.0f} cm   {v['within20px'] * 100:3.0f}% <20 px")
    lines.append(""); lines.append("where the LiDAR-fit and aerial-fit homographies put the same road pixel:")
    for k, v in disagree.items(): lines.append(f"  {k:16s} n={v['n']:4d}  median {v['median_m'] * 100:4.0f} cm   90% {v['p90_m'] * 100:4.0f} cm")
    open(os.path.join(out, "comparison.txt"), "w").write("\n".join(lines)); print("\n".join(lines))

    # figures
    map_from_m = np.array([[1 / RES, 0, -E0 / RES], [0, -1 / RES, N1 / RES], [0, 0, 1]])
    if abs(lam) > 1e-6:
        xs, ys = np.meshgrid(np.arange(Ww, dtype=np.float32), np.arange(Hh, dtype=np.float32))
        g = np.stack([xs.ravel(), ys.ravel()], 1); qg = (g - [Ww / 2, Hh / 2]) / model.sc; r2 = np.sum(qg ** 2, 1, keepdims=True); s = np.ones_like(r2)
        for _ in range(4): s = 1 + lam * r2 * s * s
        srcp = qg * s * model.sc + [Ww / 2, Hh / 2]
        und = cv2.remap(im, srcp[:, 0].reshape(Hh, Ww).astype(np.float32), srcp[:, 1].reshape(Hh, Ww).astype(np.float32), cv2.INTER_LINEAR)
    else:
        und = im
    for name, Hm, src in (("overlay_ortho.png", H_o, und), ("overlay_lidar_on_ortho.png", Hl_o, im)):
        warped = cv2.warpPerspective(src, map_from_m @ Hm, (oc.shape[1], oc.shape[0]))
        ov = cv2.addWeighted(oc, 0.5, warped, 0.7, 0)
        for poly in polys: cv2.polylines(ov, [F.m2px(poly).astype(np.int32)], True, (0, 255, 255), 1)
        cv2.putText(ov, name.replace(".png", "") + " (yellow = paint ROIs on the aerial)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out, name), ov)
    ci = im.copy()
    for poly in polys:
        u = unwarpL(np.asarray(poly, np.float32))
        if np.all((u[:, 0] > -500) & (u[:, 0] < Ww + 500) & (u[:, 1] > -500) & (u[:, 1] < Hh + 500)): cv2.polylines(ci, [u.astype(np.int32)], True, (0, 255, 255), 3)
    w = warpL(pi); q = F.m2px(w); okq = (q[:, 0] >= 0) & (q[:, 0] < mm_o.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < mm_o.shape[0] - 1)
    d1 = np.full(len(q), 1.0); d1[okq] = dt_o[q[okq, 1].astype(int), q[okq, 0].astype(int)]
    for (uu, vv), d in zip(pi[::4], d1[::4]):
        cv2.circle(ci, (int(uu), int(vv)), 2, (0, 200, 0) if d < 0.15 else ((0, 165, 255) if d < 0.4 else (255, 0, 255)), -1)
    cv2.putText(ci, f"{cam} -> aerial: yellow = paint ROIs projected with the fitted H; paint dots: green <15 cm, orange <40 cm, magenta worse", (30, Hh - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out, "control_points.jpg"), ci, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(os.path.join(out, "cp_small.jpg"), cv2.resize(ci, (1600, 900)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out)


if __name__ == "__main__":
    main()
