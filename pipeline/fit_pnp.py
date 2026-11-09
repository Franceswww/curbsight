"""
fit_pnp.py - replace the flat-plane homography by a real camera model on a LiDAR ground surface.

    python fit_pnp.py --pairs control_points/cam02_manual_20261020.json --compare cam02_fit_manual

Ground model: USGS 2019 airborne LiDAR ground returns (usgs_2019/corner_ground_enu.npz), gridded
at 1 m and smoothed. Control points: the hand-picked image<->aerial pairs, lifted to 3-D with the
ground model. Camera: pinhole with one radial term; focal length is scanned (near-coplanar points
cannot fix it on their own), pose from solvePnP on the RANSAC inliers. A pixel is then put on the
ground by casting its ray onto the surface (iterative), so sidewalks and slopes are honoured.
Outputs in out_refined/<cam>_fit_pnp/: cam.json (K, dist, R, t, DEM reference), residuals.csv,
comparison.txt, points.jpg
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import fit_cam_H as F
import fit_manual as M
import fit_cam_H_ortho as O

OUTR = F.OUTR
DEM_E0, DEM_E1, DEM_N0, DEM_N1, DEM_RES = -70.0, 50.0, -60.0, 50.0, 1.0


class Ground:
    """ground surface z(E, N) by bilinear lookup. src 'usgs' = 1 m DEM from the USGS airborne ground
    returns (smoothed); src 'fused' = usgs_2019/fused_ground_enu.npz, 0.25 m, car LiDAR detail on the USGS base"""
    def __init__(self, src="usgs"):
        self.src = src
        if src == "fused":
            u = np.load(os.path.join(F.HERE, "..", "calibration", "ground", "fused_ground_enu.npz"))
            self.res = 0.25; W_, H_ = int((DEM_E1 - DEM_E0) / self.res), int((DEM_N1 - DEM_N0) / self.res)
            self.z = u["Z"].reshape(H_, W_); return
        self.res = DEM_RES
        u = np.load(os.path.join(F.HERE, "..", "calibration", "ground", "corner_ground_enu.npz"))
        E, N, Z = u["E"], u["N"], u["Z"]
        k = (E > DEM_E0 - 5) & (E < DEM_E1 + 5) & (N > DEM_N0 - 5) & (N < DEM_N1 + 5)
        ge, gn = np.meshgrid(np.arange(DEM_E0, DEM_E1, DEM_RES), np.arange(DEM_N0, DEM_N1, DEM_RES))
        z = griddata((E[k], N[k]), Z[k], (ge, gn), method="linear")
        zn = griddata((E[k], N[k]), Z[k], (ge, gn), method="nearest"); z[np.isnan(z)] = zn[np.isnan(z)]
        self.z = gaussian_filter(z, 1.0); self.ge, self.gn = ge, gn

    def __call__(self, E, N):
        E = np.asarray(E, float); N = np.asarray(N, float)
        x = np.clip((E - DEM_E0) / self.res, 0, self.z.shape[1] - 1.001); y = np.clip((N - DEM_N0) / self.res, 0, self.z.shape[0] - 1.001)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int); fx, fy = x - x0, y - y0
        z = self.z
        return (z[y0, x0] * (1 - fx) * (1 - fy) + z[y0, x0 + 1] * fx * (1 - fy) + z[y0 + 1, x0] * (1 - fx) * fy + z[y0 + 1, x0 + 1] * fx * fy)


class Camera:
    def __init__(self, K, dist, rvec, tvec):
        self.K, self.dist, self.rvec, self.tvec = K, dist, rvec, tvec
        self.R, _ = cv2.Rodrigues(rvec); self.C = (-self.R.T @ tvec).ravel()      # camera centre, world

    def project(self, P):
        p, _ = cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 1, 3), self.rvec, self.tvec, self.K, self.dist)
        return p.reshape(-1, 2)

    def rays(self, uv):
        q = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        d = np.c_[q, np.ones(len(q))] @ self.R                                     # R^T * [x y 1]
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_ground(self, uv, ground, iters=12):
        d = self.rays(uv); C = self.C
        z0 = ground(C[0], C[1]) - 0.0
        t = (z0 - C[2]) / d[:, 2]                                                  # flat start at the camera-foot height
        for _ in range(iters):
            P = C + t[:, None] * d; zg = ground(P[:, 0], P[:, 1])
            t = (zg - C[2]) / d[:, 2]
        P = C + t[:, None] * d
        return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--compare", default=None); ap.add_argument("--tag", default="")
    ap.add_argument("--fix-centre-from", default=None, help="comma-separated fit folders (e.g. cam02_fit_pnp,cam03_fit_pnp): fix the camera centre at their mean and solve orientation, f, k1 only")
    ap.add_argument("--ground-src", default="usgs", choices=["usgs", "fused"])
    ap.add_argument("--k2", action="store_true", help="free a second radial distortion term (wide lenses)")
    ap.add_argument("--fix-focal", type=float, default=None, help="hold the focal length at this many pixels (e.g. the datasheet value)")
    ap.add_argument("--fix-focal-from", default=None, help="comma-separated fit folders: hold the focal length at their mean (same camera model on the same pole)")
    ap.add_argument("--init-from", default=None, help="fit folder whose cam.json gives the starting pose/f/k1 (skips the focal-length scan)")
    a = ap.parse_args()
    F_fixed = None
    if a.fix_focal is not None:
        F_fixed = float(a.fix_focal); print(f"focal length fixed at {F_fixed:.0f} px (given)")
    elif a.fix_focal_from:
        F_fixed = float(np.mean([json.load(open(os.path.join(OUTR, f_, "cam.json")))["f_px"] for f_ in a.fix_focal_from.split(",")]))
        print(f"focal length fixed at {F_fixed:.0f} px (mean of {a.fix_focal_from})")
    C_fixed = None
    if a.fix_centre_from:
        C_fixed = np.mean([json.load(open(os.path.join(OUTR, f_, "cam.json")))["centre"] for f_ in a.fix_centre_from.split(",")], axis=0)
        print(f"camera centre fixed at E {C_fixed[0]:+.2f} N {C_fixed[1]:+.2f} Z {C_fixed[2]:.2f} (mean of {a.fix_centre_from})")
    P = json.load(open(a.pairs)); cam = P["camera"]
    out = os.path.join(OUTR, f"{cam}_fit_pnp{a.tag}"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(O.ORTHO_DIR, "ortho_index.csv"), encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["file"] == P["aerial"])
    g = {k: float(row[k]) for k in ("merc_x_left", "merc_y_top", "merc_m_per_px")}
    uv = np.array([[p["u"], p["v"]] for p in P["pairs"]], np.float64)
    en = M.aerial_px_to_m([p["x"] for p in P["pairs"]], [p["y"] for p in P["pairs"]], g)
    ground = Ground(a.ground_src); X = np.c_[en, ground(en[:, 0], en[:, 1])]
    im = cv2.imread(os.path.join(F.HERE, "camera_frames", P["image"])); Hh, Ww = im.shape[:2]
    print(f"{cam}: {len(uv)} pairs; ground heights of the picked points span {X[:, 2].min():.2f}..{X[:, 2].max():.2f} m")

    from scipy.optimize import least_squares

    def make(params):
        f, k1 = params[6], params[7]; k2 = params[8] if len(params) > 8 else 0.0
        if F_fixed is not None: f = F_fixed
        K = np.array([[f, 0, Ww / 2], [0, f, Hh / 2], [0, 0, 1]], float); dist = np.array([k1, k2, 0, 0, 0], float)
        rvec = np.array(params[:3]); tvec = np.array(params[3:6])
        if C_fixed is not None:
            R, _ = cv2.Rodrigues(rvec); tvec = (-R @ C_fixed).ravel()
        return Camera(K, dist, rvec, tvec)

    def ground_err(c):
        return np.hypot(*(c.to_ground(uv, ground)[:, :2] - en).T)

    # 1. initial pose per focal length from a plain PnP on all points (robust: trimmed to the best half, twice)
    best = None
    if a.init_from:
        j0 = json.load(open(os.path.join(OUTR, a.init_from, "cam.json")))
        best = (0.0, np.r_[j0["rvec"], j0["tvec"], j0["f_px"], j0["k1"]] if not a.k2 else np.r_[j0["rvec"], j0["tvec"], j0["f_px"], j0["k1"], j0.get("k2", 0.0)]); print(f"starting from {a.init_from}")
    for f in (np.arange(0.25 * Ww, 0.80 * Ww, 0.025 * Ww) if best is None else []):
        K = np.array([[f, 0, Ww / 2], [0, f, Hh / 2], [0, 0, 1]], float); dist = np.zeros(5)
        idx = np.arange(len(uv))
        for _ in range(3):
            ok, rvec, tvec = cv2.solvePnP(X[idx].reshape(-1, 1, 3), uv[idx].reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok: break
            c = Camera(K, dist, rvec, tvec); err = ground_err(c)
            idx = np.argsort(err)[:max(len(uv) * 2 // 3, 8)]
        if not ok: continue
        h = c.C[2] - ground(c.C[0], c.C[1])
        if h < 2 or h > 15: continue
        s = np.median(err)
        if best is None or s < best[0]: best = (s, np.r_[rvec.ravel(), tvec.ravel(), f, 0.0])
    p0 = best[1]
    if a.k2: p0 = np.r_[p0, 0.0]

    # 2. refine pose, focal length and k1 together on the ground residuals of ALL points, robust loss
    def resid(params):
        c = make(params); G = c.to_ground(uv, ground)
        return (G[:, :2] - en).ravel()

    if C_fixed is not None:                                   # start from the fixed centre and keep the translation out of the optimisation
        R0, _ = cv2.Rodrigues(p0[:3]); p0[3:6] = (-R0 @ C_fixed).ravel()
        resid_fixed = lambda q: resid(np.r_[q[:3], p0[3:6], q[3:]])
        r = least_squares(resid_fixed, np.r_[p0[:3], p0[6:]], loss="soft_l1", f_scale=0.25, x_scale=np.r_[0.01, 0.01, 0.01, 100, 0.05] if not a.k2 else np.r_[0.01, 0.01, 0.01, 100, 0.05, 0.05], max_nfev=600)
        r.x = np.r_[r.x[:3], p0[3:6], r.x[3:]]
    else:
        r = least_squares(resid, p0, loss="soft_l1", f_scale=0.25, x_scale=np.r_[0.01, 0.01, 0.01, 0.5, 0.5, 0.5, 100, 0.05][:len(p0)] if not a.k2 else np.r_[0.01, 0.01, 0.01, 0.5, 0.5, 0.5, 100, 0.05, 0.05], max_nfev=600)
    c = make(r.x); f, k1 = (F_fixed if F_fixed is not None else r.x[6]), r.x[7]; k2 = float(r.x[8]) if a.k2 else 0.0; err = ground_err(c); ii = np.where(err < 0.5)[0]
    v_near, v_far = 0.65 * Hh, 0.42 * Hh          # image rows, not metres: scales with the frame height
    near = uv[:, 1] > v_near; far = uv[:, 1] < v_far; mid = ~near & ~far
    print(f"best: f {f:.0f} px, k1 {k1:+.2f}, k2 {k2:+.3f}, camera at E {c.C[0]:+.1f} N {c.C[1]:+.1f}, {c.C[2] - ground(c.C[0], c.C[1]):.1f} m above ground, inliers {len(ii)}")
    print(f"ground residual (m): median {np.median(err):.2f}, 90% {np.percentile(err, 90):.2f}; near {np.median(err[near]):.2f}, mid {np.median(err[mid]):.2f}, far {np.median(err[far]):.2f}")
    json.dump(dict(cam=cam, K=c.K.tolist(), dist=c.dist.tolist(), rvec=c.rvec.ravel().tolist(), tvec=c.tvec.ravel().tolist(), centre=c.C.tolist(), f_px=float(f), k1=float(k1), k2=k2,
                   n_pairs=int(len(uv)), n_inliers=int(len(ii)), residual_median_m=float(np.median(err)), residual_p90_m=float(np.percentile(err, 90)),
                   ground_src=a.ground_src, dem=dict(source="USGS 3DEP GA_Statewide_2018 ground returns" + (" + car LiDAR 2026-09-01 detail" if a.ground_src == "fused" else ""), grid_m=ground.res, extent=[DEM_E0, DEM_E1, DEM_N0, DEM_N1]),
                   note="pinhole + k1 on the USGS ground surface; world = east/north metres from 33.94760,-83.37630, z NAVD88 m; foot points by ray casting (fit_pnp.Camera.to_ground)"),
              open(os.path.join(out, "cam.json"), "w"), indent=1)
    with open(os.path.join(out, "residuals.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["i", "u", "v", "E", "N", "Z", "err_m", "inlier"])
        for i, ((u, v), x, d) in enumerate(zip(uv, X, err)): w.writerow([i + 1, u, v, round(x[0], 3), round(x[1], 3), round(x[2], 3), round(d, 3), int(i in set(ii.tolist()))])
    lines = [f"{cam}: camera model on the USGS ground surface. f {f:.0f} px, k1 {k1:+.2f}, camera {c.C[2] - ground(c.C[0], c.C[1]):.1f} m above ground, {len(ii)}/{len(uv)} inliers",
             f"  residual on the picked points: median {np.median(err) * 100:.0f} cm, 90% {np.percentile(err, 90) * 100:.0f} cm, max {err.max() * 100:.0f} cm",
             f"  near (v>{v_near:.0f}) {np.median(err[near]) * 100 if near.any() else float('nan'):.0f} cm (n={near.sum()}), mid {np.median(err[mid]) * 100 if mid.any() else float('nan'):.0f} cm (n={mid.sum()}), far (v<{v_far:.0f}) {np.median(err[far]) * 100 if far.any() else float('nan'):.0f} cm (n={far.sum()})"]
    if a.compare:
        J = json.load(open(os.path.join(OUTR, a.compare, "H.json"))); Hc = np.array(J["H"])
        ec = np.hypot(*(cv2.perspectiveTransform(uv.reshape(1, -1, 2), Hc)[0] - en).T)
        lines += ["", f"  flat-plane homography ({a.compare}) on the same points: median {np.median(ec) * 100:.0f} cm, 90% {np.percentile(ec, 90) * 100:.0f} cm, max {ec.max() * 100:.0f} cm",
                  f"    near {np.median(ec[near]) * 100:.0f} cm, mid {np.median(ec[mid]) * 100:.0f} cm, far {np.median(ec[far]) * 100:.0f} cm"]
        hi = X[:, 2] - np.median(X[:, 2]); up = hi > 0.10
        if up.any(): lines.append(f"  points standing >10 cm above the road level (n={up.sum()}): flat plane {np.median(ec[up]) * 100:.0f} cm -> ground surface {np.median(err[up]) * 100:.0f} cm")
        worst = np.argsort(-ec)[:8]
        lines.append("  the 8 worst points of the flat plane, flat -> surface (cm): " + ", ".join(f"#{i + 1} {ec[i] * 100:.0f}->{err[i] * 100:.0f}" for i in worst))
    open(os.path.join(out, "comparison.txt"), "w").write("\n".join(lines)); print("\n".join(lines))
    ci = im.copy(); back = c.project(np.c_[en, ground(en[:, 0], en[:, 1])])
    for i, ((u, v), (bu, bv), d) in enumerate(zip(uv, back, err)):
        col = (0, 200, 0) if d < 0.15 else ((0, 165, 255) if d < 0.4 else (0, 0, 255))
        cv2.circle(ci, (int(u), int(v)), 10, col, 3); cv2.line(ci, (int(u), int(v)), (int(bu), int(bv)), col, 2); cv2.putText(ci, f"{i + 1}", (int(u) + 12, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    cv2.putText(ci, f"{cam}: camera model on the LiDAR ground; line = where the aerial point lands; green <15 cm, orange <40 cm, red worse", (30, Hh - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out, "points.jpg"), ci, [cv2.IMWRITE_JPEG_QUALITY, 88]); cv2.imwrite(os.path.join(out, "points_small.jpg"), cv2.resize(ci, (1600, 900)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out)


if __name__ == "__main__":
    main()
