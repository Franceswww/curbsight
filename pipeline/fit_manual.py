"""
fit_manual.py - camera -> aerial homography from hand-picked point pairs (point picker page).

    python fit_manual.py --pairs control_points/cam02_manual_20261020.json --compare cam02_fit_ortho

Aerial clicks are pixels of maps/ortho/11_SanfordatCedarjunction_2025.png; they are turned into
metres in the aerial frame (east/north from 33.94760,-83.37630, the frame of the *_fit_ortho H's)
through the image's Web-Mercator georeference. H is solved with RANSAC (0.25 m) and refined on the
inliers; optionally with the one-parameter division lens model. Outputs in
out_refined/<cam>_fit_manual/: H.json (same format as the other fits), residuals.csv, points.jpg
(camera frame with the picked points and their residual arrows), overlay.jpg (frame on the aerial),
comparison.txt (vs the paint-registration H on the same points and on a road grid).
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

import fit_cam_H as F
import fit_cam_H_ortho as O

OUTR = F.OUTR


def aerial_px_to_m(x, y, g):
    mx = g["merc_x_left"] + np.asarray(x, float) * g["merc_m_per_px"]; my = g["merc_y_top"] - np.asarray(y, float) * g["merc_m_per_px"]
    lon = np.degrees(mx / O.RADIUS); lat = np.degrees(2 * np.arctan(np.exp(my / O.RADIUS)) - np.pi / 2)
    return np.stack([(lon - O.LON0) * O.KX, (lat - O.LAT0) * O.KY], 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--compare", default=None); ap.add_argument("--lens", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    P = json.load(open(a.pairs)); cam = P["camera"]
    out = os.path.join(OUTR, f"{cam}_fit_manual{a.tag}"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(O.ORTHO_DIR, "ortho_index.csv"), encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["file"] == P["aerial"])
    g = {k: float(row[k]) for k in ("merc_x_left", "merc_y_top", "merc_m_per_px")}
    uv = np.array([[p["u"], p["v"]] for p in P["pairs"]], np.float64)
    en = aerial_px_to_m([p["x"] for p in P["pairs"]], [p["y"] for p in P["pairs"]], g)
    im = cv2.imread(os.path.join(F.HERE, "camera_frames", P["image"])); Hh, Ww = im.shape[:2]
    cen = np.array([Ww / 2, Hh / 2]); sc = np.hypot(Ww, Hh)

    def undist(p, lam):
        q = (p - cen) / sc; r2 = np.sum(q ** 2, 1, keepdims=True); return q / (1 + lam * r2) * sc + cen

    def project(H, lam, p):
        q = undist(p, lam); return cv2.perspectiveTransform(q.reshape(1, -1, 2).astype(np.float64), H)[0]

    H0, mask = cv2.findHomography(uv.astype(np.float32), en.astype(np.float32), cv2.RANSAC, 0.25)
    inl = mask.ravel() > 0
    print(f"{cam}: {len(uv)} pairs, RANSAC inliers {inl.sum()}")

    def resid(params):
        H = np.append(params[:8], 1.0).reshape(3, 3); lam = params[8] if a.lens else 0.0
        return (project(H, lam, uv[inl]) - en[inl]).ravel()

    p0 = np.append((H0 / H0[2, 2]).ravel()[:8], 0.0)
    r = least_squares(resid, p0, loss="soft_l1", f_scale=0.3)
    H = np.append(r.x[:8], 1.0).reshape(3, 3); lam = float(r.x[8]) if a.lens else 0.0
    err = np.hypot(*(project(H, lam, uv) - en).T)
    near = uv[:, 1] > 1400; far = uv[:, 1] < 900; mid = ~near & ~far
    print(f"residuals (m): all median {np.median(err):.2f}, 90% {np.percentile(err, 90):.2f}, max {err.max():.2f}; "
          f"near {np.median(err[near]) if near.any() else float('nan'):.2f} ({near.sum()}), mid {np.median(err[mid]) if mid.any() else float('nan'):.2f} ({mid.sum()}), far {np.median(err[far]) if far.any() else float('nan'):.2f} ({far.sum()})  lambda {lam:+.4f}")
    json.dump(dict(cam=cam, H=H.tolist(), lambda_=lam, centre=cen.tolist(), scale=float(sc), n_pairs=int(len(uv)), n_inliers=int(inl.sum()),
                   residual_median_m=float(np.median(err)), residual_p90_m=float(np.percentile(err, 90)),
                   note="camera -> aerial from hand-picked pairs; frame = aerial, east/north metres from 33.94760,-83.37630 (same frame as *_fit_ortho)",
                   lidar_to_ortho=json.load(open(os.path.join(OUTR, "cam02_fit_ortho", "H.json")))["lidar_to_ortho"]), open(os.path.join(out, "H.json"), "w"), indent=1)
    with open(os.path.join(out, "residuals.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["i", "u", "v", "E", "N", "err_m", "inlier"])
        for i, ((u, v), (e, n), d, k) in enumerate(zip(uv, en, err, inl)): w.writerow([i + 1, u, v, round(e, 3), round(n, 3), round(d, 3), int(k)])

    # compare with the paint-registration H on the same points and on a road grid
    lines = [f"{cam}: hand-picked pairs -> H.  {len(uv)} pairs, {inl.sum()} inliers, lambda {lam:+.4f}",
             f"  residual on the picked points: median {np.median(err) * 100:.0f} cm, 90% {np.percentile(err, 90) * 100:.0f} cm, max {err.max() * 100:.0f} cm",
             f"  near (v>1400) {np.median(err[near]) * 100 if near.any() else 0:.0f} cm (n={near.sum()}), mid {np.median(err[mid]) * 100 if mid.any() else 0:.0f} cm (n={mid.sum()}), far (v<900) {np.median(err[far]) * 100 if far.any() else 0:.0f} cm (n={far.sum()})"]
    if a.compare:
        J = json.load(open(os.path.join(OUTR, a.compare, "H.json"))); Hc = np.array(J["H"]); lc = float(J.get("lambda_", 0))
        ec = np.hypot(*(project(Hc, lc, uv) - en).T)
        lines += ["", f"  paint-registration H ({a.compare}) on the same picked points: median {np.median(ec) * 100:.0f} cm, 90% {np.percentile(ec, 90) * 100:.0f} cm, max {ec.max() * 100:.0f} cm",
                  f"    near {np.median(ec[near]) * 100 if near.any() else 0:.0f} cm, mid {np.median(ec[mid]) * 100 if mid.any() else 0:.0f} cm, far {np.median(ec[far]) * 100 if far.any() else 0:.0f} cm"]
        gy, gx = np.mgrid[300:Hh:60, 0:Ww:60]; grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
        dd = np.hypot(*(project(H, lam, grid) - project(Hc, lc, grid)).T)
        for name, m in (("near (v>1400)", grid[:, 1] > 1400), ("mid", (grid[:, 1] > 900) & (grid[:, 1] <= 1400)), ("far (v<=900)", grid[:, 1] <= 900)):
            lines.append(f"  the two H's differ on the road grid, {name:14s}: median {np.median(dd[m]) * 100:.0f} cm, 90% {np.percentile(dd[m], 90) * 100:.0f} cm")
    open(os.path.join(out, "comparison.txt"), "w").write("\n".join(lines)); print("\n".join(lines))

    # figures
    ci = im.copy(); back = project(H, lam, uv)
    Hinv_pts = lambda m: cv2.perspectiveTransform(np.asarray(m, np.float64).reshape(1, -1, 2), np.linalg.inv(H))[0]
    for i, ((u, v), (e, n), d, k) in enumerate(zip(uv, en, err, inl)):
        col = (0, 200, 0) if d < 0.15 else ((0, 165, 255) if d < 0.4 else (0, 0, 255))
        tgt = Hinv_pts([[e, n]])[0]
        cv2.circle(ci, (int(u), int(v)), 10, col, 3); cv2.line(ci, (int(u), int(v)), (int(tgt[0]), int(tgt[1])), col, 2)
        cv2.putText(ci, f"{i + 1}", (int(u) + 12, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    cv2.putText(ci, f"{cam}: picked points, line = where the aerial point lands with the fitted H; green <15 cm, orange <40 cm, red worse", (30, Hh - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out, "points.jpg"), ci, [cv2.IMWRITE_JPEG_QUALITY, 88]); cv2.imwrite(os.path.join(out, "points_small.jpg"), cv2.resize(ci, (1600, 900)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    oc = cv2.imread(os.path.join(OUTR, "cam02_fit_ortho", "ortho_corner_grid.png"))
    map_from_m = np.array([[1 / F.RES, 0, -F.E0 / F.RES], [0, -1 / F.RES, F.N1 / F.RES], [0, 0, 1]])
    src = im
    if abs(lam) > 1e-9:
        xs, ys = np.meshgrid(np.arange(Ww, dtype=np.float32), np.arange(Hh, dtype=np.float32)); gpts = np.stack([xs.ravel(), ys.ravel()], 1)
        q = (gpts - cen) / sc; r2 = np.sum(q ** 2, 1, keepdims=True); s = np.ones_like(r2)
        for _ in range(4): s = 1 + lam * r2 * s * s
        sp = q * s * sc + cen; src = cv2.remap(im, sp[:, 0].reshape(Hh, Ww).astype(np.float32), sp[:, 1].reshape(Hh, Ww).astype(np.float32), cv2.INTER_LINEAR)
    w = cv2.warpPerspective(src, map_from_m @ H, (oc.shape[1], oc.shape[0])); mask = cv2.warpPerspective(np.full((Hh, Ww), 255, np.uint8), map_from_m @ H, (oc.shape[1], oc.shape[0])) > 0
    ov = oc.copy(); ov[mask] = (0.35 * oc[mask] + 0.65 * w[mask]).astype(np.uint8)
    for (e, n), d in zip(en, err):
        q = F.m2px([[e, n]])[0].astype(int); cv2.circle(ov, tuple(q), 4, (0, 200, 0) if d < 0.15 else ((0, 165, 255) if d < 0.4 else (0, 0, 255)), -1)
    cv2.putText(ov, f"{cam} frame on the aerial with the hand-picked H; dots = picked aerial points", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(out, "overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == "__main__":
    main()
