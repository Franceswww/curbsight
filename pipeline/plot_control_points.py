"""
plot_control_points.py - look at the calibration instead of trusting its residual number.

    python plot_control_points.py --out ../figures

For each camera it writes a side-by-side: the frame the points were clicked on, and the aerial
photograph, with the same point carrying the same number and the same colour in both. Colour is how
far that point lands from where it should once the solved camera projects it onto the ground -
green under 10 cm, red over 50. Then one more picture with all three cameras' points on the aerial
together, which is where the gaps show.

This is the honest way to present a calibration. A median residual is one number and it hides
everything: whether the points cover the whole scene or huddle near the pole, whether the bad ones
are all in one corner, whether a point was simply clicked on the wrong kerb. The pictures show it.
"""
import argparse
import csv
import json
import math
import os

import cv2
import numpy as np

import fit_cam_H as F
import fit_manual as M
import fit_pnp
import render_demo as RD

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "calibration")
CAMS = ("cam02_1019", "cam03_1019", "cam04_1019")
CAM_COLOUR = {"cam02_1019": (231, 76, 60), "cam03_1019": (60, 200, 90), "cam04_1019": (60, 120, 240)}
GOOD_M, BAD_M = 0.10, 0.50                                        # the ends of the green-to-red ramp


def residual_colour(e):
    """green at GOOD_M or better, yellow between, red at BAD_M or worse (BGR)"""
    t = min(1.0, max(0.0, (e - GOOD_M) / (BAD_M - GOOD_M)))
    return (60, int(200 - 120 * t), int(60 + 180 * t))


def load(cam):
    """the clicked pairs, as camera pixels, ground metres, and the ground point the camera gives"""
    P = json.load(open(os.path.join(CAL, "control_points", cam + ".json")))
    with open(os.path.join(CAL, "ortho", "ortho_index.csv"), encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["file"] == P["aerial"])
    g = {k: float(row[k]) for k in ("merc_x_left", "merc_y_top", "merc_m_per_px")}
    uv = np.array([[p["u"], p["v"]] for p in P["pairs"]], float)
    truth = M.aerial_px_to_m([p["x"] for p in P["pairs"]], [p["y"] for p in P["pairs"]], g)
    got = RD.Cam(cam, cam + "_fit_pnp").ground(uv)                  # the same call the pipeline makes
    err = np.hypot(*(got - truth).T)
    return P, uv, truth, got, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
    ap.add_argument("--cams", default=",".join(CAMS))
    ap.add_argument("--width", type=int, default=2200, help="width of each written picture")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ext = json.load(open(os.path.join(CAL, "aerial_wide.json")))
    E0, N1, RES = ext["E0"], ext["N1"], ext["res"]
    aerial = cv2.imread(os.path.join(CAL, "aerial_wide.png"))
    mpx = lambda e, n: (int(round((e - E0) / RES)), int(round((N1 - n) / RES)))

    every = []
    for cam in a.cams.split(","):
        P, uv, truth, got, err = load(cam)
        every.append((cam, truth, err))

        frame = cv2.imread(os.path.join(CAL, "frames", cam + ".jpg"))
        left = frame.copy()
        right = aerial.copy()
        for i, (p, t, e) in enumerate(zip(uv, truth, err), 1):
            col = residual_colour(e)
            cv2.circle(left, tuple(np.round(p).astype(int)), 9, col, 2, cv2.LINE_AA)
            cv2.putText(left, str(i), (int(p[0]) + 11, int(p[1]) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            q = mpx(*t)
            cv2.circle(right, q, 9, col, 2, cv2.LINE_AA)
            cv2.putText(right, str(i), (q[0] + 11, q[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        # crop the aerial to the points, then stack: side by side, equal height, squashes whichever
        # panel is the taller, and the numbers are the whole point of the picture
        qs = np.array([mpx(*t) for t in truth])
        pad = int(8 / RES)
        x0, y0 = max(0, qs[:, 0].min() - pad), max(0, qs[:, 1].min() - pad)
        x1, y1 = min(right.shape[1], qs[:, 0].max() + pad), min(right.shape[0], qs[:, 1].max() + pad)
        right = right[y0:y1, x0:x1]

        W = a.width
        fit = lambda im: cv2.resize(im, (W, int(round(im.shape[0] * W / im.shape[1]))),
                                    interpolation=cv2.INTER_AREA)
        left, right = fit(left), fit(right)
        img = np.vstack([left, np.full((10, W, 3), 255, np.uint8), right])
        bar = np.zeros((76, W, 3), np.uint8)
        cv2.putText(bar, "%s   %d points clicked   median %.0f cm, worst %.0f cm   "
                         "green <= %.0f cm, red >= %.0f cm"
                    % (cam, len(err), np.median(err) * 100, err.max() * 100, GOOD_M * 100, BAD_M * 100),
                    (22, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        img = np.vstack([bar, img])
        p = os.path.join(a.out, "control_points_%s.jpg" % cam)
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("%s: %d points, median %.1f cm, p90 %.1f cm, worst %.1f cm -> %s"
              % (cam, len(err), np.median(err) * 100, np.percentile(err, 90) * 100, err.max() * 100, p))

    # all three together: where each camera has points, and where nobody does
    img = aerial.copy()
    for cam, truth, err in every:
        col = CAM_COLOUR.get(cam, (200, 200, 200))
        for t, e in zip(truth, err):
            q = mpx(*t)
            cv2.circle(img, q, 10, col, 2, cv2.LINE_AA)
            if e > BAD_M:                                           # mark the ones worth re-clicking
                cv2.drawMarker(img, q, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
    y = 56
    cv2.rectangle(img, (0, 0), (img.shape[1], 70 + 54 * len(every)), (0, 0, 0), -1)
    cv2.putText(img, "control points by camera; white cross = lands more than %.0f cm out"
                % (BAD_M * 100), (24, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    for cam, truth, err in every:
        y += 54
        cv2.circle(img, (40, y - 10), 12, CAM_COLOUR.get(cam, (200, 200, 200)), -1)
        cv2.putText(img, "%s   %d points   median %.0f cm" % (cam, len(err), np.median(err) * 100),
                    (68, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    img = cv2.resize(img, (1500, int(img.shape[0] * 1500 / img.shape[1])), interpolation=cv2.INTER_AREA)
    p = os.path.join(a.out, "control_points_all.jpg")
    cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("all three ->", p)


if __name__ == "__main__":
    main()
