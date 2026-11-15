"""
Every person and vehicle on every processed frame of a recording, without tracking, for the
pixelation in render_fused.py.

    python blur_boxes.py --video /path/to/cam02_1019.mp4 --out ../sample/cam02_1019

The tracker (track_clip_seg.py) keeps what it can follow: a road user it loses behind a bus, or
picks up a second late, has no row in trajectories.csv for those frames and would stay visible in a
rendered camera view. This pass keeps every detection instead, at a low confidence, at two image
sizes and from every model given, and writes <out>/blur_boxes.csv (frame, cls, conf, x1, y1, x2, y2).
render_fused.py covers those boxes as well as the tracked outlines. A false detection costs a few
blocks of background, a missed one shows a person, so the threshold is low on purpose.

Several models see more than one: --weights takes a comma-separated list.
"""
import argparse
import csv
import os

import cv2
from ultralytics import YOLO

KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="the camera's folder, beside trajectories.csv")
    ap.add_argument("--weights", default="yolo26x-seg.pt", help="one model or several, comma-separated")
    ap.add_argument("--skip", type=int, default=2, help="the same frame step as track_clip_seg.py")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--sizes", default="1920,2880")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    models = [YOLO(w) for w in a.weights.split(",")]
    sizes = [int(v) for v in a.sizes.split(",")]
    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fr = n = rows = 0
    with open(os.path.join(a.out, "blur_boxes.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "cls", "conf", "x1", "y1", "x2", "y2"])
        while True:
            ok, im = cap.read()
            if not ok:
                break
            fr += 1
            if fr % a.skip:
                continue
            for model in models:
                for size in sizes:
                    r = model.predict(im, conf=a.conf, imgsz=size, classes=list(KEEP), device=a.device, verbose=False)[0]
                    for b, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        wr.writerow([fr, KEEP[int(c)], round(float(p), 2)] + [round(float(v), 1) for v in b])
                        rows += 1
            n += 1
            if n % 150 == 0:
                print(n, "frames", rows, "boxes", flush=True)
    print(a.video, "->", os.path.join(a.out, "blur_boxes.csv"), n, "frames", rows, "boxes")


if __name__ == "__main__":
    main()
