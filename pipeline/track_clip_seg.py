"""
track_clip_seg.py - like share_ai_team/track_clip.py but with an instance-segmentation model, so the foot point
comes from the MASK and not the box: the mask's lowest pixel is the tyre contact, the box's bottom edge often
includes the vehicle's cast shadow (RTK check of 2026-10-27: +-0.5-0.9 m along the direction of travel).

    python track_clip_seg.py --video clip.mp4 --out results/clip --model yolo26x-seg.pt

Writes <out>/trajectories.csv with the same columns as track_clip.py (foot_x = centre of the mask's horizontal
extent, foot_y = the mask's lowest row; x1..y2 = the mask's bounding box) plus
    tyre_x, tyre_y   mean column / lowest row of the mask pixels in its lowest 3 % (the nearest tyre), and
    box_foot_x, box_foot_y, bx1, by1, bx2, by2   the detector's own box, for comparison.
"""
import argparse, csv, os, time
import cv2, numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
SHARE = HERE                                        # botsort.yaml sits beside this script
CLASS_NAMES = {0: "Pedestrian", 1: "Bicycle", 2: "Passenger Vehicle", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
BIG = {5, 7}            # bus, truck
CONTAIN = 0.7           # a big-vehicle instance whose mask lies this much inside another big vehicle's mask is a PART of it


def suppress_parts(predictor):
    """Runs after the detector and before the tracker (callback order). COCO-trained YOLO often returns a long bus as
    two instances, the whole bus and its front; NMS keeps both (small box IoU) and the tracker gives them separate ids.
    The front of the bus often comes back as a second instance, as a bus or as a CAR. A vehicle instance whose MASK
    lies more than CONTAIN inside a bus/truck mask is a part of that bus: drop it. Masks, not boxes: a real car in
    front of a bus sits inside the bus's box but the bus's mask leaves it out."""
    import torch
    VEH = {2, 3, 5, 7}
    for i, r in enumerate(predictor.results):
        if r.boxes is None or len(r.boxes) < 2 or r.masks is None: continue
        cls = r.boxes.cls.int().tolist(); veh = [k for k, c in enumerate(cls) if c in VEH]
        if len(veh) < 2 or not any(cls[k] in BIG for k in veh): continue
        m = r.masks.data[veh] > 0.5; area = m.flatten(1).sum(1).float(); drop = set()
        for a in range(len(veh)):
            for b in range(a + 1, len(veh)):
                small, other = (a, b) if area[a] <= area[b] else (b, a)
                if cls[veh[other]] not in BIG or area[small] == 0: continue              # only parts of a bus/truck
                inter = (m[a] & m[b]).sum().float()
                if inter / area[small] > CONTAIN: drop.add(veh[small])
        if drop:
            keep = [k for k in range(len(r.boxes)) if k not in drop]
            predictor.results[i] = r[keep]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo26x-seg.pt"); ap.add_argument("--tracker", default=os.path.join(SHARE, "botsort.yaml"))
    ap.add_argument("--skip", type=int, default=2); ap.add_argument("--conf", type=float, default=0.10); ap.add_argument("--imgsz", type=int, default=1920); ap.add_argument("--device", default="0"); ap.add_argument("--no-suppress", action="store_true", help="skip the bus/truck part suppression")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    w = a.model if os.path.exists(a.model) else os.path.join(SHARE, a.model)
    model = YOLO(w); cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0; total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not a.no_suppress: model.add_callback("on_predict_postprocess_end", suppress_parts)   # added before the tracker registers, so it runs first
    f = open(os.path.join(a.out, "trajectories.csv"), "w", newline="", encoding="utf-8"); wr = csv.writer(f)
    wr.writerow(["frame", "time_sec", "track_id", "class", "conf", "foot_x", "foot_y", "x1", "y1", "x2", "y2", "tyre_x", "tyre_y", "box_foot_x", "box_foot_y", "bx1", "by1", "bx2", "by2", "poly"])
    frame_no, processed, t0 = 0, 0, time.time()
    while cap.grab():
        frame_no += 1
        if frame_no % a.skip: continue
        ok, frame = cap.retrieve()
        if not ok: break
        t = frame_no / fps
        res = model.track(frame, persist=True, tracker=a.tracker, conf=a.conf, iou=0.45, agnostic_nms=True, classes=list(CLASS_NAMES), imgsz=a.imgsz, device=a.device, verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            polys = res.masks.xy if res.masks is not None else [None] * len(res.boxes)
            for i in range(len(res.boxes)):
                bx1, by1, bx2, by2 = res.boxes.xyxy[i].tolist(); P = polys[i]
                if P is None or len(P) < 3:
                    x1, y1, x2, y2 = bx1, by1, bx2, by2; fx, fy = (x1 + x2) / 2, y2; tx, ty = fx, fy
                else:
                    x1, y1 = P[:, 0].min(), P[:, 1].min(); x2, y2 = P[:, 0].max(), P[:, 1].max()
                    fx, fy = (x1 + x2) / 2, y2
                    band = P[P[:, 1] >= y2 - max(0.03 * (y2 - y1), 2.0)]; tx, ty = band[:, 0].mean(), y2
                # the mask outline, thinned to <= 40 vertices, "x,y;x,y;..." so the renderer can draw the mask itself
                poly = "" if P is None or len(P) < 3 else ";".join(f"{int(q[0])},{int(q[1])}" for q in P[::max(1, len(P) // 40)])
                wr.writerow([frame_no, round(t, 3), int(res.boxes.id[i]), CLASS_NAMES.get(int(res.boxes.cls[i]), "unknown"), round(float(res.boxes.conf[i]), 3),
                             int(fx), int(fy), round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1), int(tx), int(ty),
                             int((bx1 + bx2) / 2), int(by2), round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1), poly])
        processed += 1
        if processed % 200 == 0: print(f"  frame {frame_no}/{total}  {processed / (time.time() - t0):.1f} frames/s", flush=True)
    cap.release(); f.close(); print(f"done: {processed} frames in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
