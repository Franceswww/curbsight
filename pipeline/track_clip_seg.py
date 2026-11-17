"""
track_clip_seg.py - detection and tracking of one camera clip with an instance-segmentation model.

    python track_clip_seg.py --video clip.mp4 --out results/clip [--model yolo26x-seg.pt]

The foot point comes from the MASK and not the box: the mask's lowest pixel is the tyre contact, the box's
bottom edge often includes the vehicle's cast shadow (RTK check of 2026-10-27: +-0.5-0.9 m along the
direction of travel).

Writes <out>/trajectories.csv: frame, time, track id, class, confidence, foot_x = centre of the mask's
horizontal extent, foot_y = the mask's lowest row, x1..y2 = the mask's bounding box, tyre_x, tyre_y = mean
column / lowest row of the mask pixels in its lowest 3 % (the nearest tyre), box_foot_x, box_foot_y,
bx1..by2 = the tracker's own box, and poly = the mask outline.

Two detection passes. At the full 1920 px a bus passing a few metres from the pole is over a thousand
pixels long, far larger than anything the detector was trained on: it comes back as two to four partial
boxes at confidence 0.1 to 0.3, and while it is in the picture the confidence of everything else drops
too (a waiting car 0.95 -> 0.24, a second bus 0.93 -> 0.31 on the same frame). The tracker then loses
the bus and its neighbours, and nothing tells the later stages that the car behind the bus is covered.
The same frame at 960 px gives one box over the whole bus at 0.9 and the neighbours at 0.9 as well.
The reduced pass cannot replace the full one: far and small road users need the pixels. So both are run
and merged, the full-size pass keeping authority over geometry wherever it can:
  - a road user both passes agree on (IoU >= 0.6) keeps the full-size box and mask and takes the higher
    of the two confidences, unless it is larger than --big-px, where the reduced pass's box is used;
  - one the passes disagree on (IoU 0.45 to 0.6, by the detector's own NMS rule the same object) is
    taken from the pass that is more confident;
  - one only the reduced pass sees is added if it is at least --mid-px across (it is there for the
    large ones; small ones are the full-size pass's business);
  - full-size fragments lying inside a bus or lorry of the reduced pass are dropped (masks, as in
    suppress_parts).
--imgsz2 0 runs the full-size pass alone.

The tracker (BoT-SORT, botsort.yaml) is driven directly, with the classes ultralytics' model.track uses,
because the detections it gets are the merged ones; make_wide_tracker adds a second, wider match for
road users too fast for the overlap test (--no-wide leaves it out). With --imgsz2 0 --no-wide the
output is what model.track gives, row for row.
"""
import argparse, csv, os, time
import cv2, numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = {0: "Pedestrian", 1: "Bicycle", 2: "Passenger Vehicle", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
BIG = {5, 7}            # bus, truck
VEH = {2, 3, 5, 7}
CONTAIN = 0.7           # a vehicle instance whose mask lies this much inside a bus/truck mask is a PART of it


def suppress_parts(predictor):
    """Runs after the detector (callback). COCO-trained YOLO often returns a long bus as two instances, the whole
    bus and its front; NMS keeps both (small box IoU). A vehicle instance whose MASK lies more than CONTAIN inside
    a bus/truck mask is a part of that bus: drop it. Masks, not boxes: a real car in front of a bus sits inside the
    bus's box but the bus's mask leaves it out."""
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


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix, 1e-6)


def detect(model, frame, a, size, tag):
    r = model.predict(frame, conf=a.conf, iou=0.45, agnostic_nms=True, classes=list(CLASS_NAMES), imgsz=size, device=a.device, verbose=False)[0]
    n = 0 if r.boxes is None else len(r.boxes)
    if n == 0: return []
    data = r.boxes.data.cpu().numpy()                                    # x1, y1, x2, y2, conf, cls (float32, as the tracker gets them)
    polys = r.masks.xy if r.masks is not None else [None] * n
    return [dict(box=data[i, :4].copy(), conf=float(data[i, 4]), cls=int(data[i, 5]), poly=polys[i], src=tag) for i in range(n)]


def raster(poly, box, shape, k=0.25):
    m = np.zeros((int(shape[0] * k) + 1, int(shape[1] * k) + 1), np.uint8)
    if poly is not None and len(poly) >= 3: cv2.fillPoly(m, [(np.asarray(poly) * k).astype(np.int32)], 1)
    else: cv2.rectangle(m, (int(box[0] * k), int(box[1] * k)), (int(box[2] * k), int(box[3] * k)), 1, -1)
    return m.astype(bool)


def merge_scales(H, L, shape, big_px, mid_px):
    """H: detections of the full-size pass, L: of the reduced pass -> one list (see the module text)"""
    out = [dict(d) for d in H]; dead = set()
    for l in sorted(L, key=lambda d: -d["conf"]):
        size = max(l["box"][2] - l["box"][0], l["box"][3] - l["box"][1])
        ious = [(iou(l["box"], h["box"]), i) for i, h in enumerate(out) if i not in dead]
        M = [(io, i) for io, i in ious if io >= 0.45]
        if not M:
            if size >= mid_px: out.append(dict(l))
            continue
        io, i = max(M); h = out[i]
        if io >= 0.6 and size < big_px:
            if l["conf"] > h["conf"]: h["conf"], h["cls"] = l["conf"], l["cls"]
            h["src"] = h["src"] + "+" if not h["src"].endswith("+") else h["src"]
            keep = i
        elif size >= mid_px and (size >= big_px or l["conf"] >= max(out[j]["conf"] for _, j in M)):
            out[i] = dict(l); out[i]["conf"] = max(l["conf"], h["conf"]) if io >= 0.6 else l["conf"]; keep = i
        else:
            continue                                                       # the full-size pass is the better view of it
        for _, j in M:                                                     # the other boxes on the same object are duplicates by the NMS rule
            if j != keep and out[j]["conf"] <= out[keep]["conf"]: dead.add(j)
    out = [d for i, d in enumerate(out) if i not in dead]
    # fragments of a large vehicle: a vehicle instance lying inside a bus or lorry that the reduced pass supplied
    cont = [d for d in out if d["cls"] in BIG and d["src"] == "L"]
    if cont:
        masks = {id(d): raster(d["poly"], d["box"], shape) for d in cont}; drop = set()
        for d in out:
            if d["cls"] not in VEH or d["src"] == "L": continue
            md = None
            for c in cont:
                if c is d or iou(d["box"], c["box"]) == 0: continue
                if md is None: md = raster(d["poly"], d["box"], shape)
                ar = md.sum()
                if ar and (md & masks[id(c)]).sum() / ar > CONTAIN and ar <= masks[id(c)].sum(): drop.add(id(d)); break
        out = [d for d in out if id(d) not in drop]
    return out


PEOPLE_CLS = {0, 1, 3}  # pedestrian, bicycle, motorcycle: one group for the wide match (a rider flickers between the three)


def make_wide_tracker(base):
    """The tracker class with a second, wider match for what the overlap test cannot follow.

    A new track is kept only if the next frame's detection overlaps it (IoU >= 0.3), and its first prediction
    has no velocity. A cyclist 60 px wide who moves 80 px between processed frames overlaps nothing: seen at
    0.8 confidence in every frame and never tracked (two-minute sample: three riders crossed the whole
    picture without one row). Where the overlap test leaves a track and a detection unmatched, they are
    compared again with both boxes grown by half their longer side on every side, and what is still
    unmatched once more with the boxes grown by the whole longer side (the cascaded buffered matching
    space of C-BIoU, Yang et al., WACV 2023), same kind of road user and a similar size only. The same
    second look is given to tracks that are already confirmed, because the filter needs several frames to
    learn a fast track's velocity, but only to a track that is young or is itself moving, and only toward
    where it is going: an established track that stands does not jump. Without that condition, on a row of
    parked cars 40 m away, where detections come and go, a track hopped from one car to the next."""
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.trackers.utils import matching

    class Wide(base):
        BUFS, GATE = (0.5, 1.0), 0.85                       # growth as a share of the longer side, small first; accept buffered IoU >= 0.15
        YOUNG, MOVING = 5, 0.25                             # frames a track counts as new; speed per frame, as a share of its longer side, that counts as moving

        def _wide_dists(self, tracks, dets, buf, established=False):
            D = np.ones((len(tracks), len(dets)), np.float32)
            for i, t in enumerate(tracks):
                a = t.xyxy; side = max(a[2] - a[0], a[3] - a[1]); ea = buf * side; A = (a[0] - ea, a[1] - ea, a[2] + ea, a[3] + ea)
                v = None
                if established and self.frame_id - t.start_frame > self.YOUNG:
                    v = (float(t.mean[4]), float(t.mean[5])) if t.mean is not None else (0.0, 0.0)
                    if np.hypot(v[0], v[1]) < self.MOVING * side: continue          # it stands: the detection beside it is someone else
                for j, d in enumerate(dets):
                    if (int(t.cls) in PEOPLE_CLS) != (int(d.cls) in PEOPLE_CLS): continue
                    b = d.xyxy; ar = ((a[2] - a[0]) * (a[3] - a[1])) / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
                    if not 0.5 <= ar <= 2.0: continue
                    if v is not None and ((b[0] + b[2] - a[0] - a[2]) * v[0] + (b[1] + b[3] - a[1] - a[3]) * v[1]) <= 0: continue   # behind it
                    eb = buf * max(b[2] - b[0], b[3] - b[1]); B = (b[0] - eb, b[1] - eb, b[2] + eb, b[3] + eb)
                    D[i, j] = 1.0 - iou(A, B)
            return D

        def _post_first_association(self, strack_pool, detections, u_track, u_detection, activated, refind):
            u_track, u_detection = list(u_track), list(u_detection)
            for buf in self.BUFS:
                cand = [i for i in u_track if strack_pool[i].state == TrackState.Tracked]
                if not cand or not u_detection: break
                m, _, _ = matching.linear_assignment(self._wide_dists([strack_pool[i] for i in cand], [detections[j] for j in u_detection], buf, established=True), thresh=self.GATE)
                mt, md = set(), set()
                for it, idet in m:
                    self._apply_match(strack_pool[cand[it]], detections[u_detection[idet]], activated, refind); mt.add(cand[it]); md.add(u_detection[idet])
                u_track = [i for i in u_track if i not in mt]; u_detection = [j for j in u_detection if j not in md]
            return u_track, u_detection

        def _unconfirmed_association(self, unconfirmed, u_detection, detections, activated, removed):
            detections = [detections[i] for i in u_detection]
            if not unconfirmed: return list(range(len(detections))), detections
            matches, u_unc, u_det = matching.linear_assignment(self.get_dists(unconfirmed, detections), thresh=0.7)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id); activated.append(unconfirmed[itracked])
            u_unc, u_det = list(u_unc), list(u_det)
            for buf in self.BUFS:
                if not u_unc or not u_det: break
                T = [unconfirmed[i] for i in u_unc]; Dd = [detections[j] for j in u_det]
                m, _, _ = matching.linear_assignment(self._wide_dists(T, Dd, buf), thresh=self.GATE)
                mt, md = set(), set()
                for it, idet in m:
                    T[it].update(Dd[idet], self.frame_id); activated.append(T[it]); mt.add(u_unc[it]); md.add(u_det[idet])
                u_unc = [i for i in u_unc if i not in mt]; u_det = [j for j in u_det if j not in md]
            for it in u_unc:
                track = unconfirmed[it]; track.mark_removed(); removed.append(track)
            return u_det, detections

    return Wide


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo26x-seg.pt"); ap.add_argument("--tracker", default=os.path.join(HERE, "botsort.yaml"))
    ap.add_argument("--skip", type=int, default=2); ap.add_argument("--conf", type=float, default=0.10); ap.add_argument("--imgsz", type=int, default=1920); ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz2", type=int, default=960, help="size of the second, reduced detection pass for large close road users; 0 = one pass only")
    ap.add_argument("--big-px", type=float, default=480.0, help="from this size (longer side, pixels) a road user's box and mask come from the reduced pass")
    ap.add_argument("--mid-px", type=float, default=200.0, help="a road user only the reduced pass sees is added from this size")
    ap.add_argument("--no-suppress", action="store_true", help="skip the bus/truck part suppression")
    ap.add_argument("--start-frame", type=int, default=0, help="frames of the video to pass over before the clip begins"); ap.add_argument("--nframes", type=int, default=0, help="length of the clip in frames, 0 = to the end")
    ap.add_argument("--no-wide", action="store_true", help="the tracker's own matching only, without the wider second look for fast road users")
    ap.add_argument("--dump", action="store_true", help="also write dets.csv: the detections handed to the tracker")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    w = a.model if os.path.exists(a.model) else os.path.join(HERE, a.model)
    model = YOLO(w); model2 = YOLO(w) if a.imgsz2 else None
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0; total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not a.no_suppress:
        model.add_callback("on_predict_postprocess_end", suppress_parts)
        if model2: model2.add_callback("on_predict_postprocess_end", suppress_parts)
    cfg = IterableSimpleNamespace(**YAML.load(a.tracker)); cfg.device = a.device
    tracker = (TRACKER_MAP[cfg.tracker_type] if a.no_wide else make_wide_tracker(TRACKER_MAP[cfg.tracker_type]))(args=cfg)
    f = open(os.path.join(a.out, "trajectories.csv"), "w", newline="", encoding="utf-8"); wr = csv.writer(f)
    wr.writerow(["frame", "time_sec", "track_id", "class", "conf", "foot_x", "foot_y", "x1", "y1", "x2", "y2", "tyre_x", "tyre_y", "box_foot_x", "box_foot_y", "bx1", "by1", "bx2", "by2", "poly"])
    fd = wd = None
    if a.dump:
        fd = open(os.path.join(a.out, "dets.csv"), "w", newline="", encoding="utf-8"); wd = csv.writer(fd); wd.writerow(["frame", "cls", "conf", "x1", "y1", "x2", "y2", "src"])
    frame_no, processed, t0 = 0, 0, time.time()
    for _ in range(a.start_frame): cap.grab()
    if a.nframes: total = a.nframes
    while cap.grab():
        frame_no += 1
        if a.nframes and frame_no > a.nframes: break
        if frame_no % a.skip: continue
        ok, frame = cap.retrieve()
        if not ok: break
        t = frame_no / fps
        dets = detect(model, frame, a, a.imgsz, "H")
        if model2: dets = merge_scales(dets, detect(model2, frame, a, a.imgsz2, "L"), frame.shape, a.big_px, a.mid_px)
        if wd:
            for d in dets: wd.writerow([frame_no, CLASS_NAMES.get(d["cls"], "unknown"), round(d["conf"], 3)] + [round(float(v), 1) for v in d["box"]] + [d["src"]])
        data = np.array([[*d["box"], d["conf"], d["cls"]] for d in dets], np.float32).reshape(-1, 6)
        tracks = tracker.update(Boxes(data, frame.shape[:2]), frame)
        for tr in tracks:
            bx1, by1, bx2, by2 = [float(v) for v in tr[:4]]; d = dets[int(tr[-1])]; P = d["poly"]
            bx1, bx2 = min(max(bx1, 0.0), float(frame.shape[1])), min(max(bx2, 0.0), float(frame.shape[1]))     # the tracker's box, kept inside the frame
            by1, by2 = min(max(by1, 0.0), float(frame.shape[0])), min(max(by2, 0.0), float(frame.shape[0]))
            if P is None or len(P) < 3:
                x1, y1, x2, y2 = bx1, by1, bx2, by2; fx, fy = (x1 + x2) / 2, y2; tx, ty = fx, fy
            else:
                x1, y1 = P[:, 0].min(), P[:, 1].min(); x2, y2 = P[:, 0].max(), P[:, 1].max()
                fx, fy = (x1 + x2) / 2, y2
                band = P[P[:, 1] >= y2 - max(0.03 * (y2 - y1), 2.0)]; tx, ty = band[:, 0].mean(), y2
            # the mask outline, thinned to <= 40 vertices, "x,y;x,y;..." so the renderer can draw the mask itself
            poly = "" if P is None or len(P) < 3 else ";".join(f"{int(q[0])},{int(q[1])}" for q in P[::max(1, len(P) // 40)])
            wr.writerow([frame_no, round(t, 3), int(tr[4]), CLASS_NAMES.get(int(tr[6]), "unknown"), round(float(tr[5]), 3),
                         int(fx), int(fy), round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1), int(tx), int(ty),
                         int((bx1 + bx2) / 2), int(by2), round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1), poly])
        processed += 1
        if processed % 200 == 0:
            print(f"  frame {frame_no}/{total}  {processed / (time.time() - t0):.1f} frames/s", flush=True)
            # over an hour of video the allocator's cache grows (two models, a varying number of masks) until the card
            # is full and everything crawls: three 71-minute runs side by side filled 24 GB and dropped to a fifth of their speed
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    cap.release(); f.close()
    if fd: fd.close()
    print(f"done: {processed} frames in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
