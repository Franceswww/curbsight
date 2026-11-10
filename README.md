# CurbSight

Roadside cameras to one map. Several cameras on one pole watch an intersection; this turns what they
see into one set of trajectories on the ground, in metres, with each road user carrying a single
identity across every view.

Everything here runs on the minute of real footage in `sample/`, and the accuracy figures below were
measured against a survey-grade GNSS track and a lidar, not estimated.

![Three camera views and the fused map, in motion](figures/demo.gif)

*Thirteen seconds of it. Three camera views along the top, one map below. The same bus, the same
pedestrian, one identity each, wherever they are. The whole minute is in
[`figures/fused_demo.mp4`](figures/fused_demo.mp4).*

## 3,770 road users in 71 minutes

![Vehicle paths over one minute](figures/1min_vehicles.jpg)

*Every vehicle path in the sample minute that lasts at least 3 seconds. Dot = where the track
starts, arrow = where it ends. Shorter paths are drawn thin and grey.*

![Pedestrian paths over one minute](figures/1min_pedestrians.jpg)

*The same minute, pedestrians. Fifty-nine road users in sixty seconds, on a university campus at
three in the afternoon.*

Over a longer recording the shape of the place appears. These are the same scripts on 71 minutes
from the same afternoon. That recording is not in the repository, only its figures:

![Vehicle paths over 71 minutes](figures/71min_vehicles.jpg)

*335 vehicle paths. The lanes, the turning movements and the bus stops draw themselves.*

![Pedestrian paths over 71 minutes](figures/71min_pedestrians.jpg)

*1,986 pedestrian paths, drawn the same way. This is the point at which lines stop working.*

![Pedestrian presence](figures/71min_density_pedestrians.jpg)

*So here is the same data as presence instead:
how long each half-metre of ground was occupied. The footways, both crossings and the waiting areas
at the corners come out, and so do the benches: the brightest half-metre in the picture collected
45 minutes of the 71, because somebody sat there.*

![Vehicle presence](figures/71min_density_vehicles.jpg)

*The same for vehicles. The bright patches are where traffic waits: the stop line, the bus stop, the
queue at the lights.*

## Five steps from pixel to metre

1. **A base map.** A county aerial photograph at 6 cm per pixel, carrying real coordinates,
   resampled onto a 4 cm grid in east/north metres from a fixed origin
   (33.94760&deg;N, 83.37630&deg;W).
2. **Calibrate each camera.** Click matching points on a camera frame and on the aerial photograph,
   then solve that camera's position, orientation, focal length and lens distortion by minimising
   the error **on the ground, in metres**, not in pixels. The ground is not assumed flat: a pixel's
   ray is cast onto a real elevation surface built from airborne lidar.
3. **Detect and track in each view.** YOLO26x-seg plus BoT-SORT, every second frame. The foot point
   comes from the segmentation mask, not from the bounding box, because the mask's lowest edge is
   where the tyres meet the road.
4. **Project to the ground.** Each detection becomes a position in metres on that elevation surface.
5. **Merge the views.** One object per road user, however many cameras can see it. Positions are
   combined by inverse-square range, so the nearest camera dominates.

![Three cameras, one map](figures/fused_frame.jpg)

*One frame, large enough to read. Orange = bus, blue = car, red = truck, yellow = pedestrian. A `*`
after a label means two cameras were seeing that object at that moment. Bus 64 is in cam03 and in
cam04, and appears on the map once.*

Nothing in step 2 tells the solver how high the pole is or where it stands. It works that out from
the clicked points alone, and the answer it returns, 5.4 m, 6.5 m and 6.7 m above the ground it
solved them on, all three within 1.6 m of each other horizontally, is the first thing worth
checking, because it is a real pole you can go and look at.

| camera | points clicked | median residual | 90th percentile |
|---|---|---|---|
| cam02 | 88 | 9 cm | 32 cm |
| cam03 | 101 | 17 cm | 41 cm |
| cam04 | 47 | 32 cm | 71 cm |

Those are how far the clicked points land from where they should, measured on the ground. cam04 is
the worst of the three and also the one with the fewest points; more points is the obvious fix.

A median is one number and it hides everything, so the points are in the repository as pictures. The
same point carries the same number and the same colour in the camera frame and on the aerial
photograph, green within 10 cm and red beyond 50, so you can follow any one of them across and judge
it yourself:

![cam02 control points](figures/control_points_cam02_1019.jpg)

*cam02, 88 points: the ends of the crossing stripes, kerb corners, a drain, the corners of the
brickwork. Almost all green. Its worst point is 78 cm out, and it is 37 m away, where one pixel of
vertical click is already 29 cm of ground, so that is under three pixels of hand.*

![cam04 control points](figures/control_points_cam04_1019.jpg)

*cam04, 47 points, median 32 cm. It has half as many points as cam02 spread over a longer reach,
out to 57 m against cam02's 40 m, so each one carries more. Its worst, point 34 at 1.53 m, is a
different problem: at that range it is twelve pixels, far more than a hand can slip, so that pair is
almost certainly matched to the wrong feature. A residual table cannot tell those two causes apart.
A picture can.*

![all three cameras' control points](figures/control_points_all.jpg)

*All three together on the aerial. The points gather on paint, kerbs and drains, because those are
the only things you can identify in a camera frame and in an aerial photograph with any confidence.
Plain tarmac and grass have none, nothing to click, which is why the junction is pinned down far
better than the ground away from it.*

Regenerate these at any time:

```bash
cd pipeline
python plot_control_points.py
```

## 19-59 cm against a survey-grade GNSS track

Measured at this intersection against three independent references, on separate recordings.

| reference | what it is | position error, median / 90% |
|---|---|---|
| GNSS car, pass 1 | a car carrying a survey-grade receiver, driven through | 59 / 108 cm |
| GNSS car, pass 2 | the same car, second pass | 19 / 72 cm |
| pedestrians, set 1 | people seen by a lidar mounted on that car | 71 / 108 cm |
| pedestrians, set 2 | ditto, second recording | 34 / 75 cm |
| other vehicles | the lidar's view of ordinary traffic, matched one-to-one | 109 / 197 cm |

The car that can be checked most precisely sits at 19–59 cm. Ordinary traffic, which the pipeline
was never tuned on and whose real size it has to assume, sits near a metre. That gap is a position
error rather than a heading or size error: once a vehicle is moving, its heading agrees with its own
direction of travel to about a degree.

## Running it

```bash
pip install -r requirements.txt
```

Steps 2, 4 and 5 need nothing beyond numpy, scipy and OpenCV. Step 3 also needs `ultralytics` and a
PyTorch build matching your CUDA version. See the note in `requirements.txt`.

The sample already contains the output of step 3, so the fusion and the rendering run on their own:

```bash
cd pipeline
python fuse_tracks.py --clipdir ../sample --cams cam02_1019,cam03_1019,cam04_1019
python render_fused.py --clipdir ../sample --cams cam02_1019,cam03_1019,cam04_1019 \
    --start 15:02:07 --date 2026-10-19 --lot-file ../calibration/lot_sanford_cedar.json
```

To redo the detection and tracking as well, which takes a few minutes per view on a GPU and considerably
longer on a CPU, run step 3 first. The weights download themselves on first use:

```bash
python track_clip_seg.py --video ../sample/cam02_1019.mp4 --out ../sample/cam02_1019
python track_clip_seg.py --video ../sample/cam03_1019.mp4 --out ../sample/cam03_1019
python track_clip_seg.py --video ../sample/cam04_1019.mp4 --out ../sample/cam04_1019
```

To get the trajectories as data rather than as a video:

```bash
python export_tracks.py --fused ../sample/fused.csv --out ../out \
    --start 15:02:07 --date 2026-10-19 --lot-file ../calibration/lot_sanford_cedar.json
```

which writes one row per object per frame, with wall-clock time, latitude and longitude, and
aerial-image pixel beside the metres, plus a line per trajectory in `tracks_summary.csv` giving its
duration, path length, mean and top speed, and both endpoints, and the two trajectory figures.

A vehicle is drawn only where its position behaves like a vehicle: on the lane network, and
turning no tighter than its body can. The lane network is learned from the traffic itself, one hour
of it, and shipped as `calibration/lane_field_sanford_cedar.json`; to rebuild it from a longer
recording:

```bash
python build_lane_field.py --tracks ../out/tracks.csv --out ../calibration/lane_field_sanford_cedar.json
```

Over a long recording the lines overlap into a tangle, and presence reads better than paths:

```bash
python density.py --tracks ../out/tracks.csv --out ../out/density_pedestrians.png
python density.py --tracks ../out/tracks.csv --out ../out/density_vehicles.png --vehicles
```

## Four folders

| | |
|---|---|
| `pipeline/` | the five steps, plus the export and the renderer |
| `calibration/` | the solved camera models, the clicked control points, the base map, the ground surface, the lane network |
| `sample/` | one minute, three cameras, 2026-10-19 15:02:07, with step 3 already done |
| `figures/` | the pictures above, and the sample minute as video |

A camera's solved model lives in `calibration/<camera>_fit_pnp/cam.json`. `_fit_pnp` is the default
value of `--fit-suffix`, so several solutions for one camera can sit side by side and be chosen on
the command line.

## Known limits

**The "pedestrian" figure includes cyclists.** The code splits road users in two: cars, motorcycles,
buses and lorries on one figure, everything else on the other. A cyclist is detected as a bicycle and
so lands with the pedestrians. Over the 71 minutes that is 118 bicycles among 3,525 pedestrians.

**Parking is not traffic.** A car standing in a bay is still detected every frame, and its solved
centre drifts a metre or two as the cameras see more or less of it. Drawn, it looks like a vehicle
wandering around a car park. `calibration/lot_sanford_cedar.json` marks the bays by hand, and
vehicles inside them are left out; the drive in and the drive out are kept, because those do affect
the junction.

**A standing vehicle has no direction.** Heading is read from the track, so it means nothing until
the vehicle has actually moved a couple of metres. A car creeping into a bay keeps whatever heading
it last had out on the aisle, which is why it is better not to draw it at all than to draw it
pointing the wrong way.

**The cameras agree with the map, not with each other.** Each is solved against its own control
points, so nothing forces two of them to place the same physical point identically. On this
recording two cameras differ by 0.6 m to 1.2 m on the same pedestrian's feet. Only about one row in
ten has two cameras at all, so single-camera accuracy is what mostly matters, but the difference
does show as a small step when an object hands over from one camera to the next.

**The base map is a photograph of one particular morning.** The vehicles parked in it are from that
morning and have nothing to do with the footage. Three that were standing on the roadway have been
painted out; everything else is as flown.

## Site

Sanford Drive at Cedar Street, University of Georgia, Athens. The cameras are on one pole on the
corner; three of them are calibrated here. The sample is a Monday afternoon.
