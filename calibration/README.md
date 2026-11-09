# Calibration

| file | what it is |
|---|---|
| `aerial_wide.png` + `.json` | the base map: 2050 x 2675 px at 4 cm/px, east/north metres from 33.94760 N, 83.37630 W |
| `ground/corner_ground_enu.npz` | ground elevation from the USGS 3DEP airborne lidar (GA Statewide 2018), NAVD88 |
| `control_points/<camera>.json` | the points clicked by hand, camera pixel against aerial pixel |
| `<camera>_fit_pnp/cam.json` | the solved camera: K, distortion, rotation, position, and the residuals it achieved |
| `ladders_tight.json` | the crosswalk markings, measured on the aerial; used by the older flat-plane fit |
| `lot_sanford_cedar.json` | the parking bays, drawn by hand, so standing cars are not reported as traffic |
| `frames/<camera>.jpg` | the frame the points were clicked on: the median background of the 2026-10-19 14:52 recording, so no moving object is in it |
| `ortho/ortho_index.csv` | the aerial tile's georeference, which turns a clicked aerial pixel into metres |

To see the points rather than their residual, `pipeline/plot_control_points.py` draws each camera's
frame and the aerial side by side with the points numbered and coloured by error, and all three
cameras together. The figures it writes are in `../figures/control_points_*.jpg`.

`cam.json` positions are in the same east/north metres as the base map, with z in NAVD88 metres.
Residuals (`residual_median_m`, `residual_p90_m`) are distances on the ground, not pixels.

To solve a new camera, click its points against the aerial and run `fit_pnp.py`:

    cd ../pipeline
    python fit_pnp.py --pairs ../calibration/control_points/cam02_1019.json

Re-solving a camera overwrites its `cam.json`. Pass `--tag` to write a separate solution, and give
`--fit-suffix` the matching name when fusing or rendering.
