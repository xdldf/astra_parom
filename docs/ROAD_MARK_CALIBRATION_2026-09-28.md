# Metre marks, road widths and lens correction — 2026-09-28

The user supplied an annotated image identifying 1 m spacings and confirmed that the two widths are **5.80 m and 5.50 m between barriers**. Barrier-mark heights above the road are **unknown**. This supersedes the previous lack of any site distance information, but does not make the raised marks ground-level controls.

**Result: a fitted lens correction, inspectable metre-mark data and an integrated projective measurement path were implemented. The new candidate fails its geometry checks and is exported for review, with numeric vehicle lengths disabled. Absolute ±10 cm has not been achieved or certified.**

## Evidence and registration

The annotated 1248×850 image was registered to an original 2592×1944 video frame using static image features. There were 214 RANSAC inliers with median error 0.274 annotation pixels. The annotation is approximately a 0.480 scale image with a 76-pixel top crop; treating it as a resized full video frame would misplace the calibration.

The clean frame used for digitization is `ST_2026-09-04_18-00-04.mp4`, frame 27750. Digitized inputs and their provenance are saved in `config/st-road-marks.json`: 14 visible near-barrier paint positions, 16 far positions read from the user's metre sketch, two width-arrow endpoint pairs and eight top/foot support pairs. The near terminal was not counted as a yellow mark. X paint centres and the upper beam edge were identified separately. Far marks are less distinct; their 5-pixel digitization uncertainty is larger than the 1.5-pixel near-paint uncertainty. These are modelling assumptions, not measured survey precision.

The final overlay is `runs/road_geometry_20260928/digitized_features.jpg`. Width endpoint placement follows the coarse sketch and is given 0.10 m uncertainty in the fitting objective. No vehicle catalogue value or previous empirical vehicle reference was used to fit the lens or metre scale.

## Lens and ruler fit

The two ruler rows were fitted jointly with Brown radial distortion and a projective mapping of their plane. The second row's origin/direction were fitted rather than assuming that the first paint marks across the road line up. The radial map was checked for monotonicity over the corrected viewing field. The focal parameter fixes the mathematical distortion parameterization; it is not a separately measured optical focal length.

Fitted app lens values:

```json
{
  "k1": -0.19113115836419048,
  "k2": 0.019840260882093858,
  "cx": 0.538504106428874,
  "cy": 0.4730942993578638,
  "focal": 0.47,
  "zoom": 0.91,
  "tilt_deg": 3.6
}
```

Nine marks were withheld from **both initialization and fitting**. Their maximum conditional ruler-plane position error was **9.71 cm**. Fitting all marks gives a maximum residual of **19.86 cm**, at a coarsely annotated far-end point. Both fitting and held-out residuals enter the application quality gate. Neither number is a vehicle-length accuracy measurement, and the larger error is not discarded.

Lens correction changes ray geometry. It does not put elevated bumper endpoints or barrier paint on the road. This separation follows the [OpenCV camera calibration model](https://docs.opencv.org/3.4.11/d9/d0c/group__calib3d.html) and its distinction between projection and distortion.

## Raised barriers versus road

I tested a common-height, upright-post model to transfer the metre-plane calibration to the road. Each post was withheld once and its foot predicted from the other posts. The worst disagreement was **43.78 cm** in the conditional metric coordinates.

| Support | Held-out transfer disagreement |
|---|---:|
| near_left | 34.1 cm |
| near_middle | 5.5 cm |
| near_right | 43.8 cm |
| near_end | 35.1 cm |
| far_left | 7.5 cm |
| far_middle | 19.1 cm |
| far_right | 26.9 cm |
| far_end | 27.9 cm |


These are consistency checks of inferred correspondences, not surveyed true ground-coordinate errors. Leaning posts, unclear contact points, different barrier heights and road grade can all violate the assumptions. A more flexible plane transfer was also tried locally and did not resolve the held-out inconsistency. It was not promoted into the application.

For example, treating the elevated ruler plane directly as ground produced a black-SUV diagnostic span around 3.78 m; transferring it using the inferred posts changed the span to about 4.45 m. That large sensitivity is why raised marks must not be fed to the existing ground-ruler scale unchanged. These exploratory spans were not saved as accepted vehicle measurements. Even a correct ground homography leaves body height, visible-wheel depth, vehicle heading and bumper localization to resolve; an axis-aligned detector box is not a 3D bumper measurement.

## Application changes

- Added `vehicle_metrology/road_survey.py` for ruler fitting, held-out checks, a tested rail-to-ground transfer and projective spans that follow the road's longitudinal vanishing point.
- Added `scripts/calibrate_road_rulers.py`, which exports a complete profile, correction preview, road-grid preview and numerical evidence.
- Added a survey profile type, bound to its exact lens parameters. Changing the lens requires recalibration. Old vehicle references cannot silently reweight this physically separate calibration.
- The backend checks both ruler and ground-transfer errors against the profile's 0.10 m target. Failed geometry produces `calibration_review` with **no assigned length**. Diagnostic candidate lengths stay separate from `length_m` and tariffs. At-line failures can still be recorded for operator review with their photo.
- The UI shows the ruler and ground-transfer errors separately. Original-frame coordinates, lens invalidation, source selection and review captures are preserved.

The exported review profile is `config/st-road-survey-review.json`. It is not substituted for the existing ST preset because it fails acceptance. Importing it in the calibration editor displays the corrected view and failed geometry status; it does not authorize ±10 cm measurements.

## Video replay and tests

Re-ran YOLO26 M at 640, confidence 0.30, on **300 original frames from six clips across all three videos**, using the new lens and the actual shared measurement function. Results: {"waiting_for_line": 504, "outside_road": 391, "calibration_review": 19}. Numeric accepted observations: **0**. This is expected from the failed geometry gate, not a claimed detector capture/accuracy rate.

Python regression suite: **192 passed**. Frontend suite: **32 passed**. New synthetic tests use independently projected camera geometry to check ruler recovery and raised-plane transfer, detect a displaced post, verify vanishing-point span geometry, reject stale/singular calibration and prevent vehicle-reference contamination. Software tests do not validate field accuracy.

## Reproduce and inspect

```bash
.venv/bin/python scripts/calibrate_road_rulers.py \
  --survey config/st-road-marks.json \
  --raw-frame runs/road_geometry_20260928/empty_raw.jpg \
  --output-dir runs/road_geometry_20260928_repeat
```

Evidence: `runs/road_geometry_20260928/calibration/fit.json`, `manifest.json`, `video_checks.json`, `video_summary.json`, `before_after.jpg`, and `calibration/corrected_full.jpg`. Original videos and the annotated image were not overwritten. Correction uses deterministic geometric resampling, not generated imagery.

## What remains necessary

To establish ±10 cm for actual vehicles, the road plane needs sufficiently precise ground controls or a verified height/plane transfer, followed by independently measured vehicles spanning the used road area and body types. The current image, metre spacings and two barrier widths have improved the lens/scale model but have not resolved that geometry. The current system must not be treated as a validated safety-critical ±10 cm measuring instrument.
