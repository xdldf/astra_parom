# Road position and an alternative to box scaling

The ±10 cm vehicle-length target is **not met**. The road polygon and position-dependent scale exist, but the previous box method does not recover physical bumper geometry. This follow-up adds an inspectable local road cross-section and tests a separate reconstruction from tyre/bumper landmarks across frames.

## What the system actually considers

- The road polygon lives in the full-resolution, lens-corrected image. The box bottom centre must lie inside it.
- The empirical path finds the local far and near polygon boundaries at that X coordinate. Their fractional depth determines the scale. This percentage is **not camera distance in metres**.
- The survey path uses a ground homography, follows the road's longitudinal vanishing point and integrates the projected box span at the bottom-centre depth. It still assumes a box can represent an aligned vehicle at one ground depth.
- The inspector now draws the actual local cross-section and, for the survey path, the projected span. The API identifies the position model and leaves `camera_distance_m` null for box estimates.
- A raised bumper, vehicle heading and a tyre hidden by the near barrier can invalidate a box-derived length even when the box is stable. The SUV footage contains tyre occlusion near the measurement line; the new landmark experiment uses earlier frames with visible contacts.

![Road polygon, detector box, local depth and landmarks](../runs/projective_multiview_20260928/road_position_inspection.jpg)

## Implemented alternative

`vehicle_metrology/projective_track.py` reconstructs two persistent bumper landmarks using poses derived from front/rear tyre contacts. It uses a 3×4 projection from the aligned ground and raised-barrier plane mappings, not a pixel-width multiplier or catalogue vehicle length.

If the two parallel planes share their metric X/Y axes and homogeneous scale, then

```
P = [G[:,0], G[:,1], R[:,2] - G[:,2], G[:,2]]
```

maps `[X metres, Y metres, Z/barrier_height, 1]` to corrected pixels. Absolute barrier height is unnecessary for longitudinal X/Y reconstruction under this model, but cannot be used to report physical Z or Euclidean camera range. The implementation reports only **horizontal camera-to-contact-midpoint distance**, conditional on the fitted geometry. The supplied post-derived ground plane remains unreliable; this construction does not repair that calibration.

Checks cover observation identity/coordinates, visible contacts, calibrated road support, physical endpoints, sufficient movement, numerical conditioning, reprojection error, wheelbase consistency, alternating-frame length agreement and pixel-placement sensitivity. Failed calibration always withholds `length_m`; `candidate_length_m` is diagnostic only. `accuracy_validated` stays false even when all numerical checks pass.

## Actual footage results

The same black SUV was annotated at full corrected resolution. No catalogue dimensions were fitted. The approximate annotations were made visually by the assistant, with an assumed 5 px uncertainty. The points are near-side bumper corners, **not independently verified bumper extrema or fine-tuning ground truth**.

| Check | Short track | Extended track |
|---|---:|---:|
| Original frame range | 12562–12600 | 12500–12600 |
| Annotated frames | 7 | 10 |
| Recovered movement baseline | 1.84 m | 3.79 m |
| Diagnostic length | 4.560 m | 4.635 m |
| Alternating-frame estimates | 4.410 / 5.146 m | 5.057 / 4.444 m |
| Difference between frame subsets | 73.6 cm | 61.3 cm |
| Bumper reprojection RMS | 2.74 px | 4.95 px |
| Physically usable pixel perturbation fits | 132/200 | 179/200 |
| Accepted length | None | None |

The longer track still fails. Its conditional pixel-noise interval is 3.557–4.682 m, excluding calibration bias and wrong landmark identity; this is not a physical accuracy confidence interval. The camera's inferred horizontal ground position is `(3.82, -8.83)` m and the extended track's horizontal distance changes from 13.39 to 11.34 m. These are model-derived diagnostics, not surveyed ranges. The existing barrier-to-road check still disagrees by up to 43.78 cm.

This tests one alternative on one vehicle in two viewing spans. It does **not** establish that every possible multi-view method fails. It establishes that this calibration and these visible-point annotations cannot justify ±10 cm. The larger-model and 300-frame tests on all three videos remain documented separately in the earlier reports.

## Reproduce

From the repository root:

```bash
.venv/bin/python scripts/reconstruct_road_track.py \
  --profile config/st-road-survey-review.json \
  --fit runs/road_geometry_20260928/calibration/fit.json \
  --landmarks runs/projective_multiview_20260928/suv_extended_landmarks.json \
  --output-dir runs/projective_multiview_20260928/extended \
  --detector runs/accuracy_10cm/models/yolo26m.pt \
  --mc-samples 200
```

The runner decodes the original video, checks image size/lens/projection agreement, renders overlays, compares YOLO boxes, and records SHA-256 hashes of the video, calibration, annotations and checkpoint. Generated evidence and videos remain local under `runs/`, outside Git. The new geometry method is an **offline diagnostic**, not the live station's measurement engine. No production settings or records were changed in this follow-up. The workbench inspector changes apply to the existing engine.

Artifacts:

- `runs/projective_multiview_20260928/suv_landmarks.json`
- `runs/projective_multiview_20260928/suv_extended_landmarks.json`
- `runs/projective_multiview_20260928/reconstruction/results.json`
- `runs/projective_multiview_20260928/extended/results.json`
- `runs/projective_multiview_20260928/extended/manifest.json`
- `runs/projective_multiview_20260928/extended/contact_sheet.jpg`

## Verification and remaining evidence

202 Python tests and 33 frontend tests pass. Independent synthetic projections verify recovery of a known 4.5 m length across road depths, headings and different unknown barrier-height units. Other tests reject stationary tracks, occluded points, incompatible plane scales, repeated frames, inconsistent landmarks and failed calibration. Synthetic success verifies the implementation, not the real camera calibration.

Fine-tuning was not performed: the current approximate boxes/landmarks and uncertain lengths cannot serve as verified metric labels. The useful next data are accurately located ground-plane controls and clearly identified vehicle landmarks with independently measured vehicle lengths for a separate validation set. A calibrated second view could also constrain hidden bumper/tyre geometry; the existing front-camera feed has not been calibrated into this reconstruction.
