# Catalogue-trained outlines in real captures

The station now supports a vehicle-outline estimator in the actual saved-capture path for both recorded videos and IP cameras. It uses corrected full frames and factory catalogue lengths. It requires neither a calibration board nor barrier heights.

**The requirement of less than 10 cm error for every car remains unmet.** This is an approximate measurement mode with explicit review outcomes. Software tests and successful captures do not establish that accuracy.

## Behaviour

- The ordinary detector/tracker continues to find passages. The additional outline inference runs once when saving a car, rather than on every preview frame.
- [YOLO26-M segmentation](https://docs.ultralytics.com/tasks/segment/) supplies the visible car silhouette at 1280 input resolution, mapped to the full corrected image. Regression uses outline width, height, road position and widths across three body-height bands. It does not reconstruct hidden tyres or substitute the car's catalogue entry at inference time.
- The saved length is explicitly an approximate outline estimate. The original box/temporal length, temporal samples, contour, image features, model checksum and calibration identity remain in the measurement evidence.
- A passenger car with an unavailable/ambiguous outline or features outside the training range is saved for review **without an assigned length**. Its original box estimate remains visible as evidence. Model failures do not drop its screenshot or vehicle record.
- Trucks and other non-passenger classes retain their existing measurement path. This passenger-car model does not solve trailers or projecting loads. Existing ambiguous temporal associations and clipping rules are preserved.
- `/evaluation` displays the contour over the full corrected photo and shows the original box estimate separately. The stored photo stays clean; it contains no burnt-in outline. Existing skip, label and JSON export actions remain available.
- The preview/studio box estimate remains a preliminary value. The outline estimate is calculated on capture. The outline coefficients are a frozen catalogue model; later evaluation labels are saved in JSON but do not silently retrain these coefficients.
- The profile is bound to its exact image size, lens correction and road polygon. Changing that geometry invalidates the outline calibration. This model is accepted only in estimate mode.

## Results and remaining failures

All 78 previously eligible catalogue passages from the six videos were retained. The feature design was developed after examining errors in those videos, so these are development results, not a fresh final test. Each model family is excluded from its own coefficient fitting and inner regularization selection; no tested family's length labels enter that fold.

| Result | Existing box method | Outline regression, before support checks |
|---|---:|---:|
| Mean absolute distance to catalogue-interval midpoint | 12.15 cm | 8.52 cm |
| Maximum distance to catalogue interval | 52.24 cm | 25.81 cm |
| Outside 10 cm even for the closest value in the interval | 45/78 | 27/78 |
| Within 10 cm for every value in the interval | 29/78 | 46/78 |
| Variant-dependent | 4/78 | 5/78 |

Applying the same feature-range and ambiguous-track rejection used at runtime changes the outcome to **42 within the entire interval, 23 outside, 5 variant-dependent and 8 review-only**, out of all 78. The 8 reviews count as unsuccessful automatic measurements. Error on the 70 reported values averages 8.17 cm; that subset average must not be presented as accuracy across all vehicles. There are still 36 cases without an unconditional within-10-cm catalogue result.

The largest remaining reported mismatch is the provisional Spacio V2-017, about 25.8 cm from the interval. The previous worst case V2-046 falls outside support when the entire Spacio family is held out, so that fold saves a review rather than restoring its old 4.797 m estimate. Some other cars still get worse. Exact filmed variants remain visually inferred, and catalogue dimensions do not cover modifications, loads or attached trailers.

All 78 segmentation masks reproduced exactly through the runtime inference adapter. The actual capture estimator also reproduced every heldout-fold numeric result and all eight review outcomes from those masks. One additional review (V4-021) preserves an ambiguous temporal-identity rejection rather than letting a clean-looking single-frame outline override it. A separate integration replay called the real `station.capture` from one original-video passage in each of the six recordings, using the supplied profile, real neighbouring-frame inference for the five at-line passages, and the existing review capture for the off-line passage. All six retained 2592×1944 corrected photos and outline evidence. This used an isolated database and an explicit offline CPU override. It tests integration, not heldout accuracy; those cars are in the exported model's training set. GPU throughput was not measured.

The full regression suite passed: **329 Python tests and 69 frontend tests**. Tests include IP/video persistence, missing models, out-of-range silhouettes, unsupported vehicle classes, stale geometry, full-frame preservation, distinct catalogue/evaluation labels, and exclusion of heldout labels from model selection. Browser rendering was not visually tested.

## Enable on the ST station

Install the optional official checkpoint; its SHA-256 is pinned and checked before loading:

```bash
.venv/bin/python scripts/prepare_outline_model.py
```

On Windows use `.venv\Scripts\python.exe` instead. Restart the app after updating the code. In the station select the intended source, load `config/st-outline-calibration.json` with the calibration JSON control, then open a recording. Alternatively import it in **Калибровка** and choose **Сохранить и применить** for the appropriate source. The station's existing CUDA requirement is unchanged. The checkpoint is installed locally, not committed as a 54 MB binary.

The supplied profile matches these ST recordings only. `config/st-outline-references.json` records the 78 source frames, candidate identities, catalogue intervals, split families and manufacturer source URLs; its hash is embedded in the profile. It contains no asserted physical ground truth. The original `config/st-calibration.json` remains available as the previous estimator.

In this workspace, the checkpoint is installed and the new profile has been saved as the local recorded-video configuration (`web_app/data/operator-calibration.json`); no previous saved operator profile existed. Reload the application to pick it up. This local setting is ignored by Git; other installations must load the supplied JSON themselves. No live IP-camera service was restarted.

## Reproduce the development check

These commands use the earlier local video audit and corrected full-frame evidence:

```bash
.venv/bin/python scripts/extract_outline_masks.py \
  --comparison runs/catalogue_verify_20261008/comparison.json \
  --manifest runs/catalogue_verify_20261008/manifest.json \
  --output-dir runs/outline_runtime_20261008/masks --device cpu

.venv/bin/python scripts/benchmark_outline_calibration.py \
  --comparison runs/catalogue_verify_20261008/comparison.json \
  --manifest runs/catalogue_verify_20261008/manifest.json \
  --masks runs/outline_runtime_20261008/masks \
  --output-dir runs/outline_runtime_20261008
```

`results.json` contains every fold, prediction, review and failure. `calibration.json` is loadable by the station; `references.json` records catalogue provenance. Missing masks stop export rather than silently removing difficult cars. The original full-duration audit and masks remain local under ignored `runs/`; these large inputs are needed to reproduce the benchmark.

The six real capture checks and clean corrected photos are under `runs/outline_runtime_20261008/captures/`, with their timings and source frames in `summary.json`.
