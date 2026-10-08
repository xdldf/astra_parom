# Reference corrections and temporal outline check

The less-than-10-cm requirement for every car is **not achieved**. This update corrects five earlier visual catalogue identifications, retrains the existing outline estimator, and tests whether additional neighbouring frames reduce errors. The revised profile is `config/st-outline-calibration.json`, calibration ID `st-catalogue-outline-20261008-reviewed`.

## Five reference mistakes

The earlier review labelled V1-050, V2-017, V2-020, V2-034 and V2-046 as Corolla Spacio E120. Comparing their full corrected frames with Toyota's own photographs supports **first-generation Wish** instead: the long rear quarter glass, nearly straight window beltline, nose and rear lamps below the quarter glass differ from the Spacio body. These are revised visual candidates, not verified VINs or trims.

Toyota lists the representative early [Wish at 4,550 mm](https://www.toyota-global.com/company/history_of_toyota/75years/vehicle_lineage/car/id60007858/index.html), and the [2005 facelift X at 4,560 mm](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-WISH/200509/10029625/). The [Spacio archive](https://www.toyota-global.com/company/history_of_toyota/75years/vehicle_lineage/car/id60004359/index.html) supplies the body comparison. The previous Spacio interval was 4.240–4.275 m; the revised Wish interval is 4.550–4.560 m. No target was selected from the estimator's output.

V2-020 has an apparently missing/damaged front bumper cover. It remains in all 78 comparison rows, with a stock-body uncertainty note, but **does not train coefficients or support bounds**. Agreement with a stock catalogue would not validate that damaged car's actual length.

`config/st-catalogue-corrections.json` retains old and new identities, manufacturer URLs, visual reasons, the original audit hash and each reviewed corrected image's hash. The original audit and its historical results remain intact. Both Wish generations now share one validation family, reducing the number of families from 15 to 14. The exported model trains on 77 passages; the reference manifest retains all 78, including the excluded damaged car.

This review was prompted by the residuals. These videos remain development data; correcting labels does not turn them into an untouched validation set. Other identities, variants, modifications and full vehicle completeness remain unverified.

## Comparable coefficient check

The following comparison holds the **revised evaluation intervals, 14 family splits, seven image features and damaged-body training exclusion fixed**. It changes only whether the training targets use the old or revised catalogue labels. In each fold, the tested family is excluded from normalization, coefficient fitting and inner regularization selection. Vehicle identity is never an inference feature.

| All 78 passages, before applicability guards | Old training labels | Revised training labels |
|---|---:|---:|
| Mean absolute distance to catalogue midpoint | 6.68 cm | 5.98 cm |
| Outside 10 cm even for the nearest catalogue value | 17 | 11 |
| Within 10 cm for the entire interval | 55 | 61 |
| Variant-dependent | 6 | 6 |
| Largest distance to catalogue interval | 19.47 cm | 22.60 cm |

The average improves and fewer cases miss, but the worst case gets worse. This is not the earlier historical 15-family score; those old numbers used different labels and splits and cannot be treated as an identical test set.

Applying actual runtime support and identity guards gives **57 within the entire catalogue interval, 8 outside, 5 variant-dependent and 8 review-only**, out of 78. Reviews are unsuccessful automatic measurements. Average midpoint error on the 70 numeric values is 5.70 cm, not a fleet-wide accuracy guarantee. The largest reported mismatch remains V6-008, provisionally Prado J150 2017, about 22.60 cm outside its catalogue length. The damaged V2-020 remains a catalogue mismatch, not a hidden exclusion.

All 78 heldout predictions and review branches reproduced through the actual `apply_outline` estimator with their fold coefficients and cached masks, to a maximum numerical difference below 1e-14 m. The shipped all-reference profile is distinct from these heldout-fold models.

## Five-frame experiment

350 original-video frames were decoded and corrected at 2592×1944, then segmented with the same pinned model. All selected masks succeeded. The selection spans up to five already associated temporal observations and always includes the saved anchor. 68 passages have five frames; the other ten retain their single anchor and remain in the denominator. No catalogue length or estimated error chooses a frame.

| Original-video pixels, revised references, all 78 | Midpoint MAE | Outside 10 cm | Largest interval distance |
|---|---:|---:|---:|
| Single anchor | 6.08 cm | 12 | 22.67 cm |
| Mean features | 6.06 cm | 12 | 23.06 cm |
| Median features | 6.03 cm | 13 | 22.73 cm |
| Robust local fit at anchor position | 6.00 cm | 13 | 23.07 cm |

Short-window aggregation does not repair the remaining errors. It was **not enabled in production**. This tests neighbouring frames in the existing measurement band, not a full-passage reconstruction from physical landmarks. The original-video anchor and earlier audit JPEG can yield slightly different silhouettes; their results are reported separately rather than treating that difference as a temporal gain.

## Reproduction and artifacts

Run the revised single-frame benchmark without recomputing masks:

```bash
.venv/bin/python scripts/benchmark_outline_calibration.py \
  --comparison runs/catalogue_verify_20261008/comparison.json \
  --manifest runs/catalogue_verify_20261008/manifest.json \
  --masks runs/outline_runtime_20261008/masks \
  --catalogue-corrections config/st-catalogue-corrections.json \
  --output-dir runs/outline_reviewed_20261008
```

For the temporal experiment, run `scripts/benchmark_temporal_outline.py extract` with the same `--comparison` and `--manifest`, `--output-dir runs/temporal_outline_20261008` and an explicit offline `--device cpu`. Then run its `benchmark` command with the same paths and `--catalogue-corrections config/st-catalogue-corrections.json`. The extractor records input/video/model hashes, mask hashes, the exact decoded corrected-pixel hash, and clean full corrected JPEGs. A missing/changed selected mask stops scoring rather than deleting a difficult passage. A different extraction implementation requires a new output directory for an extraction rerun.

Local evidence:

- `runs/outline_reviewed_20261008/results.json`: revised folds, predictions, old-target comparison and guards.
- `runs/outline_reviewed_20261008/runtime_validation.json`: actual-estimator parity for all 78.
- `runs/outline_reviewed_20261008/captures/`: actual saved-capture replays from each of the six videos, isolated database and full corrected photographs. These are integration checks with the exported profile, not heldout accuracy.
- `runs/temporal_outline_20261008/extraction.json`: all 350 masks and frame provenance.
- `runs/temporal_outline_20261008/results.json`: four temporal comparisons after the reference correction.
- `runs/temporal_outline_20261008/original_labels_results.json`: the temporal result before the label review.

The revised recorded-video profile is also installed locally, with the previous local profile backed up under `runs/outline_reviewed_20261008/previous-operator-calibration.json`. Restart/reload the app to use it; other installations should load the revised JSON as described in `OUTLINE_CAPTURE_2026-10-08.md`. No live IP-camera service was restarted and GPU throughput was not tested.

Verification: **333 Python tests passed**. New tests ensure damaged stock labels cannot influence any fitted fold, revised Wish generations remain in one family, image-bound label corrections preserve evidence, and frame selection/aggregation cannot choose samples using catalogue targets. Frontend code was unchanged.
