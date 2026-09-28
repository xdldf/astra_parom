# Measurement accuracy audit — 2026-09-28

Follow-up: the user relaxed the target to ±10 cm. See [the larger-model and multi-frame report](ACCURACY_10CM_2026-09-28.md) for the current implementation and remaining limits.

**Requested target: every vehicle within ±0.05 m. Result: not achieved or independently validated.** The user confirmed that no measured vehicle lengths or surveyed road distances are available. No physical ground truth has been invented from catalogue guesses or predictions.

## What was inspected

The three supplied, unchanged recordings are 2592 × 1944 HEVC, approximately 25 fps, 45,000 frames and 30 minutes each. We inspected 36 distributed road-strip images, processed 366 half-second samples in five selected sequences, ran the final measurement code on 540 evenly distributed frames (every 10 seconds), and processed another 150 frames at the original 25 fps around three line crossings. These sets overlap. This was sampled analysis, not exhaustive playback, a vehicle census or full-file decode verification.

The offline comparisons used the bundled YOLO26n model with CPU PyTorch, 640 and 1280 input sizes. The live station still requires CUDA. Camera placement, lens parameters, source videos and recorded vehicle lengths were not changed. No site measurements or survey values were inferred from the apparent guardrail spacing.

| Source | SHA-256 |
| --- | --- |
| ST_2026-09-04_18-00-04.mp4 | `5ffac45e261f1784f08252680e08528f0a1db7209a99c6c00daa745f7d8a3869` |
| ST_2026-09-04_18-30-04.mp4 | `cdfe84a61556a94c725479dccb4f8bdc3a564786a87139e52ac2a26ab0166502` |
| ST_2026-09-04_19-00-04.mp4 | `7fa80db0ada41afd138efee5bed7b48f9ad602b3b4443bc20dbd3c606bd86727` |

Artifacts are under `runs/accuracy_20260928/`: contact sheets, native crops, detection JSON, calibration comparison, and the `audit`, `dense_prado`, `dense_hatchback`, and `dense_wish` reports. `inspection_manifest.json` identifies original paths, hashes and sampled frames. The supplied filenames are retained throughout.

## Demonstrated faults and changes

1. **Duplicate reference weighting:** the bundled calibration had 16 rows but only eight unique observations. Frame 7650 occurred nine times. Fitting now counts an observation once, rejects contradictory lengths for the same observation, and preserves separate frame observations. The bundled JSON contains the eight unique rows. Repeated “Use as reference” updates the existing manual observation instead of appending another copy. Legacy profiles are deduplicated during fitting too.
2. **Unsupported depth extrapolation:** a depth fit previously produced values beyond its reference range; a single reference covered the whole road. Measurements now return `outside_calibration` and no length beyond that support. Single-reference fits cover only a local ±0.05 road-depth band, matching the existing verified-local behavior. This deliberately reduces measurement coverage; that band is a support rule, not a precision guarantee.
3. **Duplicate detector boxes:** a vehicle in `18-30-04` at +1260 s had overlapping car/truck detections. Shared inference uses class-agnostic suppression and a final confidence-ordered IoU > 0.8 duplicate filter, including for end-to-end model versions. The sampled result falls from six detections to five while retaining the adjacent vehicles. This is not a general solution to occlusion or track identity.
4. **Hidden calibration inconsistency:** the API now reports residuals, unique counts and the ±5 cm reference-consistency count. The calibration editor visibly reports the maximum discrepancy; saved measurements carry a warning when references disagree by more than 5 cm. A good training fit is never labelled independently validated.
5. **Acceptance accounting:** the evaluator reports both measured-case pass rate and pass rate over all ground-truth cases. Missing or rejected cases cannot create a false 100% result. Duplicate prediction IDs and invalid numeric lengths are rejected.
6. **Reproducible video audit:** `scripts/audit_measurements.py` uses the same lens correction, detector output handling, measurement line and scaling as the station. It records hashes, exact sample IDs, rejection reasons and resolution sensitivity; it exports a blank ground-truth template and accepts independently measured truth separately. It never copies estimates into truth.

Existing saved station profiles also benefit from the fitting fix; the bundled JSON is not automatically installed over a user's existing settings. Historical captured measurements are not recomputed.

## Quantitative evidence

The following comparison is against the eight unique values already in the old calibration. Their independent physical provenance is unknown. These are **training residuals, not actual vehicle errors**:

| Quantity | Before deduplication | After deduplication |
| --- | ---: | ---: |
| Mean absolute reference residual | 11.06 cm | 8.21 cm |
| Maximum absolute reference residual | 22.49 cm | 16.21 cm |

The corrected fit still misses the requested tolerance on five of the eight calibration values. Deduplication repairs weighting; it does not prove better accuracy on unseen cars.

| Dense sequence | Accepted frame observations | Estimate range | Spread |
| --- | ---: | ---: | ---: |
| 18-00-04, +504 to +506 s; black SUV | 4 | 4.5684–4.6815 m | 11.31 cm |
| 18-00-04, +524 to +526 s; silver hatchback | 2 | 3.7233–3.7535 m | 3.02 cm |
| 18-30-04, +1264 to +1266 s; white MPV | 5 | 4.2891–4.3221 m | 3.31 cm |

The SUV observations span only +504.40 to +504.52 s. A fixed physical length cannot be within ±5 cm of both extremes of an 11.31 cm spread. At least one of these estimates therefore violates the requested tolerance, even without knowing its true length. Smaller spreads for the other cars establish repeatability over those frames, not absolute accuracy. The car-identification labels in directory names are provisional visual descriptions.

The ten-second audit produced 602 detection observations: 327 `waiting_for_line`, 263 `outside_road`, 11 `clipped`, one `outside_calibration`, and zero numeric lengths. Sparse sampling usually misses the ±10 px measurement window; this is not a live-station capture-rate estimate. The dense runs exercise successful numeric measurement through that same window. Do not widen the line merely to improve the audit count.

Across 327 associated road detections, the median absolute width change between 640 and 1280 inputs was 3.11 source pixels. Large tails include partial/full truck-envelope differences and possible association ambiguity. In the initial +510 s truck example from `19-00-04`, changing resolution shifted the old diagnostic estimate from 6.858 m to 7.835 m. That example was away from the measurement line and is not a recorded station measurement. It demonstrates that resolution changes alone do not establish accuracy; the production input size remains 640.

## Catalogue checks

Correction after close inspection of the measured passages: the compact SUV is a likely Toyota Yaris Cross, not the provisional Raize below, and the measured white wagon is a likely Corolla Fielder. See [the catalogue-versus-estimate comparison](CATALOGUE_COMPARISON_2026-09-28.md) for the corrected candidates, exact capture frames and signed differences. The Wish and white Vitz at +60 s are different vehicles from the measured white wagon and silver hatchback.

These are candidate identifications from appearance, not confirmed chassis/year/trim matches. Catalogue numbers were not added to calibration or independent ground truth.

| Visual candidate | Official published example | Source |
| --- | --- | --- |
| Grey MPV, `18-00-04` +60 s, consistent with second-generation Toyota Wish | 4.590 m for the listed 2009 representative grades | [Toyota vehicle history](https://www.toyota-global.com/company/history_of_toyota/75years/vehicle_lineage/car/id60000991/index.html) |
| White hatchback, same frame, consistent with Toyota Vitz | 3.885 m for the 2014 U grade; exact filmed grade unconfirmed | [Toyota used-car specification](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-VITZ/201404/10087952/) |
| White compact SUV, `18-00-04` +510 s; earlier Raize identification withdrawn after close inspection | Likely Yaris Cross: 4.180 m for the Japanese launch model | [Toyota catalogue specifications](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-YARIS_CROSS/202008/10130870/) |

Model year, body kit, accessories, protruding cargo and the measurement definition can change the relevant overall length. Internet specifications alone cannot establish the exact photographed vehicle's bumper-to-bumper length to 5 cm.

## Verification and limitations

- Full Python regression suite: **175 passed**. Existing API, station, camera, geometry, calibration, OCR and persistence tests remain green.
- Frontend tests: **30 passed**, including duplicate-reference prevention and the calibration warning.
- JavaScript syntax and `git diff --check` passed.
- The browser loaded the calibration editor and imported the eight-reference JSON successfully. The warning logic is covered by frontend tests; a complete browser image/measurement workflow was not verified in this run.
- Local API tests needed execution outside the sandbox because its socket policy blocks `socketpair()`, which FastAPI's in-process client needs.
- Actual inference was run offline on the supplied videos. No live GPU throughput, live camera capture or independently measured physical accuracy was validated.

## Reproduction

From the repository root, with the existing environment:

```bash
.venv/bin/python scripts/audit_measurements.py \
  --videos /home/user/Downloads/ST_2026-09-04_18-00-04.mp4 \
           /home/user/Downloads/ST_2026-09-04_18-30-04.mp4 \
           /home/user/Downloads/ST_2026-09-04_19-00-04.mp4 \
  --profile config/st-calibration.json \
  --output-dir runs/accuracy_audit_repeat --device cpu \
  --step-seconds 10 --compare-imgsz 1280
```

For the dense SUV case, use just `18-00-04`, a separate output directory, and `--start-seconds 504 --end-seconds 506 --step-seconds 0.04`. The other dense windows are +524–526 s in `18-00-04` and +1264–1266 s in `18-30-04`.

After independently measuring and associating vehicles, supply a CSV with `track_id,vehicle_id,length_m` using output sample IDs and `--ground-truth PATH`. Include rejected/missed annotated cases as well. Group repeated observations by physical vehicle and adjudicate whole passages before making vehicle-level coverage claims. The blank template lists measured observations as a starting point; it is not a complete validation set.

## What is still needed for ±5 cm

Measure stable road references across the actual measurement zone and independently measure vehicles representing the supported sizes, positions and configurations. Establish what is included in length (bumpers, cargo, trailer, towbar). Keep some physical vehicles entirely outside calibration for validation. Use the existing surveyed camera/3D landmark workflow to quantify lens, road and endpoint errors; one global bbox width/depth conversion is not sufficient evidence for centimetre precision. Detection/tracking and multi-frame endpoint estimation then need validation against those measurements, including rejected and missed passages. If the visible image cannot support the tolerance, camera geometry, exposure or sensing must change.

**The target remains unmet. The fixes and audit make the existing failure observable and prevent several false results; they are not a 100% accuracy claim.**
