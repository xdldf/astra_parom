# Additional September 7 videos — evaluation on 2026-09-28

**All three additional recordings were used. The system still does not demonstrate ±10 cm vehicle-length accuracy.** RT-DETR X supplies more stable estimates on these clips, but the likely Patrol is about 26 cm shorter than its catalogue length. Stable estimates and physical accuracy are separate tests.

## Scope and reproducibility

- Original files: `ST_2026-09-07_15-30-01.mp4`, `ST_2026-09-07_16-00-04.mp4`, `ST_2026-09-07_16-30-04.mp4`. All 2592 × 1944, approximately 25 fps; roughly 90 minutes total.
- Scouted 1,081 frames at five-second intervals across all three files. The 87 near-line observations are candidate observations, not a vehicle census. Fast passages between samples can be missed.
- Visually selected 20 passages spanning passenger cars, vans, modified vehicles, trucks and trailers. Each clip contains 50 consecutive original frames: 1,000 frames × 3 detectors = 3,000 timed inferences, plus explicit warm-up and scouting/replay inference.
- The existing `config/st-calibration.json` was held fixed, including lens correction, road polygon, relative road-depth scale and the original line gate. No new catalogue value was added to calibration.
- Confidence 0.30; YOLO26 M / RT-DETR X at 640, RF-DETR Large at its native 704. CPU-only offline evaluation; live GPU requirements remain in place. This is not GPU throughput verification.
- Cases, original video/model/profile hashes, package versions and raw detections are in `runs/additional_videos_20260928/`. The machine-readable comparison is `catalogue_comparison.json`.

## Results for every selected passage

Numbers below are accepted temporal **estimates in metres**, not verified lengths. Dashes retain failed cases in the denominator. This table uses the benchmark anchor closest to the measurement line. Saved-capture replay is reported separately below.

| Case / file time / offset | Vehicle description | YOLO M | RT-DETR X | RF-DETR L |
|---|---|---:|---:|---:|
| new_00 / 15-30-01 / +109.6s | White Prado, no sunroof | — (few frames) | — (few frames) | — (few frames) |
| new_01 / 15-30-01 / +328.8s | White Prado, sunroof | 4.762 | 4.768 | 4.713 |
| new_06 / 15-30-01 / +929.2s | Silver Corolla Fielder | — (few frames) | — (few frames) | — (few frames) |
| new_08 / 15-30-01 / +1055.0s | Silver Ipsum, roof load | — (few frames) | — (few frames) | — (few frames) |
| new_10 / 15-30-01 / +1368.8s | Modified silver Honda Fit | 3.739 | 3.773 | 3.742 |
| new_15 / 15-30-01 / +1734.2s | Dark Corolla Fielder | — (few frames) | — (few frames) | — (few frames) |
| new_18 / 16-00-04 / +15.0s | Hiace, roof rack | — (few frames) | 5.258 | — (no line evidence) |
| new_19 / 16-00-04 / +29.2s | Bare three-axle chassis | — (few frames) | 8.282 | 8.250 |
| new_27 / 16-00-04 / +209.2s | White Ipsum | — (few frames) | — (few frames) | — (few frames) |
| new_28 / 16-00-04 / +234.2s | Black Nissan Patrol Y62 | 4.837 | 4.876 | 4.834 |
| new_34 / 16-00-04 / +1140.4s | Lumber flatbed | — (unstable) | 9.083 | — (unstable) |
| new_41 / 16-00-04 / +1300.4s | Land Cruiser 200 towing | — (few frames) | — (few frames) | — (few frames) |
| new_48 / 16-30-04 / +225.0s | White box truck | 7.012 | 7.070 | 7.045 |
| new_49 / 16-30-04 / +341.2s | Blue Vitz XP130 facelift | 3.901 | 3.942 | 3.889 |
| new_51 / 16-30-04 / +615.0s | L200 pickup, front accessory | — (few frames) | — (few frames) | — (few frames) |
| new_53 / 16-30-04 / +649.6s | Older silver wagon | — (few frames) | — (few frames) | — (few frames) |
| new_54 / 16-30-04 / +704.2s | Blue Rush / Be-go | — (few frames) | — (few frames) | — (few frames) |
| new_58 / 16-30-04 / +875.0s | Probox / Succeed wagon | 4.048 | 4.027 | 4.021 |
| new_65 / 16-30-04 / +1035.0s | Small trailer component | — (unstable) | — (few frames) | — (unstable) |
| new_71 / 16-30-04 / +1420.0s | Tractor / container semitrailer | — (no line evidence) | — (few frames) | — (few frames) |

| Detector | Stable temporal estimates | Review / no estimate |
|---|---:|---:|
| YOLO26 M 640 | 6/20 | 14/20 |
| RT-DETR X 640 | 9/20 | 11/20 |
| RF-DETR Large 704 | 7/20 | 13/20 |

The main rejection is fewer than five usable frames inside the ±30 px temporal band. Long vehicles also encounter calibration-depth limits or inconsistent boundaries. Relaxing these checks would increase output coverage without establishing its accuracy.

## Catalogue length differences

Identities are visual inferences, not confirmed chassis/year/trim identifications. Manufacturer dimensions describe the stated candidate; accessories, body conversions and towing combinations can change actual overall length. No physically measured ground truth is available. Negative differences mean an underestimate.

YOLO and RT-DETR columns use the real `station.capture` path in isolated databases, selecting the first usable target-associated detection at the original line. RF-DETR is the offline benchmark result; it is not integrated into live capture.

| Case / likely model | Catalogue (m) | Saved YOLO m / difference cm | Saved RT-DETR m / difference cm | Offline RF-DETR m / difference cm |
|---|---:|---:|---:|---:|
| new_00: [Toyota Prado J150, 2017 facelift](https://toyota.jp/pages/contents/carlineup/archive/landcruiserprado/2017-09/pdf/landcruiserprado_main_s_201709.pdf) | 4.825 | — | — | — |
| new_01: [Toyota Prado J150, 2017 facelift](https://toyota.jp/pages/contents/carlineup/archive/landcruiserprado/2017-09/pdf/landcruiserprado_main_s_201709.pdf) | 4.825 | 4.762 / -6.3 | 4.768 / -5.7 | 4.713 / -11.2 |
| new_08: [Toyota Ipsum ACM2x](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-IPSUM/200706/10041622/) | 4.690 | — | — | — |
| new_27: [Toyota Ipsum ACM2x](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-IPSUM/200706/10041627/) | 4.690 | — | — | — |
| new_28: [Nissan Patrol Y62](https://www.nissan.co.mz/media/lbmddaef/patrol-rhd-eng.pdf) | 5.140 | 4.837 / -30.3 | 4.876 / -26.4 | 4.834 / -30.6 |
| new_49: [Toyota Vitz XP130, 2017 facelift](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-VITZ/201701/10108200/) | 3.945 | 3.901 / -4.4 | 3.942 / -0.3 | 3.889 / -5.6 |
| new_54: [Toyota Rush J200 / Daihatsu Be-go candidate](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-RUSH/200601/10031995/) | 3.995 | — | — | — |

Only **three of the seven catalogue candidates** have numeric estimates with each detector; four have no accepted length. Among those three, YOLO and RT-DETR each have two differences within 10 cm; RF-DETR has one. These selected, conditional comparisons are not a fleet accuracy percentage.

| Detector | Compared numeric cases | Mean absolute catalogue difference | Maximum difference |
|---|---:|---:|---:|
| YOLO26 M 640 | 3 | 13.70 cm | 30.32 cm |
| RT-DETR X 640 | 3 | 10.79 cm | 26.44 cm |
| RF-DETR Large 704 | 3 | 15.79 cm | 30.58 cm |

The other passenger vehicles were withheld from exact catalogue scoring because the generation/variant or accessories are unresolved. In particular, the Honda Fit is visibly modified, the commercial wagon could be Probox or Succeed, the Hiace body-length variant is unknown, and the pickup has a front accessory. Truck bodies and loads cannot be assigned a true length from a generic model name.

A single additive correction cannot put both the conditional Patrol and Vitz comparisons inside ±10 cm: even their best shared offset (+13.36 cm) leaves a minimum worst difference of 13.08 cm. Their allowed global scale intervals also do not overlap. This diagnostic was not applied to the calibration.

## Saved capture and complete vehicle limits

- YOLO26 M 640: 19 saved review records out of 20 selected passages; 6 have numeric temporal estimates. Status counts: temporal_review=13, temporal_consistent=6, no_usable_line_evidence=1.
- RT-DETR X 640: 20 saved review records out of 20 selected passages; 10 have numeric temporal estimates. Status counts: temporal_review=10, temporal_consistent=10.
- Replay storage, source frames, exact bboxes, temporal evidence and record IDs are under `production_capture/`; the actual production database was not used.
- `new_41` contains a towing Land Cruiser and `new_65` targets only a trailer component in a later recording. They are not established as the same combination. A component box cannot be interpreted as total towing-combination length. The towing Land Cruiser has no accepted length. In contrast, RT-DETR saved a stable **3.796 m trailer-only estimate** in `new_65`; this is not a successful full-combination measurement.
- The trailer also exposes anchor sensitivity: the benchmark chooses the closest line frame and rejects four usable observations; the saved-capture replay starts at the first usable line frame and accepts a different associated set. Thus RT-DETR has 9/20 numeric benchmark estimates but 10/20 in saved replay, including the invalid-for-combination trailer component. All saved numeric records retain the operator-review status.
- The container tractor/semitrailer `new_71` also has no accepted length. No 40-foot-container guess was used as vehicle truth.

## Camera position and evaluation correction

Local barrier/pole patch checks against the earlier view found median displacements of 0.22, 0.66 and 1.19 original pixels in the three recordings. Only 3, 2 and 3 of 14 patches per image met the correlation threshold, so this is a limited alignment check, not proof of unchanged calibration. A low-inlier barrier homography was rejected. No alignment warp was applied.

The road polygon and relative depth of each box are used by the existing scale model. This does not recover metric camera distance or a vehicle’s 3D bumper positions. Unknown barrier heights and the failed ground-plane transfer from the earlier ruler experiment remain unresolved.

Fixed an evaluation bug: the length summary previously could choose a neighbouring car when the selected target lacked a usable detection. Summaries now match the selected target independently at each reference frame (IoU ≥ 0.5), preserving missing detections. Raw detections remain available. A regression test covers the neighbour-at-line case.

Target association used earlier YOLO detections, not manually labelled truth. Of 1,000 clip frames, 810 pass the central-target reference association rule; the others are absent or too far from its central box. YOLO and RT-DETR match 810/810, RF-DETR 809/810 within that restricted denominator. These are association counts, not recall, and selection from YOLO scouting can omit vehicles missed by YOLO.

## Visual evidence

Images are vehicle-identification aids; displayed crops may be resized. Measurement uses original frames and recorded coordinates.

![First ten selected passages](../runs/additional_videos_20260928/selected_0.jpg)

![Second ten selected passages](../runs/additional_videos_20260928/selected_10.jpg)

## Reproduction and checks

```bash
runs/detr_20260928/env/bin/python scripts/benchmark_detr.py \
  --profile config/st-calibration.json \
  --cases runs/additional_videos_20260928/cases.json \
  --models runs/additional_videos_20260928/models.json \
  --reference-detections runs/additional_videos_20260928/reference_detections.json \
  --output-dir runs/additional_videos_20260928/benchmark --confidence .3
.venv/bin/python runs/additional_videos_20260928/replay_capture.py
.venv/bin/python runs/additional_videos_20260928/write_report.py
.venv/bin/python -m pytest -q tests/test_detr_benchmark.py tests/test_detector_backends.py tests/test_temporal.py tests/test_temporal_capture.py
```

Targeted regression result: **16 passed**. These checks validate code behavior, not physical measurement accuracy. Existing model/profile settings were retained after this evaluation.
