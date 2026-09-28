# RT-DETR and RF-DETR on the supplied videos

Follow-up: [catalogue dimensions versus saved estimates](CATALOGUE_COMPARISON_2026-09-28.md) shows approximately 13 cm underestimation on three of four provisional vehicle matches. The temporal residuals below must not be interpreted as physical length error.

**RT-DETR X at 640 px is a useful additional detector for this installation.** It passed the temporal consistency check on all five moving examples, including the truck that the YOLO26 M baseline withheld. RF-DETR Medium and Large did not improve overall consistency in this sample. This is **not verification of ±10 cm physical vehicle-length accuracy**.

RT-DETR L and X are now selectable in the calibration editor. They use the station's existing frame processing, road polygon, perspective correction, temporal capture and review workflow. The default remains YOLO26 M because it has better passenger-car temporal residuals and lower CPU cost. RF-DETR remains an offline comparison backend; it was not promoted to the live station.

## Controlled comparison

- 300 identical original frames: six 50-frame clips from all three supplied recordings.
- Five configurations: YOLO26 M 640, RT-DETR L 640, RT-DETR X 640, RF-DETR Medium 576, RF-DETR Large 704. RF variants use their native resolutions.
- 1,500 timed inferences, excluding one warm-up per model. Confidence 0.30; vehicle classes only; identical final duplicate suppression.
- Fixed `config/st-calibration.json` lens, road polygon, empirical scale, ±10 px capture gate and ±30 px temporal band. This deliberately uses the existing calibration to isolate detector changes. The later ruler-derived survey profile still fails its geometry checks and was **not** substituted or silently approved.
- CPU, four PyTorch threads, OpenCV one thread; PyTorch 2.14.0+cpu, Ultralytics 8.4.150, RF-DETR 1.11.0. CUDA/TensorRT throughput was not measured.
- All five variants associated a target box with the earlier YOLO target in each of 300 frames at IoU ≥0.5. This is an association check, **not annotated detection recall or localisation accuracy**.

Official APIs/checkpoints: [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/) and [Roboflow RF-DETR](https://github.com/roboflow/rf-detr/blob/1.11.0/docs/learn/run/detection.md). RF-DETR was installed in a separate benchmark environment. Its sparse COCO class IDs and RGB input contract are handled explicitly; tests guard against confusing them with YOLO's IDs/BGR input.

## Results

The passenger-car statistic below is the mean of each pass's maximum residual around its robust local length fit. The truck statistic is its maximum residual. They measure consistency, not error against an independently measured length.

| Model | Passenger mean worst residual | Worst passenger residual | Truck residual / result | Median CPU inference |
|---|---:|---:|---:|---:|
| YOLO26 M 640 | 1.83 cm | 2.96 cm | No supported measurement at line | 56.7 ms |
| RT-DETR L 640 | 2.38 cm | 3.08 cm | 25.39 cm — review | 115.0 ms |
| **RT-DETR X 640** | **2.50 cm** | **3.93 cm** | **4.65 cm — consistent candidate** | **230.2 ms** |
| RF-DETR Medium 576 | 3.88 cm | 7.09 cm | 33.34 cm — review | 97.2 ms |
| RF-DETR Large 704 | 4.39 cm | 6.83 cm | 31.10 cm — review | 144.9 ms |

RT-DETR X also has a slightly lower mean raw near-line length range on the four passenger cars: 4.44 cm versus YOLO26 M's 4.76 cm. YOLO remains better after the temporal trend fit. No model dominates every metric.

For the truck, RT-DETR X's six usable nearby frames give a 6.606 m **unvalidated candidate**, 7.08 cm raw range, 4.65 cm maximum fit residual and 1.19 cm alternating-frame fit difference. The near barrier hides part of the tyres, and physical ground-contact/bumper geometry remains uncertain. A consistent box cannot resolve that uncertainty.

The off-line stationary van generated no capture for any model. There are no independent vehicle-length labels, so neither absolute length error nor an accuracy percentage can be computed.

![Detector comparison](../runs/detr_20260928/benchmark/detector_comparison.png)

## Actual station save-path verification

A separate replay used the real RT-DETR factory, original-video decoding, original capture gate, temporal neighbourhood and `station.capture`. It used an isolated SQLite database and an explicit offline CPU override; the application's live CUDA requirement was preserved.

| Example | Saved candidate | Saved result |
|---|---:|---|
| SUV | 4.625 m | Temporal consistency passed; operator review required |
| Hatchback | 3.759 m | Temporal consistency passed; operator review required |
| MPV | 4.364 m | Temporal consistency passed; operator review required |
| Small SUV | 4.044 m | Temporal consistency passed; operator review required |
| Truck | 6.606 m | Temporal consistency passed; operator review required |
| Stationary van | None | No measurement-line evidence |

The records retain `rtdetr-x` as their detector and preserve the selected original frame, box, calibration and temporal evidence. These are estimates from the existing calibration, not verified dimensions. No existing production records or active camera settings were overwritten.

**209 Python tests and 34 frontend tests pass.** New tests cover RF colour/class conversion, duplicate/invalid boxes, missing-target denominators, rejection of empty output as a measurement, RT-DETR factory routing, model caching, capture-source validation and selector state. JavaScript syntax, Python compilation and whitespace checks pass.

## Use and reproduce

In the app, open **Калибровка → Detector → RT-DETR X**, choose **640 px**, rerun detection, and review the calibration before applying it. RT-DETR L is also available. The app fetches the allowlisted official checkpoint on first use. GPU inference remains required for the live application.

The test profile is exported locally at `runs/detr_20260928/rtdetr-x-profile-review.json`. It changes the existing ST profile's detector only; it is not a newly validated calibration.

The comparison environment for this session is `runs/detr_20260928/env`. For a separate setup, install the project dependencies and the optional `requirements-detr-benchmark.txt`, then obtain the official local checkpoints. Benchmark model specifications use `name`, `family`, `weights`, and `resolution`; families are `yolo`, `rtdetr`, `rfdetr-medium`, and `rfdetr-large`.

```bash
runs/detr_20260928/env/bin/python scripts/benchmark_detr.py \
  --profile config/st-calibration.json \
  --cases runs/accuracy_10cm/cases.json \
  --models runs/detr_20260928/models.json \
  --reference-detections runs/accuracy_10cm/all_model_comparison.json \
  --output-dir runs/detr_20260928/benchmark \
  --confidence .3

.venv/bin/python runs/detr_20260928/replay_capture.py
```

Evidence remains local under `runs/detr_20260928/`:

- `benchmark/manifest.json`: complete profile, video/checkpoint/reference hashes, versions and execution settings.
- `benchmark/model_comparison.json`: every frame's detections, timings and measurement statuses.
- `benchmark/comparison_summary.json`: temporal samples, rejected evidence and per-case results.
- `benchmark/coverage_summary.json`, `benchmark/compact_summary.json`: association counts and aggregate statistics.
- `benchmark/*_*.jpg`: same-frame visual comparisons.
- `production_capture/summary.json`: real save-path output; `replay_capture.py` reproduces it in the isolated database.
- `benchmark_requirements.txt`: exact installed packages, including inherited project packages.

The underlying calibration limitations documented in the road-mark and multi-view reports remain unresolved. Changing detector families improved the truck's repeatability, but does not establish the requested absolute ±10 cm accuracy.
