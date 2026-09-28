## Working-mode multi-frame crossing recovery — 2026-09-28

- **216 Python tests and 39 frontend tests passed.** Added regression coverage for intermediate detections between browser polls, bounded per-viewer result history, real-frame crossing search through the API, duplicate persistence, ambiguous neighbours, adaptive temporal bands and IP recovery beyond the three nearest packets.
- Working video capture searches neighbouring original frames after a skipped crossing. IP recovery searches a bounded packet window. Temporal evidence expands only when the original narrow band lacks five samples; the actual line gate, depth support, association and stability checks remain required.
- Recomputed existing detections on 20 passages: RT-DETR X gives 19 numeric temporal estimates versus 9 before. Actual original-video replay saved three formerly rejected passenger estimates using 7–14 frames; an out-of-calibration truck anchor stayed unmeasured. These results are coverage/consistency evidence, not ±10 cm validation or measured live GPU throughput. [Details](docs/CROSSING_RECOVERY_2026-09-28.md).

## Detector selector fix — 2026-09-28

- Reproduced in the browser: selecting RT-DETR before opening media silently reverted to YOLO26 M because no profile existed. The selected model and resolution now carry into the next new media profile and its first detection request. Imported profiles keep their own detector settings.
- During frame processing, profile loading and saving, detector controls are visibly disabled with an explanation; they become available again after completion or failure. Updated the frontend asset version to load the fix after refresh.
- **37 frontend tests passed**, including regressions for selection before media, request parameters, error recovery and imported settings. Browser verification confirmed RT-DETR X and resolution remain selected on an empty calibration page. JavaScript syntax and whitespace checks passed.

## Additional September 7 recordings — 2026-09-28

- Final pre-push verification: **210 Python tests and 34 frontend tests passed**; Python compilation, JavaScript syntax and staged whitespace checks passed.
- Scouted 1,081 frames across all three new recordings; selected 20 two-second passages. YOLO26 M, RT-DETR X and RF-DETR Large processed the same 1,000 original frames (3,000 timed inferences). Stable benchmark estimates: 6/20, 9/20 and 7/20 respectively, not accuracy percentages.
- Actual isolated saved-capture replay: YOLO produced 6 numeric estimates; RT-DETR produced 10, including a trailer-only component that must not be treated as whole-combination length. All saved records retain operator review. Three provisional catalogue pairs give RT-DETR mean absolute difference 10.79 cm and maximum 26.44 cm; four other catalogue candidates have no accepted length.
- Fixed benchmark target association so a neighbouring car cannot supply the selected vehicle's missing length. **16 targeted regression tests passed** across benchmark, detector backends, temporal estimation and capture. The active calibration and production records were not changed by this evaluation.
- [Full report, timestamps, images, limitations and reproduction](docs/ADDITIONAL_VIDEOS_2026-09-28.md). ±10 cm physical accuracy remains unestablished.

## RT-DETR / RF-DETR comparison and selectable RT-DETR — 2026-09-28

- **209 Python tests and 34 frontend tests passed**; syntax, compilation and whitespace checks passed.
- Five models processed the same 300 original frames from all three videos (1,500 timed inferences). RT-DETR X 640 passed temporal checks on five moving examples, including the truck; RF-DETR M/L truck residuals were 33.34/31.10 cm. No absolute length accuracy is established.
- Added RT-DETR L/X to profile validation, the correct Ultralytics factory, model cache, calibration selector and saved-source schema. YOLO26 M remains the default. RF-DETR uses an isolated optional benchmark environment.
- Actual saved-capture replay with RT-DETR X persisted five estimates with operator-review status and complete source evidence; stationary van had no crossing. Live CUDA requirement retained; GPU throughput not measured. [Full evidence and reproduction](docs/DETR_COMPARISON_2026-09-28.md).

## Road position and multi-frame landmark reconstruction — 2026-09-28

- **202 Python tests and 33 frontend tests passed**; JavaScript syntax and whitespace checks passed.
- Workbench inspection now shows the local near/far road cross-section and survey span. Relative road depth is explicitly distinguished from metric camera distance.
- Added an offline 3D projection/landmark alternative. Independent synthetic cases recover physical length across depths and headings; failed calibration, occlusion and inconsistent landmarks reject measurements.
- Actual SUV tests at 7 and 10 original frames reject: alternating frame sets disagree by 73.6 and 61.3 cm. Longer movement did not establish ±10 cm. No production calibration or records changed in this follow-up. [Evidence and reproduction](docs/ROAD_POSITION_AND_MULTIVIEW_2026-09-28.md).

## Road marks and lens correction — 2026-09-28

- User confirmed 1 m barrier marks and 5.80/5.50 m widths; heights above road are unknown. Added a raw-coordinate survey, fitted correction, held-out ruler/post checks and a projective measurement path.
- 300 original frames from all three videos rerun with the new lens. The candidate fails geometry acceptance (9.71 cm maximum held-out ruler error, 19.86 cm maximum all-mark residual, 43.78 cm maximum held-out post-transfer disagreement). Numeric lengths are withheld; this is not ±10 cm vehicle validation.
- Python: **192 passed**; frontend: **32 passed**. Existing preset retained; failed candidate exported for review. [Details and artifacts](docs/ROAD_MARK_CALIBRATION_2026-09-28.md).

## ±10 cm larger-model and temporal follow-up — 2026-09-28

- Python suite: **186 passed**; frontend: **31 passed**. Added tests for model/resolution routing, original-video neighbourhoods, saved temporal evidence, insufficient frames, camera reconnects, delayed/stale capture, uncalibrated review records, and preservation of the original line gate.
- N/M/L at 640/1280 ran on **300 identical original frames across six clips from all three videos** (1,800 inferences). M 640 reduced the black SUV raw length range from 11.31 to 2.40 cm. The four passenger-car examples had temporal maximum residuals of 1.21–2.96 cm. These are repeatability metrics, not physical errors.
- Real saved-capture replay with M 640, confidence 0.30 and original video decoding persisted four approximate multi-frame lengths, one truck review with no length, and no false crossing for the stationary off-line van. Used an isolated database and explicit offline CPU override; live CUDA throughput was not tested.
- **Absolute ±10 cm is not established**: no independently measured lengths; the eight legacy references still disagree with their fitted curve by up to 16.21 cm. The loaded truck is unresolved. See [full evidence and reproduction](docs/ACCURACY_10CM_2026-09-28.md).

## Supplied-video accuracy audit — 2026-09-28

- Python suite: **175 passed**; frontend tests: **30 passed**. New checks cover duplicate reference weighting, conflicting lengths, calibration depth support, final detector duplicate suppression, explicit ±5 cm acceptance accounting and offline audit outputs with blank independent truth.
- Actual CPU inference on all three supplied ST recordings: 540 distributed frames, 150 dense crossing frames with the final path, plus preliminary sparse/sequence comparisons. Video originals were not modified. See [the full audit](docs/ACCURACY_AUDIT_2026-09-28.md) for hashes, sampling scope, catalogue checks and reproduction commands.
- One SUV varies by **11.31 cm over 0.12 s** with the updated system; the ±5 cm target is therefore **not met**. Physical accuracy remains unvalidated because no independently measured vehicle lengths or surveyed road distances were available. Regression success is not metrology validation.

## Human-approved calibration references — 2026-09-21

- Full Python suite: **167 passed**. Station UI: **11 passed**; workbench UI: **15 passed**.
- Tests reject every status without a human confirmation audit, require independent length attestation, cover old approved records and paid records, stale-version rejection, automatic JSON merging/deduplication and provenance, coordinate mismatch, missing source data, and rejection/withdrawal/reconfirmation.
- A running profile stops using a revoked reference without a restart. Eligibility is cached outside the per-frame database path and invalidated on operator updates. Verified same-lane samples use a local median scale and reject measurements outside their observed lane band; meter rulers retain priority. Legacy manual-reference JSON remains compatible.
- An isolated synthetic browser preview verified the initially empty actual-length field, saving a verified reference under the named operator, automatic addition in the calibration editor, and JSON export. No production records or camera settings were used. Physical accuracy remains unverified.

## Camera calibration and JSON continuation — 2026-09-21

- Targeted Python tests: **29 passed** (`tests/test_ip_cameras.py`, `tests/test_bbox_workbench.py`). Workbench UI: **15 passed**; station UI: **9 passed**. Python compilation, JavaScript syntax and whitespace checks passed.
- Camera tests cover reuse of a fresh, original receiver image without stopping capture, stale-frame rejection, snapshot acquisition before the first calibration, and saved-profile retrieval. The workbench persists the original JPEG and renders lens correction once through its existing frame path.
- Frontend tests cover importing JSON before an image, loading the station's saved profile, preserving calibration across camera snapshots, safe resolution mismatch handling, editing imported reference lengths/ruler spacing/ruler points, and retaining edits when applying to a running IP station is rejected.
- An isolated browser preview with a synthetic 1200×800 camera image loaded the saved calibration, displayed its overlays, edited ruler spacing, applied it to the test station, and restored the edited value after reloading. Controls and image scroll independently on desktop. Production settings and camera connections were not used.
- No physical camera or GPU measurement validation was available. These checks establish the calibration workflow and coordinate preservation, not real-world vehicle measurement accuracy.

## Optional tariffs — 2026-09-18

- Station regression tests: **77 passed**; camera tests: **10 passed**; station UI tests: **9 passed**.
- Tests cover persisted shared configuration, cache invalidation, disabled tariff calculation during capture and confirmation, preserving historical prices when toggling, CSV columns, unsaved form preservation, re-confirmation before payment after re-enabling, and rejecting a stale operator's confirmation after a mode change.
- An isolated browser preview confirmed a measurement with tariffs off, displayed its photos and length without a price on the client screen, and hid tariff columns and money totals in reports. Production settings and records were not used.

## Truck photo timing and category guidance — 2026-09-18

- Full Python suite: **146 passed**. Front-history tests cover an eight-second cab/body separation, bounded age/count/bytes, tracking loss, reconnects, different readable plates, and rejecting a mismatched recorded-video session.
- Capture integration preserves the original simultaneous photo and stores an earlier tracked cab photo with its time difference. Persisted live OCR survives loss of the camera buffer and avoids a second GPU OCR pass; explicit re-read processes the selected passage samples again.
- Uploading a replacement photo clears old front evidence; a late OCR result from the replaced photo cannot overwrite it.
- A 17.16 m generic truck remains unpriced until the operator chooses a supported category. Choosing tractor/semitrailer produces the existing 7.4 tariff (14,000 RUB); this is a software test, not verification of the photographed vehicle's classification or true length.
- Seven station UI tests passed. Browser checks in an isolated synthetic database verified the earlier-photo time caption, synchronized-photo viewer, and operator category selection showing 14,000 RUB.
- No real truck video, independently measured full length, or on-site GPU test was available. Tracking remains a heuristic requiring operator confirmation; short-reference calibration warnings do not establish physical measurement accuracy.

## Multi-operator, tariff and ruler update — 2026-09-18

Validation performed in an isolated Linux Python 3.12 environment with OpenCV 4.14.0.94, NumPy 2.2.6, SciPy 1.17.1, FastAPI 0.141.1 and CPU PyTorch for mocked detector tests. Production GPU requirements were not changed.

- Python regression suite passed; final targeted runs cover the additional parallel-capture and stream-viewer cases. Existing Brown/fisheye calibration, geometry, video, OCR, tariff and persistence tests passed.
- Fifteen frontend tests passed (`test_live_tracks.cjs`, `test_workbench_ui.cjs`, `test_station_ui.cjs`). They exercise ETag reuse without replacing queue rows, paging/filter resets, dirty-form preservation, customer-screen routing, source switching, and original-image coordinates on a half-size preview.
- Concurrent confirmations/payments of one record produce one success and one HTTP 409 with one audit event. Twelve independent records can be confirmed concurrently onto separate customer displays. Four simultaneous captures produce one record and one photo. Eight simultaneous camera starts share one station. Forty-eight asynchronous stream consumers share pre-encoded frames and wait for changes.
- A legacy database migrates without changing historical prices; connections close after use. A 301-car fixture returns at most 250 lightweight rows, preserves totals and older-page boundaries against inserts, and exports all records to CSV. OCR updates invalidate ETags.
- Synthetic rulers recover unequal pixel intervals corresponding to equal metric distances, interpolate depth, reject uncovered regions and invalid geometry, and retain lengths after consistent coordinate scaling. These tests establish numerical behavior, not real-world vehicle accuracy.
- Browser verification used an isolated 301-record database: queue showed 250 rows, next page showed 51, a 25-tonne capacity tariff saved as 10,950 ₽ (6.7), reports paginated, and ruler controls loaded without JavaScript errors.
- `python scripts/benchmark_station.py --cars 10000`: prior full-history scan plus serialization 167.07 ms / 50,388,890 bytes; capped page median 7.89 ms / 69,373 bytes; unchanged request median 1.20 ms / empty response body. Synthetic 4 KB calibration payload per record; results are local API performance, not GPU throughput.

Live camera/GPU throughput, several physical operator computers, and independent real-vehicle measurement accuracy require on-site verification. Meter marks must be physically surveyed; browser clicks alone do not establish scale.

# Executed verification

## Commands actually run

`bash scripts/run_tests.sh tests -q`

Final result: **32 passed in 3.62s**.

`python -m vehicle_metrology.synthetic --output-dir examples/synthetic`

Executed with `.venv/Scripts/python.exe`. Created a real local MJPG AVI with 120 synthetic frames, camera artifact, noisy observation sidecar and ground-truth CSV. Three passages of one synthetic rigid 4.5 m body at different lateral positions/headings. This is generated test data, NOT camera footage or detector accuracy evidence.

`python test.py --video examples/synthetic/SYNTHETIC.avi --calibration examples/synthetic/calibration.json --observations examples/synthetic/observations.json --ground-truth examples/synthetic/ground_truth.csv --mc-samples 100 --headless --output-video runs/synthetic/overlay.avi --output-dir runs/synthetic`

Executed with `.venv/Scripts/python.exe`. Returned complete, 120 decoded frames, 3 tracks. Independently reopened saved overlay and decoded all 120 frames. Preview is runs/synthetic/preview.png. Reviewed visible diagnostic overlays and moved measurement labels away from landmark labels.

Synthetic conditional results:
- synthetic-1: 4.502120506338303 m
- synthetic-2: 4.497968726323529 m
- synthetic-3: 4.506265615879156 m
- Mean absolute error: 0.0034724652979768087 m
- Observed maximum absolute error: 0.006265615879155639 m

These small errors reflect perfect synthetic camera/road geometry with small seeded image noise and correct point correspondences. They do NOT predict real accuracy, tail error or coverage.

Repeated the processing command into runs/synthetic_repeat with identical parameters. Parsed result JSON objects compare exactly equal, including after final integration fixes. Monte Carlo had 100/100 successful perturbations per track; its intervals explicitly exclude calibration/survey/shared biases. No confidence-coverage guarantee follows from three synthetic passages.

Opened an actual OpenCV HighGUI window with an exported preview; `WND_PROP_VISIBLE` returned 1.0; closed it. Keyboard playback state and annotation edit serialization are tested programmatically. Manual operator pointing, full interactive seek fidelity on camera-specific codecs and desktop scaling remain unverified.

CLI `--help` for test.py, calibrate.py and annotate.py executed successfully. Calibration tests execute real OpenCV Brown/fisheye fitting and surveyed pose recovery with independent check points. Another test creates an encoded checkerboard AVI and extracts its board corners. No real checkerboard recording was available for a full installation calibration.

## Remaining acceptance gates

1. Specify required absolute/relative error, high-percentile target and allowed rejection rate.
2. Supply original recorded traffic and calibration videos, measured road controls/checks/elevations and independently measured vehicle lengths.
3. Validate the manual rich-landmark reconstruction across the site's usable region and viewing angles; identify actual dominant errors through controlled ablations.
4. Implement and independently validate semantic detection, contact/extremum localization, occlusion-aware tracking and richer joint 3D fitting only after geometry is established.
5. Propagate common calibration/survey/model errors and validate interval coverage; do not use conditional pixel-noise intervals for tariffs.

No claim of production automation, real-world precision, tariff readiness, or universal observability is made.
