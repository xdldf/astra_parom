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
