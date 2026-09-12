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
