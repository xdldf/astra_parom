# Recorded-video vehicle length metrology

Read DESIGN.md first: it derives the geometry, observability limits, selected methodology, calibration protocol, alternative approaches and experimental validation. INTERFACE.md defines camera/landmark coordinates.

## Status: runnable conditional geometry proof of concept, not production automation

The project reconstructs physical front/rear longitudinal extrema in 3D from calibrated motion and persistent manual observations. It does NOT estimate meters from a bounding box. No actual camera recordings, survey or measured real vehicles have been supplied. Synthetic accuracy is not evidence of real traffic accuracy.

Implemented:
- Local recorded files only; raw distorted Brown/fisheye projection and ray inversion.
- Checkerboard calibration from recorded training and separate heldout video; surveyed road-pose fit with independent metric checkpoints.
- Manual multi-vehicle landmark annotation, persistent track IDs, repeatable playback.
- Rigid multi-frame endpoint reconstruction with ground-contact-derived position/heading.
- Whole-track, prefix and sliding-window estimates; reprojection/conditioning/anchor diagnostics; explicit rejection.
- Optional seeded conditional pixel-noise Monte Carlo. This excludes calibration uncertainty and shared biases; it is NOT a certified confidence interval.
- Frame-by-frame foreground detection/association diagnostic, video overlays, JSON/CSV outputs and known-length evaluation.

Not implemented / not validated:
- Trained semantic vehicle detector, automatic accurate contact/extremum recognition or production occlusion-aware identity tracking. Unannotated foreground tracks output null length. Manual annotations are the reference experiment that separates geometric feasibility from automation error.
- Full jointly optimized vehicle surface/trajectory/shape with silhouettes, nonplanar road, articulation, pitch/roll, rolling-shutter correction, camera drift correction, or shared calibration covariance propagation.
- Population confidence intervals/cluster bootstrap, automatic distortion-model selection, tariff decisions, or guaranteed accuracy/coverage. Fixed rejection thresholds are POC heuristics, not deployment acceptance limits.

## Environment

Windows Python 3.11 environment has been created at .venv. No live camera, network access or model download is required to run tests or demos. For recreation:

    uv venv .venv --python 3.11
    uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt

For commands below use `.venv/Scripts/python.exe` on Windows, or activate the environment and use `python`. On Linux/macOS use `.venv/bin/python`.

## Run the supplied synthetic recorded-video experiment

From C:\job\astra_paroms:

    .venv/Scripts/python.exe -m vehicle_metrology.synthetic --output-dir examples/synthetic
    .venv/Scripts/python.exe test.py --video examples/synthetic/SYNTHETIC.avi --calibration examples/synthetic/calibration.json --observations examples/synthetic/observations.json --ground-truth examples/synthetic/ground_truth.csv --mc-samples 100 --output-dir runs/synthetic --output-video runs/synthetic/overlay.avi

Add `--headless` to process/save without opening a window. Display opens after whole-track processing; playback is intentionally offline, not live inference. The synthetic file contains three passages of the same artificial 4.5 m rigid body, varying lateral position and heading; sidecar observations contain seeded pixel noise. It tests video I/O and conditional geometry, not automatic recognition.

Space: pause/resume. N or period: next frame. B or comma: previous frame. R: replay from start paused. Q/Escape: quit viewer. Closing the viewer does not cancel already completed output generation.

To inspect any real file immediately:

    .venv/Scripts/python.exe test.py --video your_video.mp4 --output-dir runs/inspection --output-video runs/inspection/overlay.mp4

This runs diagnostic foreground detection, not semantic vehicle classification, and does not produce metric estimates without calibration and annotated geometric evidence. IDs from this baseline can split/merge and are not reliable under occlusion.

## Real-camera procedure

### 1. Define acceptance limits and measurement policy

State maximum acceptable errors, target tail error, allowed rejection rate and tariff thresholds. Define bumper-to-bumper length, protrusions/load/trailer treatment, ground-truth measurement uncertainty and classes to support. Do not use catalog vehicle dimensions without checking exact configuration.

### 2. Intrinsic calibration from recorded checkerboard video

Use a rigid measured board. Lock resolution, lens focus, zoom, crop, digital stabilization and camera dewarp. Gather diverse tilts/depths/positions including image edges; use separate poses/recording for heldout checks. Recommended starting data quantities and experimental rationale are in DESIGN.md.

    .venv/Scripts/python.exe calibrate.py intrinsics --video board_training.mp4 --heldout-video board_heldout.mp4 --cols 8 --rows 6 --square-m 0.04 --stride 15 --model fisheye --output calibration/intrinsics_fisheye.json

Rows/columns are INNER CORNERS. Square size and board dimensions above are EXAMPLES: use your physically measured board. Repeat with `--model brown` and compare heldout residuals by spatial location and independent metric checks. Do not pick a lens model based only on training RMS. The CLI requires at least 10 training and 3 heldout detections, but counts do not establish diversity/quality. Repeated near-identical views are not sufficient. This implementation supports checkerboards, not ChArUco.

### 3. Survey road controls and independent checkpoints

Record a road frame in the same video mode. Survey points in a road-plane coordinate frame in meters; Z=0. Place origin/axes explicitly. Need raw image pixels matched to each point. Use at least 6 control and 3 noncollinear independent check points for this CLI; use the more redundant layout in DESIGN.md for the real experiment. Check points are NEVER used in pose optimization.

Survey JSON schema:

    {
      "schema_version": 1,
      "image_size": [1920,1080],
      "units": "m",
      "road_polygon": [[-10,0],[10,0],[10,8],[-10,8]],
      "control_points": [{"id":"control-1","xyz_m":[0,0,0],"uv":[900,700]}],
      "check_points": [{"id":"check-1","xyz_m":[1,2,0],"uv":[950,650]}]
    }

The snippet shows structure only; one point in each list is deliberately insufficient. Supply your measured redundant points, not these illustrative numbers. Choose the check tolerance from your error budget; 0.02 m below is an example, not a justified deployment threshold.

    .venv/Scripts/python.exe calibrate.py survey --intrinsics calibration/intrinsics_fisheye.json --survey calibration/survey.json --calibration-id camera-A-v1 --max-check-error-m 0.02 --output calibration/camera.json

Camera artifacts contain K/D, Rcw/tcw, pixel size, road polygon, metric check errors and input hashes. Independent point residuals identify bad controls/model mismatch but do not certify endpoint measurement. Survey road elevation first; this version rejects supplied nonzero Z rather than pretending to model road crown. Define a planar patch only where justified. An arbitrary oversized polygon is not proof the scene is calibrated there.

### 4. Annotate a small reference set before automating

    .venv/Scripts/python.exe annotate.py your_video.mp4 --output data/observations.json --track-id vehicle-001 --sigma-px 1.0

Pause on a clear frame. Click in this order:
1. Rear wheel-road contact.
2. Front wheel-road contact on the SAME SIDE.
3. A persistent physical rearmost body point.
4. A persistent physical frontmost body point.

Enter commits/saves the frame. N/B step, space plays/pauses, R replays, U undoes last point, C clears pending points, D deletes current frame observation, S saves, Q saves/quits. Rerun with another `--track-id` to add another vehicle while preserving existing tracks; Tab cycles existing IDs. Select at least four well-separated usable views; more diverse observations are preferable. Prefix/window estimates may be rejected when a short window has insufficient baseline.

Do NOT substitute arbitrary silhouette extrema that change physical correspondence. Do NOT guess hidden contacts. Do NOT track rotating tire material points. The points must actually represent longitudinal extremes, not just headlights near the bumper. The POC requires all four points in each selected frame; partially visible observations without all four are not currently supported. Body pitch/roll, steering/contact changes and articulations violate its simplified rigid planar pose assumptions.

Annotations are bound to SHA-256 of the entire source video. Cropping/resizing/re-encoding a file invalidates that sidecar; never relabel the hash to bypass coordinate verification. The tool uses original image pixels; desktop DPI scaling/codec seeking must be checked on your footage. No automatic distortion removal is applied to annotations.

### 5. Process and evaluate independently

    .venv/Scripts/python.exe test.py --video your_video.mp4 --calibration calibration/camera.json --observations data/observations.json --ground-truth data/truth.csv --mc-samples 200 --seed 0 --output-dir runs/experiment-001 --output-video runs/experiment-001/overlay.mp4

Ground truth CSV, measured independently:

    track_id,vehicle_id,length_m
    vehicle-001,physical-car-A,4.52

Those numbers illustrate the CSV only. Repeat passes use different track IDs and the same physical vehicle_id. Length truth is consumed after measurement and does not affect estimates.

Outputs:
- results.json: configuration, hashes, versions, frame timestamps, track status, lengths, diagnostics, conditional uncertainty and optional evaluation.
- measurements.csv: one row per track; JSON diagnostics field.
- track_frames.csv: flat trajectory, heading, per-frame residual, prefix length, sliding-window length.
- frames.json / frames.csv: video-indexed detection/annotation/measurement data.
- evaluation.json: signed/absolute/relative errors, mean/median/RMSE/maximum/p90/p95/p99, rejection coverage, physical-vehicle groups, position/heading bins, trajectory-window errors and conditional interval coverage.
- overlay AVI/MP4 if requested. Whole-track values use all observations; window values expose temporal behavior and must not be misread as independent measurements.

Use a NEW output directory per parameter experiment; reusing a directory overwrites result files. Output/input collisions are rejected. Source data should still be archived separately. MP4 exports use a constant FPS timeline even if original PTS are variable; measurements are indexed by original decoded frame, not export timing. Recorded clips are decoded twice for fitting then display/export; diagnostics are held in memory, suitable for trial clips rather than unbounded archives.

Spatial summary bins reuse each track once per visited cell. They describe track-level accuracy conditional on visiting cells, not independent local estimates. Use the flat window errors / track_frames.csv to study within-trajectory changes. Per-vehicle bootstrap confidence intervals and confidence-vs-coverage curves are future analysis extensions; current percentiles are descriptive only.

## Tests and evidence

    bash scripts/run_tests.sh tests -q

The test suite exercises real OpenCV projection/distortion inversion, Brown/fisheye intrinsic fits, surveyed pose and independent-check rejection, elevated endpoint geometry, lanes/headings, no-baseline rejection, seeded uncertainty, actual video encoding/decoding, overlays, playback state controls, annotation serialization, CLI integration and known-length metrics. Source videos generated for tests are synthetic and labeled as such. Test-first failures were observed before implementation.

The Windows GUI was smoke-tested with an actual visible OpenCV window; keyboard state transitions and annotation edits are tested programmatically. Full manual pointing accuracy, desktop scaling, and interactive codec seeking on real camera footage still need operator verification.

## Next decision gate

Supply one original camera clip plus candidate intrinsic/road survey data and independently measured lengths. Use the manual reference experiment to determine whether contacts, rigid extrema and viewpoint baseline are sufficient. If not, change the observation model or sensing arrangement before training an automatic pipeline. A second calibrated viewpoint or independent ranging sensor may be required for the requested tolerance/coverage. A low residual or a stable number alone is not evidence of correct physical length.
