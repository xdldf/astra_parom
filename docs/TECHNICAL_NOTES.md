## Multi-operator update (2026-09-18)

Run one Uvicorn process bound to `0.0.0.0:8000`. Multiple browsers share the IP receiver/inference pipeline and encoded MJPEG buffers. Async stream generators do not reserve a FastAPI worker thread per viewer. Switching a browser to recorded-video mode no longer stops shared cameras. Video-file playback sessions remain independent (at most four); use one playback producer to avoid duplicate observations from different frames.

SQLite uses WAL and short write transactions. Connections commit/rollback and close on context exit. The first connection migrates existing data transactionally: a compact indexed `vehicle_list` table is maintained by insert/update/delete triggers, including OCR writes. Original vehicle JSON, stored prices and audit history remain intact. Queue queries filter in SQL, enforce `1 <= limit <= 250`, and return summaries without calibration/evidence blobs. `offset` and a returned upper-rowid `snapshot` anchor older pages against new inserts. Status/category edits can still change filtered page membership. Totals cover the filtered snapshot, not just the visible page; CSV streams all matching rows in batches of 250. ETags skip unchanged queue/detail/customer responses; the UI preserves unchanged rows and dirty forms.

Edits/payments retain transactional version checks. `station_id` scopes customer-screen publication to a workstation; the browser's **Место** field and `?page=client&station=...` link pair its display. Names and station IDs are audit/routing labels, not authentication. Use a trusted LAN.

Calibration profiles optionally contain `metric_rulers: [{points: [[x,y],...], step_m: 1}]` in full-resolution corrected-image coordinates. Marks represent real surveyed equal intervals along vehicle travel at one road depth. The model interpolates cumulative metric position within each ruler and subtracts endpoint coordinates; it interpolates effective pixels/metre between bracketing depth rulers. It never extrapolates outside horizontal/depth coverage (a single ruler allows ±0.05 normalized depth). This corrects varying scale across the frame without assuming one center scale for the whole vehicle. Rulers override legacy bbox references; old profiles still use the depth-linear approximation below. Browser CSS scaling never changes stored coordinates. This remains an empirical side-on bbox estimate, not surveyed 3D body geometry; field validation is required.

Profile fits are cached; identity lens transforms avoid full-frame remapping. Capture now passes the corrected frame internally and encodes only saved crops/evidence, avoiding a full-resolution JPEG→base64→JPEG-decode round trip. Image encoding runs before the database write lock.

Reproduce the synthetic queue benchmark with `python scripts/benchmark_station.py --cars 10000`; it uses a temporary database. The September test (10,000 records, 4 KB calibration fixture each) returned 250 rows / 69,373 bytes instead of 50,388,890 bytes for the previous full-history scan. Median page request: 7.89 ms; unchanged request: 1.20 ms / zero body bytes; prior scan plus serialization: 167.07 ms. These are local CPU/API measurements, not camera/GPU throughput claims.

## IP cameras and recorded-video modes

The operator source selector now switches between independent recorded-video and network-camera configurations. See [Russian setup instructions](../README_RU.md#8-подключение-настоящих-ip-камер). `/api/ip` owns a single persistent server-side station: two FFmpeg receivers with open/read timeouts and reconnects, bounded JPEG buffers, receive-time pairing with an adjustable offset, independent preview/vehicle/plate workers, and server-side automatic passage capture. Captures keep their front-frame evidence on disk for recognition after disconnect. Closing the browser does not stop IP processing; switching back to video explicitly stops it. Optional autostart applies when the server process starts, not when Windows boots.

Pairing uses server receive time, not hardware exposure timestamps. The correction needs on-site validation. Fresh side frames and matching side calibration resolution are required for length measurement. Without a matched front frame, a side-only review record is saved with a missing-front note. Front recognition and preview continue independently when the side camera is unavailable; no length is invented. Reconnecting only the front camera does not reset side tracking. Network URLs stay in the local `web_app/data/ip-cameras.json` file (plaintext, including any credentials); API reads return only configured flags/hosts. Keep this folder private. Camera calibration can be imported independently, and a side snapshot can be downloaded for calibration before starting the station. Physical cameras have not been tested yet; two local RTSP publishers, loss/recovery, automatic captures and saved recognition evidence have been exercised.

Инструкция на русском: [README_RU.md](../README_RU.md).

## Local GPU number plate recognition

HiWatch now has independent CUDA plate detection and OCR while both videos keep playing. FastALPR uses the official YOLOv9-t 640 plate detector and CCT-s global v1 OCR. Vehicle measurement still uses YOLO26n. Install `requirements-web.txt` and then `requirements-gpu.txt`; ONNX Runtime is pinned to 1.23.2 for the existing CUDA 12/cuDNN 9 PyTorch build. Models download once from the upstream GitHub releases into ignored `web_app/data/plate-models`. Footage and recognition stay local. Both inference sessions are checked for CUDA; initialization errors are exposed in the UI.

Each paired capture automatically queues five front frames spanning ±0.6 seconds. Two-line plates get a contour-based perspective correction and row reflow OCR alternative. Exact text votes select the leading suggestion; original crops, frame indices, detector confidence, OCR confidence and audit entries are retained. OCR confidence is a model score, not a measured accuracy guarantee. Saved jobs marked queued resume after a server restart.

Select a vehicle to see **Распознавание номера**. Click a plate image to expand it, use **Использовать номер** to copy a reading into the editable number field, then save/confirm normally. **Распознать снова** processes an older front photo or its recorded frame window. Multiple candidates remain visible; recognition never overwrites an operator-entered number or a confirmed/paid record. A common timestamp does not establish cross-camera vehicle identity: the operator must match the two vehicle photos. Suggestions appear as plain Cyrillic plate text; the review status remains visible. They are not published to the client as confirmed numbers.

Tested on the supplied HiWatch recording: `C900HT14` was recovered on three of five frames around frame 2480. Warm recognition took approximately 30–40 ms per sampled frame. A 12-second paired playback check with YOLO and OCR active reported 23.9 display FPS, 9.2 vehicle inference FPS and 20.5 ms source-frame alignment error, without a stream/OCR error. Blurred, occluded, distant and clipped plates can be incomplete or wrong and require correction. Tests: 101 Python and 8 browser-logic tests passed.

## Synchronized two-camera station

The supplied ST and HiWatch recordings are configured as a pair. Both cameras use one server clock, one Play/Pause button, and one timeline. At side-camera elapsed time `t`, HiWatch uses `t + 3 seconds` because its file starts three seconds earlier. The nearest front frame is selected using its own FPS. The UI reports the source-frame time difference after the offset. A slider can adjust the offset; changes restart both streams at the current side time. The recordings stop when their common valid interval ends.

Automatic and manual captures now save a padded corrected side crop plus the full synchronized front frame. The original media IDs, frame indices, elapsed times, calibration and offset are stored with the record. Both saved images expand on click; the customer screen uses the saved pair after confirmation. Previous records without front images remain unchanged. A synchronized front frame may include several vehicles; temporal pairing alone does not prove which front-view vehicle corresponds to the side detection. Plate OCR is integrated as described above; cross-camera identity verification remains an operator step.

Camera configuration is in ignored `web_app/data/camera-pair.json`; uploaded recordings and calibration stay local. The interface supports replacing either video. Default filenames are `ST_2026-09-04_19-30-04.mp4` and `HiWatch_2026-09-04_19-30-01.mp4`, found directly in Downloads. A 12-second two-camera benchmark produced 288 paired display frames and 133 GPU inferences; frame-time alignment error at the last sample was 19.8 ms. Browser rendering/network delays are separate from this source-frame alignment statistic.

## File camera stream

The operator now displays a server-generated MJPEG stream. One persistent decoder reads the recording sequentially, paced to its source FPS. A separate GPU worker consumes only the newest full-resolution frame; display and inference do not wait for each other. Lens maps are cached. The display is corrected at up to 1280 pixels wide; detection and metric measurement retain the full calibration resolution. Boxes older than 0.5 seconds are hidden. The operator shows display and inference FPS separately.

Pause closes the stream; seeking restarts it at the selected frame. Disconnected sessions expire after 30 seconds. This is a local HTTP camera-like stream from a file, not an RTSP network camera connection. A 12-second run of the supplied ST recording produced 293 display frames and 123 GPU inferences (including model startup), with no error.

## Continuous operator measurement

The operator uses one corrected camera view with YOLO26n boxes and lengths on CUDA GPU 0. Open a video, import **Калибровка JSON**, and press **Пуск**. **Автоматическое измерение** is enabled by default. A valid road and known-length references are required. Calibration is saved on the local server; resolution mismatches are rejected on import. The newest supplied calibration was loaded from `roadscale-calibration (1).json`.

Eligible measured vehicles automatically enter the review queue. When a line is configured, the bbox center must reach it. If sampled observations straddle the line, the estimated crossing frame is decoded and detected again; only a valid actual-frame measurement is saved. Short-term bbox overlap association suppresses repeat captures of a visible passage; occlusion, tracking loss, seeking/replaying, and multiple operator tabs can still produce duplicates requiring review. Plate evidence is now attached by the separate OCR worker described above.

The file is played against a video clock as a simulated camera source. Only the corrected processed feed is displayed; IP cameras are available through the separate mode described above. Inference skips stale frames when overloaded, so the displayed frame rate depends on processing throughput. Start playback in one operator tab to avoid duplicate runs.

# Recorded-video vehicle length metrology

## Ferry operator station

The home page now follows the supplied `demo2.html` operator/client/report workflow. Run the same web server command below and open http://127.0.0.1:8000. The measurement studio remains available at http://127.0.0.1:8000/calibration and in the **Калибровка** tab.

- **Калибровка:** load footage and the camera calibration, detect on the NVIDIA GPU, select a car and click **Отправить в очередь**. Play opens the operator video player; playback runs continuously while GPU inference samples the current video time independently. The server recomputes length from the submitted bbox/calibration and stores a cropped side photo plus the original measurement metadata. Identical media/frame/bbox submissions are idempotent; different frames are separate observations, not automatic unique-vehicle tracking.
- **Оператор:** select a queued car, enter its plate, verify its category against vehicle documents and adjust length if needed. The original bbox measurement remains visible. Auto prices use the supplied Appendix 1; a manual price or changed details require a reason. Save, confirm or reject. Confirmation publishes the record to the customer screen. **Отметить полученную оплату** records an operator-reported payment after confirmation; this does not process a payment or issue a fiscal receipt. Paid records cannot be edited. A version check prevents one browser from overwriting another operator's newer changes.
- **Экран клиента:** displays the most recently published confirmed/paid vehicle for the selected workstation and follows backend updates, including in a separate `/?page=client&station=<workstation-code>` window. Selecting or editing an unconfirmed row does not publish it. Editing or rejecting the displayed record clears it until it is confirmed again.
- **Отчёты:** filter by local registration date (Yakutsk UTC+9), category and status. Reports show confirmed/paid sums separately from payments marked received; CSV uses UTF-8 BOM and semicolons for Excel and neutralizes formula-like text. The full tariff reference is included.

The station starts empty and persists records, audit events and the selected client record in `web_app/data/station/station.sqlite3`. Photos live beside the database; retain both when backing up. Front-camera photos can be attached manually. Front-camera pairing and local GPU plate recognition are supported. A cash register and payment gateway are not connected. The operator name is a local audit label, not an authenticated identity. Run one server for all browsers on the trusted local network.

There are **26 tariff rows**, including section 6 from order 37/ПЯ effective 2026-09-08. Rows 6.6–6.8 use explicitly entered load capacity in tonnes, never inferred from image length. The server uses literal decimal ranges without rounding across gaps: for example passenger lengths between 4.00 and 4.01 m, buses at exactly 10.01 m, and road-train lengths between 12.00 and 12.10 m need operator resolution. The special GAZ/light-truck wording in 7.2 is noted for operator review and can be applied with a reasoned manual tariff. No rates were invented for unlisted lengths (including lowbed combinations over 16 m). Lengths within 0.10 m of a boundary are flagged; this margin is an operator attention threshold, not a statistical uncertainty bound. Approximate bbox estimates are never automatically charged.

Station API/price tests are in `tests/test_tariffs_station.py`.

## Visual bbox calibration studio

The new **Roadscale** web interface implements an empirical bbox-length approximation alongside the original surveyed 3D workflow described below.

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-web.txt
uv pip install --python .venv/Scripts/python.exe -r requirements-gpu.txt
.venv/Scripts/python.exe -m uvicorn web_app.main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000. Everything is drawn on the image; no polygon coordinates or coefficient text entry is needed.

YOLO26n always uses NVIDIA GPU 0 (`device=0`). Install the CUDA-enabled PyTorch build from `requirements-gpu.txt` after installing the web dependencies. Detection returns a clear error if CUDA is unavailable or GPU inference fails; there is no CPU fallback. Image decoding, lens correction, drawing and calibration calculations still use their existing CPU/browser implementations.

1. Open an image or recorded video. Adjust radial/edge correction, focal scale, zoom and lens-center sliders using the grid and a physically straight barrier. This is a manual radial lens approximation, not a surveyed intrinsic calibration. The barrier itself must be straight for this visual check to be meaningful.
2. Click **Draw road**, place at least four boundary dots in order, then **Finish**. Drag vertices to refine. Near/far boundaries are evaluated locally at the bbox bottom-center x, including sloped road boundaries. Lens changes clear coordinate-dependent calibration.
3. Click **Detect · YOLO26n**. The first inference downloads the official `yolo26n.pt` weights (network required); subsequent detections reuse the loaded model. Detection runs on the same corrected full-resolution image used for all pointing and measurements. Cars, motorcycles, buses and trucks are included. Inference requires a working NVIDIA CUDA GPU. A manual bbox drawing tool is also available.
4. Click a car, adjust the **Known length** slider to its independently measured length, and click **Use as reference**. Repeat at a different road depth (another video frame or a same-resolution image from the same fixed camera). One reference provides constant scale only. At least two distinct depths are needed for depth correction. Import/export preserves lens, polygon, references and resolution.
5. Inspect each car's width, road depth, cm/px, near-edge multiplier and approximate length. Export the current frame's measurements to JSON. Video controls preview frames; detection is explicit per selected frame, not persistent vehicle tracking.

### Measure at a side-on position

The **Tilt · left / right (degrees)** slider in step 1 rotates the lens-corrected image from −30° (left/counterclockwise) to +30° (right/clockwise), in 0.1° steps. Drawing and YOLO detection use that same rotated image. Tilt is saved in calibration exports; older profiles default to 0°. Finish tilt adjustment before drawing the road: changing it clears coordinates, the measurement line and reference cars. The output resolution stays fixed, so rotation can expose black corners and crop image edges.

Click **Place measurement line**, then click the image where cars appear side-on. Drag the purple vertical line to refine its position. A measurement is eligible only when `abs(bbox_x + bbox_width / 2 - line_x) <= tolerance_px`; simply overlapping the line with a bbox edge does not qualify. The shaded band shows the adjustable center tolerance (default ±10 original-image pixels). Road depth still uses the bbox bottom center. Place the line based on the vehicle's actual view: the line does not automatically estimate or guarantee a 90-degree viewing angle.

**Play** in the embedded calibration studio opens continuous playback in the operator view, using the current calibration and frame. The operator can also upload video directly. Native video playback does not wait for inference. A separate corrected preview displays timestamped GPU results; busy inference skips to the current video time rather than queuing old frames. The line filters measurements without pausing. This is not unique-vehicle tracking; fast crossings can be missed between processed frames. Disable the line to restore measurements across the road. Line settings are included in calibration import/export. Use references at the selected line when collecting new calibration samples.

Road drafts with four or more points are now validated and saved automatically when proceeding to detection, reference calibration, or line placement. Incomplete drafts stay visible and prompt for more points. Editing a known length does not reset the road. Uploaded media remain accessible after a server restart; browser calibration still needs export before reloading the page.

Frontend regression tests: `node --test tests/test_workbench_ui.cjs`.

The fitted relationship is `pixels_per_meter(t) = a + b*t`, where `t=0` is the local far edge and `t=1` is the local near edge. References supply `bbox_width_px / known_length_m`. Least squares fits `a,b`; nonpositive intercept/slope and insufficient depth separation are rejected. Thus `cm_per_pixel = 100/(a+b*t)`, `coefficient = (a+b)/(a+b*t)`, and `length_m = bbox_width_px/(a+b*t)`. Farther cars get larger cm/px. This is a deliberately simple empirical curve, not recovered camera geometry. Measurements beyond reference depths are labeled extrapolated; outside-road and image-clipped boxes have no length. Occlusions and heading bias are not automatically corrected. Resolution matching does not prove two images came from the same camera.

The 6 m camera height and 12 m road offset are context, not sufficient metric calibration by themselves. Validate against held-out measured cars at multiple depths and headings. Do not treat reference-fit accuracy as independent measurement accuracy. Uploads and downloaded weights are local under ignored `web_app/data/`; uploaded media remain accessible after server restart. Export calibration to retain it. Run locally, not as a public upload service.

YOLO26 model reference: https://docs.ultralytics.com/models/yolo26/

The remainder of this README describes the original 3D metrology mode; its restrictions and implementation status do not describe the new bbox studio.

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
