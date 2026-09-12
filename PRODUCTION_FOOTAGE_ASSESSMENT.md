# Assessment of the supplied production-camera recording

## Evidence and scope

Original, unchanged source:
`C:\Users\alexe\Downloads\ST_2026-09-04_19-30-04.mp4`

SHA-256: `dcd6b64c6055598f6faf8cf225a82015f2fb64af089f1a19ee7d1f31fc254196`

FFprobe reports 2592x1944, HEVC Main, YUV420p, 25 fps, 45,000 frames, 1800 s, 590,008,873 bytes. Full-file FFmpeg decode completed with exit code 0 and no decoder warning/error output. This is a decode-integrity check, not a full visual/measurement validation.

Visual inspection used 16 evenly distributed frames over the recording, native-resolution crops, and a denser road-strip sequence from 100 to 144 seconds. No exhaustive vehicle census or ground-truth annotation was performed.

Artifacts: `runs/production_inspection/`
- `inspection_manifest.json`: original hash, metadata and sampled frame indices.
- `contact_sheet.jpg`: full-duration sparse sampling.
- `trajectory_100_144s.jpg`: road-strip temporal sequence.
- `frame_002999.jpg`: original-resolution decoded example at nominal 119.96 seconds (JPEG inspection derivative).
- `scene_plan.jpg`: candidate pilot region and static-registration region. Neither rectangle is calibrated or a validated operational boundary.
- `static_region_registration.json`: sparse camera-stability diagnostic.
- `pipeline_summary.json`: measured diagnostic-pipeline output counts.
- `pipeline_preview.jpg`: visual evidence of foreground-detector fragmentation.
- `pipeline_110_130s/overlay.mp4`: actual processed excerpt.

The 110–130 s excerpt was transcoded to H.264 at original image dimensions, 25 fps, CRF18 for a bounded diagnostic run. It is a derivative, not an original-quality measurement reference. Its sidecars must never be used as if bound to the original video. All precision annotation and later evaluation should use original decoded pixels/frame indices.

## Observations from the actual scene

### Favorable evidence

The road crosses the lower image, giving useful side/oblique vehicle views. In central-region samples, both near-side wheels and the front/rear body outline are visually identifiable for some passenger vehicles. This is materially better for a manual geometry experiment than a remote near-overhead view where contacts are hidden.

The dark passenger vehicle around 100–144 seconds moves from the right toward the center/left, with a pause/slow queue interval and later displacement. Its sequence offers candidate motion baseline, not just repeated stationary frames. The sampled road-strip images support selecting separated moving observations. Exact persistent bumper-extremum correspondence still requires operator confirmation; silhouette identity must not be assumed.

The camera appears broadly fixed across the sparse sample. ORB/RANSAC registration of a building/vegetation band against frame 0 found no obvious large image shift; all sample median displacement components were zero at this detector's coordinate granularity, with nonzero residuals. This does NOT establish subpixel stability, absence of vibration or absence of within-frame rolling shutter. Foliage in the mask and changing light are limitations. Stable surveyed landmark patches should replace this exploratory test for production drift monitoring.

### Material complications

1. Vehicles queue and stop. More elapsed video is not necessarily more geometric information; stationary observations add no translational baseline. A foreground model may absorb stationary vehicles. Motion-only identity tracking is inadequate.
2. The near guardrail overlaps the low road/vehicle region in portions of the view. Left-edge vehicles become clipped/occluded; some contacts and lower body boundaries cannot be treated as visible. The central region is a sensible pilot, not justification to ignore the rest of the required area permanently.
3. The road is unpaved and visibly irregular, with surface texture/ruts and transitions. One global plane is unverified. The image alone does not determine the size of elevation deviations. Survey the road before deciding between a plane, local planes or a mesh. The current POC supports only a plane and must not silently flatten this site.
4. Long trucks and tractor–trailer combinations occur. A combination is not a single rigid body when articulation changes. Determine whether the tariff uses trailer-inclusive straightened overall length, instantaneous envelope, or individual rigid-body dimensions. The current passenger-vehicle contact model is not a universal truck model.
5. Illumination changes markedly over the recording. Later samples have stronger headlight highlights and visibly smeared moving-vehicle detail compared with clearer earlier central views. Potential contributors include longer exposure, compression and denoising; their individual contributions cannot be identified from these samples alone. Frame rate is NOT exposure duration. Contact/bumper localization can become unreliable even though the nominal resolution remains unchanged.
6. Vehicles have roof loads/accessories and different body shapes. An extremal mask point can correspond to cargo or another protrusion instead of the specified bumper envelope. Define the physical measurement policy and annotate the correct surfaces.
7. The image is wide-angle with curved scene structures, but real guardrail/road curvature cannot be separated from lens distortion simply by treating those structures as straight. The building supplies candidate geometric checks, not a ready calibration target. Do not choose lens coefficients from visual appearance or assume all guardrail points are on the road plane.
8. Foreground/background occlusions, reflections, vegetation and image overlays contaminate background subtraction. The actual baseline run confirms this problem rather than just predicting it.

## Actual recorded-file pipeline test

Ran the existing `test.py` on the 20-second derivative at original dimensions, headless, saving overlay and structured outputs. Read the outputs back and reopened the encoded overlay.

- Input frames decoded by runner: 500.
- Saved overlay frames independently decoded: 500.
- Distinct diagnostic fragment IDs: 6842.
- Median foreground detections per frame: 179.5.
- Maximum foreground detections in a frame: 1283.
- Metric results: 0, correctly rejected without calibration/landmarks.

These counts are foreground components/association fragments, NOT vehicles or validated identities. The visual overlay contains many detections in vegetation, ground texture and vehicle subregions. The full-frame MOG2/greedy-center baseline is INVALIDATED as a production detector/tracker for this scene. This result does not invalidate calibrated metric multi-view geometry; that geometry was not exercised with real calibration because none exists yet.

Adding an ROI alone cannot solve stopping vehicles, contact recognition, body-end localization or persistent identity. It will help suppress irrelevant scene content but does not make background subtraction a metrology pipeline.

## Revised architecture decision

Retain the calibrated, rigid object-relative multi-view measurement principle, but do not assume the initial four-click planar model covers all production traffic.

1. Use semantic vehicle detection/instance segmentation, including stopped vehicles, inside a surveyed scene region. Select/evaluate the model on this actual camera, at an input resolution that retains small wheel/body details. No framework/model is selected solely on popularity. Segmentation is evidence for fitting, not a ground footprint by itself.
2. Use identity tracking that survives stops, partial occlusion and neighboring vehicles; test merges/splits independently of length accuracy. Large-frame fixed pixel-distance gates are not a sufficient production association rule.
3. Maintain separate visibility and uncertainty for ground contacts, wheel centers and body extremities. Contact points must be genuinely observed or estimated under an explicit wheel/pose model, not the bottom of a detector box. Support manual review in the initial dataset.
4. Reconstruct metric motion on the surveyed surface. Use planar patches only where the survey and error budget support them. Add joint body pose/shape refinement if pitch/roll/road deviations matter.
5. Use multiple sharp views of persistent physical bumper evidence and, where necessary, visibility-aware contour/surface constraints. Do not triangulate a silhouette tangent as a fixed material point. Sparse keypoints near bumpers alone do not guarantee actual extreme length.
6. Separate articulated parts and establish a consistent combination-length definition; otherwise explicitly reject those tracks from the rigid POC.
7. Gate on blur, clipping, contact/end visibility, angular baseline, calibration drift and reconstruction consistency. Retain both rejection coverage and accepted-case error. Low-light accepted coverage must be measured separately.
8. Use known actual vehicle lengths for held-out validation and constrained refinement only after scene/lens identifiability is established. No pixel-to-meter correction based on guessed make/model dimensions.

The camera view looks promising for a useful subset of passenger vehicles. No defensible claim of accuracy across the entire road, all classes or evening conditions is possible yet. If the required tolerance/coverage cannot be met due to hidden endpoints or blur, camera exposure/lighting/placement or additional sensing will need to change; software cannot recover absent information reliably.

## Minimum next collection for this specific site

### Required user decisions

- Required length error and allowed rejection rate, particularly near tariff boundaries.
- Which vehicle categories must be supported and how trailer/cargo/protrusion length is defined.
- Whether camera settings and access to the road for safe measurements are available.

### Survey and camera data

- Camera model and exact recording settings: lens/zoom/focus, stream resolution, crop/dewarp/stabilization, shutter/exposure limit and denoising. No credentials are needed.
- Calibrate actual pixel-to-ray mapping with a measured rigid target at varied angles/positions, including edges. Keep the production settings unchanged. Independent heldout target views are necessary.
- Establish a road coordinate frame and survey well-spread points along BOTH longitudinal extent and lateral width, covering the central pilot and near/far trajectories. Record X/Y/Z in meters plus a corresponding original camera frame. An initial practical plan is 12–20 fitting controls and 6–10 independent checkpoints; choose more if geometry/residuals require it.
- Survey elevations across tire paths and transitions. The upper edges of guardrails are not road-plane control points. If elevated structures are surveyed as 3D controls, use a compatible 3D calibration routine rather than assigning Z=0 in the current planar CLI.
- Obtain direct measured lengths of accessible vehicles. Repeated safe passes of the SAME vehicles across lateral positions separate positional bias from vehicle-class bias. Reserve entire physical vehicles/sessions for final validation.

For the fastest honest first experiment: choose one clear passenger vehicle whose true length can be measured, annotate separated sharp moving frames through the central region, and supply the camera/road calibration. First test whether the manual-reference estimator is accurate. Only then replace manual observations with learned detections and measure the loss in accuracy.

Existing scene structures and known vehicle dimensions can supplement calibration, but this file alone contains no supplied trustworthy meter reference. Do not infer metric scale from estimated mounting height, guardrail post spacing or assumed catalog car dimensions.

## Current status

The original recording is readable and suitable for scene-specific development. The existing local-file playback/export path has been exercised on real footage. The automatic diagnostic detector fails this scene. Real metric accuracy, object completeness, contact-localization reliability and production reject rate are still unknown. No original video was modified and no fabricated metric measurements were produced.
