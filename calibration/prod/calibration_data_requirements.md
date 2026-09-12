--- SCHEMA 1: Intrinsic (required before survey) ---
{
  "schema_version": 1,
  "image_size": [2592, 1944],
  "model": "brown" or "fisheye",
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "D": [k1, k2, p1, p2, (k3)]  // brown: 4 or 5 coeffs; fisheye: 4
}
Notes: model is user-selected explicitly, not inferred from training RMS. Lock recording settings to match the video mode.

--- SCHEMA 2: Survey (required; controls + independent checks) ---
{
  "schema_version": 1,
  "image_size": [2592, 1944],
  "units": "m",
  "road_polygon": [[xmin,ymin], [xmax,ymin], [xmax,ymax], [xmin,ymax]],
  "control_points": [{"id":"c01","xyz_m":[X,Y,0.0],"uv":[pixel_x,pixel_y]}],
  "check_points": [{"id":"chk01","xyz_m":[X,Y,0.0],"uv":[pixel_x,pixel_y]}]
}
Required: control_points >= 6; check_points >= 3; independent IDs; Z=0 (this CLI); no overlap with controls.
The user must supply real surveyed measurements; do not invent xyz_m.

--- SCHEMA 3: Ground truth (optional; only after calibration exists) ---
{
  "track_id", "vehicle_id", "length_m"
}
length_m must be independently measured; this file is never used in calibration.

--- Production footage assessment (facts from user file only) ---
Source: C:/Users/alexe/Downloads/ST_2026-09-04_19-30-04.mp4
Verified: exists (590,008,873 bytes), OpenCV readable.
FFprobe: 2592x1944, HEVC Main, YUV420p, 25/25 fps, 45000 frames, 1800 s duration.
Diagnostic detector (MOG2 + greedy nearest-center association) on real footage:
  - 20 s excerpt (500 frames): 6,842 diagnostic fragments, median 179 detections/frame, max 1,283.
  - These are fragments, NOT vehicle counts; the detector fails as a production recognition pipeline.
Actual pipeline result (diagnostic overlay, no calibration, no annotations):
  - 500 frames decoded; 0 metric results; output saved at runs/production_inspection/pipeline_110_130s/
Visual inspection (contact_sheet + frame samples): central region visible; fences/guardrails occlude lower-left; lighting degrades later frames. Real metric accuracy remains unverified.

What is MISSING to make this physically meaningful:
1. Intrinsic model decision (brown vs fisheye) from a real calibration rig, NOT from synthetic examples alone.
2. Surveyed control/check points measured physically on the actual scene at this resolution.
3. Independent real vehicle lengths (measured, not catalog estimates).
4. Manual annotations identifying the SAME physical bumper/front and rear endpoints and same-side wheel-road contacts across selected sharp frames.
Only when 1-4 exist does the measurement become a physical claim rather than a geometric demonstration.
