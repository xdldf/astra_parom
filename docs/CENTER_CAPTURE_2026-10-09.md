# Centre-only capture, once per passage — 9 October 2026

This correction supersedes the fallback behavior in `95280b2`. That change still saved off-centre observations after a car disappeared, and searched backwards from an exit fragment for another centre capture. Frame/bbox-based database keys treated a second observation as another vehicle. A disabled measurement line also left a path for measurements anywhere in the image.

## Current behavior

- The station requires the complete detected box inside the corrected image support and within the configured centre-line tolerance. A legacy profile without a line uses the horizontal image midpoint. Explicitly configured lines are preserved.
- Off-centre and clipped submissions receive HTTP 422, including requests from an older browser using `review_fallback`. Estimate, outline and wheel models cannot override this gate.
- There are no automatic queue inserts on disappearance, stream stop or video end. If there is no usable centre frame, that passage is not automatically saved.
- A skipped crossing can still be recovered between two observations of the same track, at most two seconds apart. Only an actual complete, centred frame is accepted. An already observed centre endpoint is also eligible. This is separate from the removed broad search following a lost exit fragment.
- IP and recorded-video inference use the same passage tracker. It assigns the full frame together, keeps identity across slow inference and brief missing detections, and retains the counted identity when an exiting box shrinks. Capture requires at least two observations and rejects a box substantially smaller than that track's largest observation. Following cars have independent IDs; there is no shared time cooldown.
- Each accepted passage has a stable server ID in the record and database uniqueness key. Multiple requests for that passage return the existing record. Recorded-video requests are serialized per passage so concurrent viewers do not rerun its measurement. Historical records remain available.
- Detection continues through the exit to maintain identity. A pending measurement can finish after the car moves away, using only its previously accepted centre frame and retained evidence. It does not save a new exit photo.

Depth/length calibration coefficients are unchanged. The existing strict/estimate policy for a valid centre capture is unchanged.

## Verification

All 407 Python tests and all 77 JavaScript tests pass. Coverage includes the IP inference loop with real persistence, slow and stopped cars, shrinking rear fragments, closely following vehicles, browser stop/end behavior, legacy profiles, concurrent requests with different frames, and outline/wheel rejection outside the gate.

Fresh CPU inference with the unchanged `config/st-wheel-recovery-calibration.json` and YOLO26 M replayed the original videos through the current tracker and capture path. Each segment includes six seconds after its centre capture. Each produced exactly one record, and saved full-frame JPEG bytes matched the original corrected frame used for measurement.

| Passage | Frames replayed | Captures | Capture frame | Centre offset | Estimated length |
|---|---|---:|---:|---:|---:|
| N1-003 | 4925–5171 | 1 | 5021 | +1.917 px | 4.578 m |
| N2-014 | 12550–12801 | 1 | 12651 | +7.123 px | 4.961 m |

These results verify capture selection and duplicate suppression for these passages, not independent physical length accuracy or live GPU performance. Local evidence is in `runs/center_once_20261009/`: `verify.py`, `report.json`, complete detection/track traces, saved record JSON and an isolated capture database. The earlier fallback audit in `runs/center_capture_20261009/` describes the superseded behavior.

## Applying the correction

Update the station checkout, restart the station process and reload the browser. Asset versions are bumped to `station.js?v=30` and `live-tracks.js?v=4`. The production station has not been restarted from this development workspace.
