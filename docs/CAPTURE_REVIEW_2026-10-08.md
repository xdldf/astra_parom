# Capture recovery and length review

Updated 9 October: the station retains detected passages that miss the measurement-line centre tolerance. IP cameras and recorded-video playback keep the closest real on-road observation while waiting for the normal line capture. Review fallback waits until the car has been absent for two seconds; an unchanged best observation never expires a car that is still visible. Pausing or reaching the end of a video also flushes retained observations. Completed video sessions retain their final detection results for browser polling.

Tracking can associate a non-overlapping box after a detection delay of up to 2 seconds when its size, vertical position and horizontal displacement are plausible. This weaker association must be unique in both directions; ambiguous nearby cars are not joined by this rule. Existing overlap tracking and searches of actual frames around a line crossing remain in use. When a crossing search finds no centred box, it saves the closer observed endpoint, without interpolating a box or an image. Expired pending IP measurements also preserve their actual anchor frame.

Before saving an off-line fallback, recordings search consecutive real frames up to four seconds either side of the anchor, capped at 241 frames. IP cameras retain a separate consecutive-frame window bounded by 241 packets and 48 MiB. Recovery follows the original vehicle and stops on a missing or ambiguous association. When recovered, the measurement, bbox, photo and front-camera pairing all use the centred frame. Lens-corrected/rotated image borders are checked as well as the rectangular canvas.

Off-line captures now remain **without a length in both estimate and strict modes**, with `missed_measurement_line`. Outline and wheel models cannot override this rejection. Clipped detections remain without a length and cannot become scale references. A recovered centred estimate is still approximate; this change does not establish physical accuracy. See [centre-capture verification](CENTER_CAPTURE_2026-10-09.md).

## Review workflow

Open **Проверка длины** in the station header, or `/evaluation`:

1. Inspect the entire corrected frame. The optional yellow box identifies the target; it is a UI overlay and is not burned into the saved image. Click the image to open it at full resolution.
2. Read the original measured length, then enter the independently established real length in metres. The input starts empty and never copies the camera estimate.
3. **Сохранить и далее** saves the real length and opens the next car. This does not change the original estimate, cashier confirmation, price or payment status.
4. **Пропустить** persistently moves the car to **Пропущенные**. Reopen it there later. **Проверенные** allows corrections to saved labels. Concurrent updates reject an outdated version and preserve the value being typed.

Every label/skip updates an atomic calibration JSON snapshot in `web_app/data/station/calibrations/<geometry-hash>.json`. **Скачать калибровку JSON** downloads a fresh profile for the currently displayed car's camera geometry. All labeled samples for that geometry appear under `evaluation_samples`, including the original estimate, real length, bbox, frame, full image filenames and reviewer provenance. Compatible, complete frames also become human-reviewed `references`; this explicit length review is independent of cashier approval. The active reference fit is limited to 100 entries, preserving manual/legacy references first and then the most recently reviewed cars. All labels remain in `evaluation_samples` regardless of this fit limit.

The existing calibration import/export/apply flow includes these labels and references. Load the JSON and use **Сохранить и применить** to update a running measurement profile. Saving labels does not restart cameras. Different image dimensions, road polygons or lens settings have separate calibration data. Metric rulers keep priority, and survey calibration never gains a vehicle-derived scale. Skipping a previously labeled car withdraws its reference and removes its label from subsequent exports.

New captures store `<id>-frame.jpg` (full corrected measurement image), alongside the existing small vehicle crop. Only the corrected full frame is saved for evaluation; uncorrected camera images are not stored. Measurement bboxes use corrected full-frame coordinates. Back up the entire `web_app/data` directory to retain the images, database and JSON together. Historical crop-only records cannot reconstruct the missing full image and are not included in the new review queue.

## Verification

Regression tests cover non-overlapping tracking, ambiguous neighbours, off-line IP capture after rolling-buffer eviction, video crossing without a centred frame, end-of-video result retention, full-image dimensions and pixels outside the crop, idempotent capture, real-length JSON persistence, skip/relabel, geometry compatibility, clipped frames, invalid input, stale writes, and browser form behavior. GPU inference and accuracy on live site traffic still need field evaluation.
