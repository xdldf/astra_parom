# Capture recovery and length review

The station now retains detected passages that miss the measurement-line centre tolerance. IP cameras and recorded-video playback keep the closest real on-road observation while waiting for the normal line capture. If the car disappears for 0.8 seconds, or the best observation has not improved for 2 seconds, that observation is saved for review. A whole-car frame takes priority over a clipped frame. Pausing or reaching the end of a video also flushes retained observations. Completed video sessions retain their final detection results for browser polling.

Tracking can associate a non-overlapping box after a detection delay of up to 2 seconds when its size, vertical position and horizontal displacement are plausible. This weaker association must be unique in both directions; ambiguous nearby cars are not joined by this rule. Existing overlap tracking and searches of actual frames around a line crossing remain in use. When a crossing search finds no centred box, it saves the closer observed endpoint, without interpolating a box or an image. Expired pending IP measurements also preserve their actual anchor frame.

In estimate mode, off-line lengths are marked approximate with `missed_measurement_line`. Strict mode saves off-line captures without a length. Clipped detections are preserved without a length and cannot become scale references. This improves capture coverage; it does not guarantee detections of vehicles that the model never sees, or validate measurement accuracy.

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
