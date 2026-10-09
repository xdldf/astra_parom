# Centre capture correction — 9 October 2026

The review fallback bypassed the configured measurement line in estimate mode, then allowed outline/wheel regression to assign a length. A second timeout could finalize a still-visible vehicle when its best off-centre observation had not improved for two seconds. The visibility guard only checked the rectangular canvas, missing invalid image borders created by lens correction and rotation.

Captures now require a complete bounding box inside the corrected image support and a centre within the existing configured line tolerance before assigning a length. Neither estimate mode nor outline/wheel refinement overrides those conditions. Unrecoverable observations are retained for review without a length. Existing historical records are not rewritten.

Recorded-video fallback searches every actual frame within four seconds of the observed anchor, at most 241 frames. It follows overlapping detections forwards and backwards and stops at missing or ambiguous identity. A recovered frame replaces the source frame, bbox, side photo and synchronized front frame together. Live IP capture retains consecutive originals separately, with a 241-packet/48 MiB bound, so receiver eviction does not erase the centre before review. The two-second fallback timer measures vehicle absence, never the age of the best picture while a vehicle remains visible.

## Verification

Fresh CPU inference replayed two edge-fragment anchors through `station.capture`, with the unchanged `config/st-wheel-recovery-calibration.json`, YOLO26 M, outline and wheel models. The production database was not used.

| Passage | Edge anchor | Recovered centre | Centre offset | Estimated length |
|---|---:|---:|---:|---:|
| N1-003 | 4950 | 5021 | +1.917 px | 4.578 m |
| N2-014 | 12575 | 12651 | +7.123 px | 4.961 m |

Both pass the existing ±10 px line tolerance. Saved full-frame JPEG bytes match the corrected source frame used by the capture. These are estimates, not independent physical accuracy measurements. CPU replay took about 29 and 25 seconds respectively; live GPU latency and recall have not been evaluated here.

Local evidence: `runs/center_capture_20261009/report.json`, per-passage records, before/after full frames and an isolated capture database. Reproduce with the updated audit tool:

```bash
.venv/bin/python scripts/audit_additional_captures.py \
  --selection docs/new-videos-selection-2026-10-09.json \
  --profile config/st-wheel-recovery-calibration.json \
  --weights runs/accuracy_10cm/models/yolo26m.pt \
  --video-dir /home/user/Downloads --ids N1-003 N2-014 \
  --output-dir runs/center_capture_reproduced --device cpu
```

Regression checks cover recovery from either side, missing/ambiguous identity, unchanged visible cars, lens-created borders, IP recovery after receiver eviction, retention bounds, off-centre outline/wheel rejection, no-centre fallback without length, and audit verification against the recovered source. The station process must load the updated code and browsers must reload the updated tracking script. No station server was running in the development workspace during verification.
