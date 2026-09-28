# Catalogue dimensions versus saved vehicle length estimates

Additional recordings from September 7 have a separate [20-passage evaluation and catalogue comparison](ADDITIONAL_VIDEOS_2026-09-28.md). The results below refer only to the earlier September 4 recordings.

**The tested system still misses the ±10 cm target on three of these four provisional catalogue matches.** This comparison uses actual saved replay estimates, not detector box jitter. There are no independently measured physical lengths for these vehicles.

The earlier 4.65 cm RT-DETR truck result is a temporal fit residual, not an error against its real length. A stable estimate can still have a systematic error.

## Conditional comparison

Vehicle identities below are visual inferences. The dimensions are verified manufacturer specifications for the stated candidate; the exact filmed year, trim, market and modifications have not been independently confirmed. Negative differences mean an underestimate.

| Likely vehicle | Catalogue | YOLO26 M estimate | YOLO difference | RT-DETR X estimate | RT difference |
|---|---:|---:|---:|---:|---:|
| [Black SUV — likely Toyota Land Cruiser Prado J150, 2013 facelift](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-LAND_CRUISER_PRADO/201309/10084725/) | 4.760 m | 4.634 m | -12.6 cm | 4.625 m | -13.5 cm |
| [Silver hatchback — likely Toyota Vitz XP130, early third generation](https://www.toyota-global.com/company/history_of_toyota/75years/vehicle_lineage/car/id60017358/) | 3.885 m | 3.748 m | -13.7 cm | 3.759 m | -12.6 cm |
| [White wagon — likely Toyota Corolla Fielder E160, early third generation](https://www.toyota-global.com/company/history_of_toyota/75years/vehicle_lineage/car/id60017618/) | 4.360 m | 4.326 m | -3.4 cm | 4.364 m | +0.4 cm |
| [White compact SUV — likely Toyota Yaris Cross XP210](https://toyota.jp/ucar/catalog/brand-TOYOTA/car-YARIS_CROSS/202008/10130870/) | 4.180 m | 4.043 m | -13.7 cm | 4.044 m | -13.6 cm |

RT-DETR X: mean absolute catalogue difference **10.04 cm**, maximum **13.64 cm**, **1 of 4** within 10 cm. YOLO26 M: **10.88 cm**, **13.75 cm**, **1 of 4** respectively. These are conditional comparisons on four selected vehicles, not fleet accuracy estimates or physical validation.

The white wagon is a likely Corolla Fielder, not the Wish implied by an old directory name. If it is the later 4.400 m facelift instead, the RT-DETR X difference is −3.56 cm; the exact grade/year still needs confirmation. The compact SUV appears to be a Yaris Cross, not the earlier provisional Raize. The previously cited Wish and white Vitz from the +60 s frame are different cars and are not used here.

The truck has an estimated 6.606 m length, but no trustworthy catalogue counterpart until its chassis/body/load configuration is identified. The van has no numeric capture. Neither is included in the four-car statistics.

## Traceability and visual review

All timestamps are offsets from the beginning of the supplied file, not clock time. The linked crops show nearby frames from the same short passage; exact numeric estimates come from the listed capture frames.

### Black SUV — likely Toyota Land Cruiser Prado J150, 2013 facelift

`ST_2026-09-04_18-00-04.mp4`, offset 504.40 s, RT-DETR X capture frame 12610.

Five-door body, rear quarter window, vertical rear lamps and front lamp shape. Year/market/trim and accessories unconfirmed.

![suv visual evidence](../runs/detr_20260928/benchmark/suv_rtdetr_x_640.jpg)

### Silver hatchback — likely Toyota Vitz XP130, early third generation

`ST_2026-09-04_18-00-04.mp4`, offset 524.96 s, RT-DETR X capture frame 13124.

Side glazing, door outline, rear lamps and front lamps match the early third generation. Exact grade unconfirmed.

![hatchback visual evidence](../runs/detr_20260928/benchmark/hatchback_rtdetr_x_640.jpg)

### White wagon — likely Toyota Corolla Fielder E160, early third generation

`ST_2026-09-04_18-30-04.mp4`, offset 1264.84 s, RT-DETR X capture frame 31621.

Headlamps, rear side window and rear lamp outline support Fielder E160. Historical case key mpv and dense_wish directory are not confirmed model labels. Exact facelift/trim unconfirmed.

![mpv visual evidence](../runs/detr_20260928/benchmark/mpv_rtdetr_x_640.jpg)

### White compact SUV — likely Toyota Yaris Cross XP210

`ST_2026-09-04_18-00-04.mp4`, offset 516.04 s, RT-DETR X capture frame 12901.

Rear quarter window, black rear-lamp surround, wheel arches and front lamps support Yaris Cross. This corrects the earlier provisional Raize identification. Exact trim unconfirmed.

![small_suv visual evidence](../runs/detr_20260928/benchmark/small_suv_rtdetr_x_640.jpg)

## Evidence and interpretation

- Saved measurements: `runs/detr_20260928/production_capture/summary.json` and `runs/accuracy_10cm/production_capture/summary.json`. Their hashes, per-case source frames, record IDs, complete precision and calculation results are in `runs/catalogue_comparison_20260928/comparison.json`.
- Both replays use the same existing lens/road/depth calibration. RT-DETR X changes the passenger estimates by less than 4 cm from YOLO26 M; it does not remove the approximately 13 cm discrepancies on three cars.
- The shared underestimation suggests remaining scale, perspective or vehicle-envelope bias. This comparison does not isolate which contributes how much.
- These catalogue values were not added to calibration or physical ground truth. Fitting these four values and then scoring on them would not establish accuracy on other cars.
- No application settings or historical measurement records changed during this comparison.
