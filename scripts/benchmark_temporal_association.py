#!/usr/bin/env python3
"""Replay temporal association from original videos with family-heldout length models.

All catalogue passages remain in the denominator. Only corrected-pixel-identical,
verified detector caches are reused. Anchor masks/wheels are frozen from the
previous benchmark to isolate association changes from new segmentation noise.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_passage_association import DETECTOR_SHA256
from vehicle_metrology.catalogue_calibration import catalogue_family, comparison_metrics
from vehicle_metrology.catalogue_labels import corrected_rows
from vehicle_metrology.detection import predict_vehicle_boxes
from vehicle_metrology.outline import WEIGHTS_SHA256
from vehicle_metrology.wheels import MODEL_FILES, MODEL_REVISION, file_digest
from web_app import workbench as wb, temporal_capture as tc, station as st
from web_app.outline_measurement import apply_outline
from web_app.wheel_measurement import apply_wheels


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('comparison','primary-results','recovery-results','profile','masks','detections',
                   'catalogue-corrections','association-audit','weights','output-dir'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda:0'], default='cuda:0')
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    rows = corrected_rows([r for r in json.loads(args.comparison.read_text())['rows'] if r.get('comparison_eligible')],
                          args.comparison,args.catalogue_corrections)
    profile = wb.Profile.model_validate_json(args.profile.read_text())
    primary, recovery = [json.loads(p.read_text()) for p in (args.primary_results,args.recovery_results)]
    masks = json.loads((args.masks/'mask-manifest.json').read_text())
    wheels = json.loads(args.detections.read_text())
    audit = json.loads(args.association_audit.read_text())
    shared = dict(comparison=file_digest(args.comparison),masks=file_digest(args.masks/'mask-manifest.json'),
                  detections=file_digest(args.detections),corrections=file_digest(args.catalogue_corrections))
    if (any(p['inputs_sha256'].get(k) != v for p in (primary,recovery) for k,v in shared.items())
            or recovery['inputs_sha256']['primary_results'] != file_digest(args.primary_results)
            or file_digest(args.weights) != DETECTOR_SHA256
            or audit['detector_sha256'] != DETECTOR_SHA256 or audit['imgsz'] != 640 or audit['confidence'] != .3
            or audit['detector_implementation_sha256'] != file_digest(ROOT/'vehicle_metrology/detection.py')
            or masks['model_sha256'] != WEIGHTS_SHA256 or masks['imgsz'] != 1280 or not masks['retina_masks']
            or wheels['model_sha256'] != MODEL_FILES['model.safetensors'] or wheels['revision'] != MODEL_REVISION
            or profile.detector_model != 'yolo26m' or profile.detector_imgsz != 640):
        raise ValueError('Input provenance does not match the frozen detector and length benchmarks')
    pf = {f['heldout_group']:f for f in primary['folds']}
    rf = {f['heldout_group']:f for f in recovery['folds']}
    baseline = {r['id']:r['runtime_m'] for r in recovery['rows']}
    cache = {}
    for entry in audit['rows'].values():
        path = args.association_audit.parent/entry['detections_file']
        if file_digest(path) != entry['detections_sha256']:
            raise ValueError('Changed dense detector cache')
        data = json.loads(path.read_text())
        if data['signature'] != {k:v for k,v in audit.items() if k != 'rows'}:
            raise ValueError('Mismatched dense detector provenance')
        for value in data['frames'].values():
            cache[value['pixels_sha256']] = value['detections']
    signatures = dict(**shared,profile=file_digest(args.profile),primary_results=file_digest(args.primary_results),
        recovery_results=file_digest(args.recovery_results),association_audit=file_digest(args.association_audit),
        temporal=file_digest(ROOT/'vehicle_metrology/temporal.py'),capture=file_digest(ROOT/'web_app/temporal_capture.py'),
        outline_runtime=file_digest(ROOT/'web_app/outline_measurement.py'),wheel_runtime=file_digest(ROOT/'web_app/wheel_measurement.py'),
        script=file_digest(Path(__file__)),detector=DETECTOR_SHA256,device=args.device)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path = args.output_dir/'results.json'
    result = dict(inputs_sha256=signatures,accuracy_validated=False,dataset_role='development_catalogue_check',rows={})
    if path.exists():
        result = json.loads(path.read_text())
        if result['inputs_sha256'] != signatures:
            raise ValueError('Replay inputs changed; use a new output directory')
    model = YOLO(str(args.weights))
    new_cache = dict(inputs_sha256=signatures,frames={})
    cache_path = args.output_dir/'new-detections.json'
    if cache_path.exists():
        new_cache = json.loads(cache_path.read_text())
        if new_cache['inputs_sha256'] != signatures:
            raise ValueError('New detector cache provenance changed')
        cache.update(new_cache['frames'])

    def detect(image, confidence=.3, **settings):
        if confidence != .3 or settings != dict(detector_model='yolo26m',imgsz=640):
            raise ValueError('Unexpected temporal detector settings')
        key = hashlib.sha256(image.tobytes()).hexdigest()
        if key not in cache:
            cache[key] = predict_vehicle_boxes(model,image,confidence,device=args.device,imgsz=640)
            new_cache['frames'][key] = cache[key]
        return copy.deepcopy(cache[key])

    with patch.object(wb,'detect_vehicles',detect):
        for row in rows:
            ident = row['id']
            if ident in result['rows']:
                continue
            started = time.monotonic()
            group = catalogue_family(row['source_ids'])
            heldout = wb.Profile.model_validate({**profile.model_dump(),
                'outline_calibration':pf[group]['outline_fallback_calibration'],
                'wheel_calibration':pf[group]['calibration'],'wheel_recovery_calibration':rf[group]['calibration']})
            capture = cv2.VideoCapture(row['video'])
            wb.media[ident] = dict(kind='video',path=Path(row['video']),fps=capture.get(cv2.CAP_PROP_FPS),
                                   frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()
            raw = wb.read_frame(ident,row['frame'])
            rendered = wb.render_raw(raw,wb.FrameRequest(profile=heldout,frame=row['frame'],boxes=[row['bbox']]),
                                     include_image=False,include_frame=True)
            payload = st.Capture(media_id=ident,profile=heldout,frame=row['frame'],bbox=row['bbox'],
                                 label=row['detector_label'],source='yolo26m',temporal=True,review_fallback=True)
            tc.refine_video(payload,rendered)
            tc.prepare_review_measurement(heldout,rendered)
            mask_path = args.masks/(ident+'.npz')
            entry, detected = masks['masks'][ident], wheels['rows'][ident]
            photo_hash = file_digest(args.comparison.parent/row['image'])
            if (entry['sha256'] != file_digest(mask_path) or entry['corrected_image_sha256'] != photo_hash
                    or detected['corrected_image_sha256'] != photo_hash or detected['vehicle_bbox'] != row['bbox']
                    or detected['frame'] != row['frame'] or entry['status'] != 'ok' or detected['status'] != 'ok'):
                raise ValueError('Changed anchor mask or wheel evidence: '+ident)
            mask = np.load(mask_path,allow_pickle=False)['mask']
            if mask.shape[::-1] != heldout.image_size:
                raise ValueError('Incorrect corrected-mask dimensions')
            details = {k:detected[k] for k in ('detections','roi','image_space')}
            with patch('web_app.outline_measurement.infer_outline',return_value=(mask,{})), \
                 patch('web_app.wheel_measurement.infer_wheels',return_value=details):
                measured = apply_outline(heldout,rendered['detections'][0],rendered['frame_image'],label=row['detector_label'])
                measured = apply_wheels(heldout,measured,rendered['frame_image'],label=row['detector_label'])
            result['rows'][ident] = dict(id=ident,family=group,video=Path(row['video']).name,frame=row['frame'],
                catalogue_interval_m=row['catalogue_length_range_m'],previous_runtime_m=baseline[ident],
                runtime_m=measured['length_m'],measurement=measured,elapsed_seconds=time.monotonic()-started)
            save(cache_path,new_cache)
            save(path,result)
            print(ident,baseline[ident],'->',measured['length_m'],round(time.monotonic()-started,1),'seconds',flush=True)
    values = np.asarray([result['rows'][r['id']]['runtime_m'] if result['rows'][r['id']]['runtime_m'] is not None
                         else np.nan for r in rows])
    intervals = np.asarray([r['catalogue_length_range_m'] for r in rows])
    groups = np.asarray([catalogue_family(r['source_ids']) for r in rows])
    available = np.isfinite(values)
    result['summary'] = comparison_metrics(values[available],intervals[available],groups[available])
    result['summary'].update(total_passages=len(rows),review_without_length=int((~available).sum()))
    result['limitations'] = ['Catalogue identities remain provisional; this does not prove physical accuracy.',
        'All eligible passages remain in the denominator; 258 other audit candidates are not catalogue scored.',
        'Length coefficients exclude the entire tested family; no catalogue identity is an inference input.',
        'Temporal detections are recomputed or reused only for identical corrected pixels.',
        'Anchor segmentation/wheel caches isolate temporal association; separate real captures test full inference.',
        'This change was developed on these same videos; no independent final accuracy validation.']
    save(path,result)
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
