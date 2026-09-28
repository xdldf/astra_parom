#!/usr/bin/env python3
"""Compare local YOLO, RT-DETR and RF-DETR weights on identical original frames.

--models accepts JSON objects with name, family, weights and optional resolution.
Accuracy is never inferred from unlabelled detector agreement or repeatability.
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np

from scripts.compare_vehicle_detectors import summarize
from vehicle_metrology.bbox_scale import measure_box
from vehicle_metrology.detection import box_iou
from vehicle_metrology.detector_backends import DetectorBackend
from vehicle_metrology.video import sha256_file,write_json
from web_app.workbench import Profile,corrected,profile_scale


def target_boxes(reference_rows,case):
    frames=[r for r in reference_rows if r['case']==case]
    if not frames:
        raise ValueError(f'Missing target reference for {case}')
    middle=frames[len(frames)//2]
    ds=middle['modes']['m640']['detections']
    eligible=[d for d in ds if d.get('depth') is not None]
    if not eligible:
        raise ValueError(f'No in-road seed for {case}')
    seed=min(eligible,key=lambda d:abs(d['line_offset_px']))['bbox']
    expected={}
    for row in frames:
        matches=sorted(((box_iou(seed,d['bbox']),d['bbox']) for d in row['modes']['m640']['detections']),reverse=True)
        if matches and matches[0][0]>=.3:
            expected[row['frame']]=matches[0][1]
    return expected


def coverage_summary(rows,reference_rows):
    results=[]
    for case in dict.fromkeys(r['case'] for r in rows):
        expected=target_boxes(reference_rows,case)
        frames=[r for r in rows if r['case']==case]
        for mode in rows[0]['modes']:
            scores=[];matched=[];ambiguous=0
            for row in frames:
                if row['frame'] not in expected:
                    continue
                ds=row['modes'][mode]['detections']
                pairs=sorted(((box_iou(expected[row['frame']],d['bbox']),d) for d in ds),
                             key=lambda pair:pair[0],reverse=True)
                if len(pairs)>1 and pairs[1][0]>.4:
                    ambiguous+=1
                if pairs and pairs[0][0]>=.5:
                    scores.append(pairs[0][0]);matched.append(pairs[0][1])
            results.append(dict(case=case,mode=mode,reference_frames=sum(r['frame'] in expected for r in frames),
                associated_frames=len(matched),ambiguous_frames=ambiguous,
                median_iou_to_reference=float(np.median(scores)) if scores else None,
                mean_confidence=float(np.mean([d['confidence'] for d in matched])) if matched else None))
    return dict(note='IoU>=0.5 association to earlier YOLO target boxes; NOT labelled recall or localisation accuracy.',results=results)


def target_measurement_rows(rows, reference_rows):
    """Keep each selected target separate from neighbouring road vehicles.

    Full detections remain in the raw artifact. This view is only for the
    target's length summary; missing associations stay empty, not replaced
    by whichever different car happens to be nearest the measurement line.
    """
    expected = {case: target_boxes(reference_rows, case)
                for case in dict.fromkeys(row['case'] for row in rows)}
    filtered = []
    for row in rows:
        target = expected[row['case']].get(row['frame'])
        modes = {}
        for name, result in row['modes'].items():
            detections = result['detections']
            match = (max(detections, key=lambda d: box_iou(target, d['bbox']))
                     if target is not None and detections else None)
            kept = [match] if match is not None and box_iou(target, match['bbox']) >= .5 else []
            modes[name] = dict(result, detections=kept)
        filtered.append(dict(row, modes=modes))
    return filtered


def benchmark(cases,profile,specs,reference_rows,output,device,confidence):
    import torch
    models={s['name']:DetectorBackend(s['family'],s['weights'],device=device,resolution=s.get('resolution')) for s in specs}
    scale=profile_scale(profile);rows=[]
    for case in cases:
        expected=target_boxes(reference_rows,case['name'])
        cap=cv2.VideoCapture(case['video'])
        try:
            fps=cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps<=0:
                raise ValueError('Invalid video rate')
            first=round(case['start_seconds']*fps);last=round(case['end_seconds']*fps)
            if not 0<=first<last<=cap.get(cv2.CAP_PROP_FRAME_COUNT) or last-first>300:
                raise ValueError('Clips must contain 1–300 frames')
            cap.set(cv2.CAP_PROP_POS_FRAMES,first)
            for index in range(first,last):
                ok,raw=cap.read()
                if not ok or tuple(raw.shape[1::-1])!=profile.image_size:
                    raise ValueError('Frame decode or calibration dimensions failed')
                frame=corrected(raw,profile.lens)
                row=dict(case=case['name'],video=Path(case['video']).name,frame=index,seconds=index/fps,modes={})
                for name,model in models.items():
                    if not rows and index==first:
                        model.predict(frame,confidence)  # Exclude one explicit warm-up per model.
                    if str(device)!='cpu':
                        torch.cuda.synchronize()
                    start=time.perf_counter();detections=model.predict(frame,confidence)
                    if str(device)!='cpu':
                        torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    for d in detections:
                        measured=measure_box(d['bbox'],profile.polygon,scale,profile.image_size)
                        measured.pop('calibration_diagnostics',None);d.update(measured)
                        d['line_offset_px']=d['bbox'][0]+d['bbox'][2]/2-profile.measurement_line_x
                    row['modes'][name]=dict(seconds=elapsed,resolution=model.resolution,detections=detections)
                rows.append(row)
                if index==(first+last)//2 and index in expected:
                    x,y,w,h=expected[index];left=max(0,int(x)-100);right=min(frame.shape[1],int(x+w)+100)
                    top=max(0,int(y)-80);bottom=min(frame.shape[0],int(y+h)+80)
                    for name in models:
                        preview=frame.copy()
                        for d in row['modes'][name]['detections']:
                            a,b,c,e=map(round,d['bbox']);cv2.rectangle(preview,(a,b),(a+c,b+e),(40,230,255),3)
                        crop=preview[top:bottom,left:right]
                        crop=cv2.resize(crop,(720,400));cv2.rectangle(crop,(0,0),(720,34),(20,25,32),-1)
                        cv2.putText(crop,f"{name} | {case['name']} | frame {index}",(10,23),0,.56,(245,245,245),1)
                        cv2.imwrite(str(output/f"{case['name']}_{name}.jpg"),crop)
            print(case['name']+' complete',flush=True)
            write_json(output/'model_comparison.json',rows)
        finally:
            cap.release()
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--reference-detections',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--confidence',type=float,default=.3)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if not 0<args.confidence<1:
        parser.error('Confidence must be between zero and one')
    profile=Profile.model_validate_json(args.profile.read_text())
    if profile.measurement_line_x is None:
        parser.error('A measurement line is required')
    cases=json.loads(args.cases.read_text());specs=json.loads(args.models.read_text())
    if not cases or not specs or len({s['name'] for s in specs})!=len(specs):
        parser.error('Nonempty cases and uniquely named model specifications required')
    for s in specs:
        if not s['name'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in s['name']):
            parser.error('Model names must be simple filename-safe identifiers')
    if any(not c['name'] or any(v not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for v in c['name']) for c in cases):
        parser.error('Case names must be simple filename-safe identifiers')
    reference_rows=json.loads(args.reference_detections.read_text())
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(1)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    versions={}
    for package in ('torch','torchvision','ultralytics','rfdetr','transformers','supervision','numpy','opencv-python'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    hashes={path:sha256_file(Path(path)) for path in {c['video'] for c in cases}}
    write_json(args.output_dir/'manifest.json',dict(profile=profile.model_dump(),profile_sha256=sha256_file(args.profile),
        reference_detections_sha256=sha256_file(args.reference_detections),cases=cases,video_hashes=hashes,
        models=[dict(s,sha256=sha256_file(Path(s['weights']))) for s in specs],device=args.device,
        confidence=args.confidence,threads=4,versions=versions,warmup_frames_per_model=1,accuracy_validated=False))
    rows=benchmark(cases,profile,specs,reference_rows,args.output_dir,args.device,args.confidence)
    write_json(args.output_dir/'comparison_summary.json',
               summarize(target_measurement_rows(rows, reference_rows),profile))
    write_json(args.output_dir/'coverage_summary.json',coverage_summary(rows,reference_rows))


if __name__=='__main__':
    main()
