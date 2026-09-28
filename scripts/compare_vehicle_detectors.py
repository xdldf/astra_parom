#!/usr/bin/env python3
"""Compare N/M/L on identical original frames, with auditable temporal evidence.

Case JSON: [{"name":"car", "video":"/path/video.mp4", "start_seconds":1,
            "end_seconds":3}]. Models are name=/path/weights.pt. CPU is offline only.
This measures detector consistency and latency, never unlabelled physical accuracy.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np

from vehicle_metrology.bbox_scale import measure_box
from vehicle_metrology.detection import predict_vehicle_boxes
from vehicle_metrology.temporal import measure_passage
from vehicle_metrology.video import sha256_file, write_json
from web_app.workbench import Profile, corrected, profile_scale
from web_app.temporal_capture import WINDOW_SECONDS


def summarize(rows, profile):
    scale=profile_scale(profile)
    results=[]
    for mode in rows[0]['modes']:
        for case in sorted({r['case'] for r in rows}):
            frames=[r for r in rows if r['case']==case]
            candidates=[(r,d) for r in frames for d in r['modes'][mode]['detections']
                        if d['length_m'] is not None and abs(d['line_offset_px'])<=profile.line_tolerance_px]
            near=[d['length_m'] for r in frames for d in r['modes'][mode]['detections']
                  if d['length_m'] is not None and abs(d['line_offset_px'])<=30]
            record=dict(case=case,mode=mode,frames=len(frames),near_line_samples=len(near),
                        raw_range_m=float(np.ptp(near)) if near else None,
                        median_inference_ms=1000*float(np.median([r['modes'][mode]['seconds'] for r in frames])))
            if candidates:
                # Select an actual detected box on the original narrow line gate.
                anchor,d=min(candidates,key=lambda pair:abs(pair[1]['line_offset_px']))
                evidence=[dict(frame=r['frame'],detections=r['modes'][mode]['detections']) for r in frames
                          if abs(r['seconds']-anchor['seconds'])<=WINDOW_SECONDS]
                record.update(anchor_frame=anchor['frame'],anchor_seconds=anchor['seconds'],
                              temporal=measure_passage(evidence,d['bbox'],profile.polygon,scale,profile.image_size,
                                line_x=profile.measurement_line_x,line_tolerance_px=profile.line_tolerance_px,
                                tolerance_m=profile.accuracy_tolerance_m))
            else:
                record.update(temporal=dict(status='temporal_review',length_m=None,reasons=['missing_line_evidence']))
            results.append(record)
    return dict(target_tolerance_m=profile.accuracy_tolerance_m,accuracy_validated=False,
                scope='Selected short clips. No independent true lengths. Raw ranges use ±30 px; temporal fits require actual evidence at the original line.',
                results=results)


def compare(cases,profile,models,output,device='cpu',confidence=.35):
    scale=profile_scale(profile)
    rows=[]
    for case in cases:
        path=Path(case['video'])
        cap=cv2.VideoCapture(str(path))
        try:
            fps=cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps<=0:
                raise ValueError(f'Invalid recording: {path}')
            first=round(case['start_seconds']*fps)
            last=round(case['end_seconds']*fps)
            if first<0 or not first<last<=cap.get(cv2.CAP_PROP_FRAME_COUNT) or last-first>300:
                raise ValueError('Each benchmark clip must contain 1–300 valid frames')
            cap.set(cv2.CAP_PROP_POS_FRAMES,first)
            for index in range(first,last):
                ok,raw=cap.read()
                if not ok or tuple(raw.shape[1::-1])!=profile.image_size:
                    raise ValueError('Missing frame or incorrect calibration resolution')
                frame=corrected(raw,profile.lens)
                row=dict(case=case['name'],video=path.name,frame=index,seconds=index/fps,modes={})
                for name,model in models.items():
                    for imgsz in (640,1280):
                        start=time.perf_counter()
                        detections=predict_vehicle_boxes(model,frame,confidence,device=device,imgsz=imgsz)
                        elapsed=time.perf_counter()-start
                        for detection in detections:
                            measured=measure_box(detection['bbox'],profile.polygon,scale,profile.image_size)
                            measured.pop('calibration_diagnostics',None)
                            detection.update(measured)
                            b=detection['bbox']
                            detection['line_offset_px']=b[0]+b[2]/2-profile.measurement_line_x
                        row['modes'][f'{name}{imgsz}']=dict(seconds=elapsed,detections=detections)
                rows.append(row)
        finally:
            cap.release()
        write_json(output/'model_comparison.json',rows)
        print(case['name']+' complete',flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cases',type=Path)
    parser.add_argument('--models',nargs='+',help='e.g. n=models/workbench/yolo26n.pt m=/path/yolo26m.pt')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--confidence',type=float,default=.35)
    parser.add_argument('--summarize',type=Path,help='Summarize existing detections without rerunning inference')
    args=parser.parse_args()
    profile=Profile.model_validate_json(args.profile.read_text())
    if profile.measurement_line_x is None:
        parser.error('Place the measurement line in the profile first')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.summarize:
        rows=json.loads(args.summarize.read_text())
    else:
        if not args.cases or not args.models:
            parser.error('--cases and --models are required for inference')
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        paths={name:Path(path) for name,path in (m.split('=',1) for m in args.models)}
        if any(not path.is_file() for path in paths.values()):
            parser.error('Use local weights; this audit never downloads models')
        cases=json.loads(args.cases.read_text())
        write_json(args.output_dir/'manifest.json',dict(profile=profile.model_dump(),device=args.device,
                   confidence=args.confidence,models={name:dict(path=str(path.resolve()),sha256=sha256_file(path)) for name,path in paths.items()},
                   cases=[dict(case,sha256=sha256_file(Path(case['video']))) for case in cases]))
        rows=compare(cases,profile,{name:YOLO(str(path)) for name,path in paths.items()},args.output_dir,args.device,args.confidence)
    write_json(args.output_dir/'comparison_summary.json',summarize(rows,profile))


if __name__=='__main__':
    main()
