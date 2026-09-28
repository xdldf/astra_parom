#!/usr/bin/env python3
"""Replay a corrected-pixel landmark track against the road/barrier plane fit.

Produces diagnostic candidates, point overlays and pixel sensitivity. Never
substitutes catalogue lengths or promotes failed geometry to accepted lengths.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from vehicle_metrology.bbox_scale import measure_box
from vehicle_metrology.projective_track import KEYS, RoadProjection, measure_projective_track
from vehicle_metrology.road_survey import corrected_projection
from vehicle_metrology.video import sha256_file, write_json
from web_app.workbench import Profile, corrected, profile_scale


def load_projection(profile, fit):
    if not profile.survey_calibration or profile.lens.model_dump() != fit['fit']['lens']:
        raise ValueError('Profile and plane fit must use identical lens settings')
    G = corrected_projection(fit['fit'], fit['ground'], profile.image_size)
    if not np.allclose(G, profile.survey_calibration.projection, atol=1e-8):
        raise ValueError('Profile and plane fit projections do not match')
    raised = np.r_[fit['fit']['parameters'][:8], 1.].reshape(3, 3)
    R = corrected_projection(fit['fit'], {'normalized_world_to_ground_image': raised}, profile.image_size)
    error = max(profile.survey_calibration.ruler_check_error_m,
                profile.survey_calibration.ground_check_error_m,
                fit['fit']['training_max_error_m'], fit['ground']['max_post_heldout_error_m'],
                fit['heldout_ruler_check']['heldout_max_error_m'])
    return RoadProjection.from_parallel_planes(G, R, profile.survey_calibration.support_world,
                                               profile.image_size, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--landmarks', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mc-samples', type=int, default=200)
    parser.add_argument('--detector', type=Path, help='Optional local YOLO checkpoint for a box comparison')
    args = parser.parse_args()
    profile = Profile.model_validate_json(args.profile.read_text())
    fit = json.loads(args.fit.read_text())
    data = json.loads(args.landmarks.read_text())
    if data.get('schema_version') != 1 or data.get('coordinate_space') != 'corrected_full_resolution':
        raise ValueError('Expected corrected full-resolution landmark schema version 1')
    if data['lens'] != profile.lens.model_dump() or tuple(data['image_size']) != profile.image_size:
        raise ValueError('Landmarks use different lens settings or image dimensions')
    video = Path(data['video'])
    if not video.is_file():
        raise ValueError('An existing local recorded video is required')
    rows = sorted(data['observations'], key=lambda row: row['frame'])
    if not rows:
        raise ValueError('At least one observation is required for replay')
    sources = {p.resolve() for p in (args.profile, args.fit, args.landmarks, video)}
    destinations = [args.output_dir/name for name in ('results.json', 'manifest.json', 'contact_sheet.jpg')]
    destinations += [args.output_dir/f"frame_{o['frame']}.jpg" for o in rows]
    if any(p.resolve() in sources for p in destinations):
        raise ValueError('Output would overwrite a source')
    camera = load_projection(profile, fit)
    result = measure_projective_track(camera, rows, tolerance_m=profile.accuracy_tolerance_m,
                                      mc_samples=args.mc_samples, seed=20260928)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    if args.detector:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        model = YOLO(str(args.detector))
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Video could not be opened')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if size != profile.image_size:
        cap.release()
        raise ValueError('Video dimensions differ from calibration')
    thumbs, comparisons = [], []
    colors = [(255,220,0), (255,220,0), (60,120,255), (60,120,255)]
    try:
        for row in rows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame'])
            ok, raw = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {row['frame']}")
            im = corrected(raw, profile.lens)
            box_result = None
            if model:
                from vehicle_metrology.detection import predict_vehicle_boxes
                detections = predict_vehicle_boxes(model, im, .3, device='cpu', imgsz=profile.detector_imgsz)
                pts = np.asarray([row[k] for k in KEYS])
                scored = [(sum(x <= u <= x+w and y <= v <= y+h for u,v in pts), d)
                          for d in detections for x,y,w,h in [d['bbox']]]
                if scored and max(s[0] for s in scored) >= 2:
                    d = max(scored, key=lambda s: s[0])[1]
                    box_result = measure_box(d['bbox'], profile.polygon, profile_scale(profile), profile.image_size)
                    x,y,w,h = map(round,d['bbox'])
                    cv2.rectangle(im,(x,y),(x+w,y+h),(40,230,255),2)
                    section = box_result['road_cross_section']
                    if section:
                        cv2.line(im,tuple(np.rint(section['far']).astype(int)),
                                 tuple(np.rint(section['near']).astype(int)),(255,80,230),3)
                        cv2.circle(im,tuple(np.rint(box_result['bottom']).astype(int)),8,(255,80,230),-1)
            polygon = np.rint(profile.polygon).astype(np.int32)
            cv2.polylines(im,[polygon],True,(80,240,130),3)
            for j,(key,color) in enumerate(zip(KEYS,colors)):
                point = tuple(np.rint(row[key]).astype(int))
                cv2.circle(im,point,7,color,-1)
                cv2.putText(im,('rear tyre','front tyre','rear bumper','front bumper')[j],
                            (point[0]-30,point[1]+(28 if j<2 else -15)),0,.55,color,2)
            xy=np.asarray([row[k] for k in KEYS])
            left=max(0,int(xy[:,0].min()-130));right=min(size[0],int(xy[:,0].max()+180))
            top=max(0,int(xy[:,1].min()-220));bottom=min(size[1],int(xy[:,1].max()+100))
            crop=im[top:bottom,left:right]
            thumb=cv2.resize(crop,(840,420));cv2.rectangle(thumb,(0,0),(840,38),(20,26,32),-1)
            cv2.putText(thumb,f"Frame {row['frame']} | cyan: tyre contacts | red: bumper points",(12,25),0,.5,(245,245,245),1)
            thumbs.append(thumb)
            cv2.imwrite(str(args.output_dir/f"frame_{row['frame']}.jpg"),cv2.resize(im,(1296,972)))
            comparisons.append(dict(frame=row['frame'],bbox_measurement=box_result))
    finally:
        cap.release()
    if len(thumbs)%2:
        thumbs.append(np.full_like(thumbs[0],24))
    sheet=np.vstack([np.hstack(thumbs[i:i+2]) for i in range(0,len(thumbs),2)])
    cv2.imwrite(str(args.output_dir/'contact_sheet.jpg'),sheet)
    write_json(args.output_dir/'results.json',dict(track_id=data['track_id'],measurement=result,
        box_comparisons=comparisons,annotation_source=data.get('annotation_source'),
        annotation_notes=data.get('notes'),accuracy_validated=False))
    write_json(args.output_dir/'manifest.json',dict(video=str(video),video_sha256=sha256_file(video),
        profile_sha256=sha256_file(args.profile),fit_sha256=sha256_file(args.fit),
        landmarks_sha256=sha256_file(args.landmarks),detector_sha256=sha256_file(args.detector) if args.detector else None,
        mc_samples=args.mc_samples,seed=20260928,image_size=size,lens=profile.lens.model_dump(),
        projection=camera.P.tolist(),observations=rows))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
