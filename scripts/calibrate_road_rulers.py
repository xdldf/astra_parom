#!/usr/bin/env python3
"""Fit the user's metre marks/widths, hold out controls, and export a reviewable profile."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from vehicle_metrology.road_survey import fit_rulers,fit_ground_transfer,corrected_projection
from vehicle_metrology.video import sha256_file,write_json
from web_app.workbench import Profile,Lens,corrected


def calibrate(survey, raw, output, tolerance=.1):
    if raw is None or tuple(raw.shape[1::-1])!=tuple(survey['image_size']):
        raise ValueError('Use the original full-resolution raw camera frame')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    n=len(survey['rulers'][0]['points_px'])
    heldout=[*range(2,n,3),*[n+i for i in range(2,len(survey['rulers'][1]['points_px']),3)]]
    # Start from training points alone: even the initialization excludes checks.
    check=fit_rulers(survey,heldout=heldout)
    fit=fit_rulers(survey)
    if not fit['optimizer_success'] or not fit['radial_monotonic']:
        raise ValueError('Lens fit is not physically usable; inspect the controls')
    ground=fit_ground_transfer(survey,fit)
    ruler_error=max(check['heldout_max_error_m'],fit['training_max_error_m'])
    projection=corrected_projection(fit,ground,survey['image_size'])
    world=np.array(fit['world_points'])
    support=cv2.convexHull(world.astype(np.float32)).reshape(-1,2)
    polygon=cv2.perspectiveTransform(support[None].astype(float),projection)[0]
    # Never silently clip controls: a cropped field needs an explicit support change.
    if np.any(polygon<0) or np.any(polygon>=survey['image_size']):
        raise ValueError('Corrected support leaves the image; adjust output zoom and refit')
    profile=Profile(image_size=survey['image_size'],lens=fit['lens'],polygon=polygon.tolist(),
        references=[],metric_rulers=[],detector_model='yolo26m',detector_imgsz=640,
        measurement_line_x=survey['image_size'][0]/2,line_tolerance_px=10,
        accuracy_tolerance_m=tolerance,survey_calibration=dict(projection=projection.tolist(),
            support_world=support.tolist(),lens=fit['lens'],
            ruler_check_error_m=ruler_error,
            ground_check_error_m=ground['max_post_heldout_error_m']))
    write_json(output/'profile.json',profile.model_dump())
    write_json(output/'fit.json',dict(fit=fit,heldout_ruler_check=check,ground=ground,
        target_tolerance_m=tolerance,accuracy_validated=False,
        numerical_lengths_enabled=max(ruler_error,ground['max_post_heldout_error_m'])<=tolerance))
    corrected_frame=corrected(raw,Lens(**fit['lens']))
    cv2.imwrite(str(output/'corrected_full.jpg'),corrected_frame,[cv2.IMWRITE_JPEG_QUALITY,96])
    preview=cv2.resize(corrected_frame,(1296,972))
    cv2.imwrite(str(output/'corrected_preview.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,95])
    overlay=corrected_frame.copy()
    for X in range(int(np.ceil(support[:,0].min())),int(np.floor(support[:,0].max()))+1):
        pts=cv2.perspectiveTransform(np.array([[[X,0.],[X,6.]]]),projection)[0]
        cv2.line(overlay,tuple(np.rint(pts[0]).astype(int)),tuple(np.rint(pts[1]).astype(int)),(255,220,70),2)
    cv2.polylines(overlay,[np.rint(polygon).astype(np.int32)],True,(0,255,255),3)
    cv2.imwrite(str(output/'ground_grid_preview.jpg'),cv2.resize(overlay,(1296,972)))
    return profile,fit,check,ground


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--survey',type=Path,required=True)
    parser.add_argument('--raw-frame',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--tolerance-m',type=float,default=.1)
    args=parser.parse_args()
    survey=json.loads(args.survey.read_text())
    profile,fit,check,ground=calibrate(survey,cv2.imread(str(args.raw_frame)),args.output_dir,args.tolerance_m)
    write_json(args.output_dir/'manifest.json',dict(survey_sha256=sha256_file(args.survey),
        raw_frame_sha256=sha256_file(args.raw_frame),survey=survey,profile=profile.model_dump()))
    print(json.dumps(dict(ruler_heldout_max_m=check['heldout_max_error_m'],
        ground_post_heldout_max_m=ground['max_post_heldout_error_m'],accuracy_validated=False)))


if __name__=='__main__':
    main()
