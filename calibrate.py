"""Recorded-video checkerboard intrinsics and independently checked road survey pose."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from vehicle_metrology.calibration import calibrate_intrinsic_views, calibrate_survey
from vehicle_metrology.video import local_video, sha256_file, write_json


def extract_boards(video, cols, rows, square_m, stride=15):
    video=local_video(video)
    if cols < 3 or rows < 3 or square_m <= 0 or not np.isfinite(square_m) or stride < 1:
        raise ValueError('Invalid checkerboard inner corners, square size or stride')
    cap=cv2.VideoCapture(str(video))
    views=[]
    size=None
    xyz=np.zeros((rows*cols,3),float)
    xyz[:,:2]=np.mgrid[:cols,:rows].T.reshape(-1,2)*square_m
    index=0
    try:
        while True:
            ok,image=cap.read()
            if not ok: break
            current=(image.shape[1],image.shape[0])
            if size is not None and size != current:
                raise ValueError('Video image size changes')
            size=current
            if index%stride == 0:
                found,corners=cv2.findChessboardCornersSB(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(cols,rows),
                    flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                if found:
                    views.append(dict(id=f'{video.name}:frame-{index}',frame=index,
                        object_points=xyz.copy(),image_points=corners.reshape(-1,2).astype(float)))
            index+=1
    finally:
        cap.release()
    if size is None:
        raise ValueError('No decodable local video frames')
    return views,size


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    intrinsic=sub.add_parser('intrinsics',help='Fit one explicitly selected lens model from recorded checkerboards')
    intrinsic.add_argument('--video',required=True)
    intrinsic.add_argument('--heldout-video',required=True,help='Separate board poses not used to fit K/D')
    intrinsic.add_argument('--cols',type=int,required=True,help='Inner corners, not squares')
    intrinsic.add_argument('--rows',type=int,required=True)
    intrinsic.add_argument('--square-m',type=float,required=True)
    intrinsic.add_argument('--stride',type=int,default=15)
    intrinsic.add_argument('--model',choices=['brown','fisheye'],required=True)
    intrinsic.add_argument('--output',required=True)
    survey=sub.add_parser('survey',help='Fit metric Z=0 road pose, reject independent-check failures')
    survey.add_argument('--intrinsics',required=True)
    survey.add_argument('--survey',required=True)
    survey.add_argument('--calibration-id',required=True)
    survey.add_argument('--max-check-error-m',type=float,required=True)
    survey.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command == 'intrinsics':
            if sha256_file(local_video(args.video)) == sha256_file(local_video(args.heldout_video)):
                raise ValueError('Training and heldout video must be independent recordings')
            training,size=extract_boards(args.video,args.cols,args.rows,args.square_m,args.stride)
            heldout,check_size=extract_boards(args.heldout_video,args.cols,args.rows,args.square_m,args.stride)
            if size != check_size:
                raise ValueError('Training and heldout image_size differ')
            if len(training)<10 or len(heldout)<3:
                raise ValueError(f'Need at least 10 training and 3 heldout board views; found {len(training)} / {len(heldout)}. Diversity still needs manual review.')
            result=calibrate_intrinsic_views(training,heldout,image_size=size,model=args.model)
            result['source_hashes']={'training':sha256_file(args.video),'heldout':sha256_file(args.heldout_video)}
            result['board']={'cols':args.cols,'rows':args.rows,'square_m':args.square_m,'stride':args.stride}
        else:
            result=calibrate_survey(json.loads(Path(args.intrinsics).read_text(encoding='utf-8')),
                json.loads(Path(args.survey).read_text(encoding='utf-8')),calibration_id=args.calibration_id,
                max_check_error_m=args.max_check_error_m)
            result['source_hashes']={'intrinsics':sha256_file(args.intrinsics),'survey':sha256_file(args.survey)}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        write_json(args.output,result)
    except (ValueError,OSError,cv2.error) as exc:
        parser.exit(2,str(exc)+'\n')
    print(json.dumps({'output':str(Path(args.output).resolve()),'diagnostics':result['diagnostics']},allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
