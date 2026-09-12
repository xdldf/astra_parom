"""Generate an explicitly synthetic recorded-video integration experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .geometry import Camera


def generate(directory, seed=7, noise_px=0.25):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    R = np.array([[1,0,0],[0,-1/np.sqrt(5),-2/np.sqrt(5)],[0,2/np.sqrt(5),-1/np.sqrt(5)]])
    cam = Camera.from_dict(dict(schema_version=1,image_size=[1280,720],model='fisheye',
        K=[[650,0,640],[0,650,360],[0,0,1]],D=[-.03,.001,0,0],Rcw=R.tolist(),
        tcw=(-R@np.array([0,-10,5])).tolist(),road_polygon=[[-20,-4],[20,-4],[20,12],[-20,12]],
        calibration_id='SYNTHETIC-NOT-A-REAL-CAMERA'))
    paths = {k:directory/v for k,v in dict(video='SYNTHETIC.avi',calibration='calibration.json',
        observations='observations.json',ground_truth='ground_truth.csv').items()}
    rng = np.random.default_rng(seed)
    tracks = [dict(track_id=f'synthetic-{i+1}',observations=[]) for i in range(3)]
    writer = cv2.VideoWriter(str(paths['video']),cv2.VideoWriter_fourcc(*'MJPG'),20,(1280,720))
    if not writer.isOpened():
        raise ValueError('Cannot create synthetic MJPG video')
    body = np.array([[0,0,0],[2.7,0,0],[-.9,.3,.6],[3.6,.4,.8]])
    box = np.array([[x,y,z] for z in (0.3,1.5) for y in (0,1.8) for x in (-.9,3.6)])
    try:
        for frame_number in range(120):
            image = np.full((720,1280,3),35,np.uint8)
            cv2.putText(image,'SYNTHETIC GEOMETRY TEST - NOT REAL ACCURACY EVIDENCE',(25,40),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,255),2)
            for lane in (0,3,6):
                road = cam.project([[-15,lane,0],[15,lane,0]]).astype(int)
                cv2.line(image,tuple(road[0]),tuple(road[1]),(90,90,90),1)
            index = frame_number//40
            local = frame_number%40
            angle = [0,.15,-.15][index]
            A = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            T = np.array([-6+12*local/39, [0,3,6][index],0])
            uv = cam.project(body@A.T+T)
            corners = cam.project(box@A.T+T).astype(np.int32)
            cv2.fillConvexPoly(image,cv2.convexHull(corners),(140,90,40))
            for p,color in zip(uv,[(255,100,0),(0,220,0),(255,0,255),(0,255,255)]):
                cv2.circle(image,tuple(np.rint(p).astype(int)),5,color,-1)
            measured = uv+rng.normal(0,noise_px,uv.shape)
            if local%2 == 0:
                tracks[index]['observations'].append(dict(frame=frame_number,sigma_px=max(noise_px,0.01),
                    **dict(zip(['rear_contact_uv','front_contact_uv','rear_uv','front_uv'],measured.tolist()))))
            writer.write(image)
    finally:
        writer.release()
    paths['calibration'].write_text(json.dumps(cam.to_dict(),indent=2,allow_nan=False))
    digest = hashlib.sha256(paths['video'].read_bytes()).hexdigest()
    paths['observations'].write_text(json.dumps(dict(schema_version=1,video_sha256=digest,tracks=tracks),indent=2,allow_nan=False))
    with paths['ground_truth'].open('w',newline='') as stream:
        w=csv.writer(stream)
        w.writerow(['track_id','vehicle_id','length_m'])
        for track in tracks:
            w.writerow([track['track_id'],'same-synthetic-rigid-body',4.5])
    return paths


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='examples/synthetic')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--noise-px',type=float,default=.25)
    args=parser.parse_args()
    print(json.dumps({k:str(v.resolve()) for k,v in generate(args.output_dir,args.seed,args.noise_px).items()},indent=2))
