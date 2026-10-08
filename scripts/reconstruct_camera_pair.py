#!/usr/bin/env python3
"""Offline diagnostic reconstruction from physically identified passage landmarks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vehicle_metrology.rig import Rig, measure_rig_component, measure_rig_passage
from vehicle_metrology.video import sha256_file, write_json
from vehicle_metrology.evaluation import evaluate
from vehicle_metrology.image_coordinates import corrected_tracks_to_raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rig',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--tolerance-m',type=float,default=.1)
    parser.add_argument('--ground-truth',type=Path,help='Independent physically measured validation CSV')
    parser.add_argument('--display-profile',action='append',default=[],metavar='CAMERA=PATH',
                        help='Profile for corrected_full_resolution annotations; repeat for every rig camera')
    args=parser.parse_args()
    data=json.loads(args.observations.read_text())
    rig=Rig.from_dict(json.loads(args.rig.read_text()))
    profile_paths={}
    try:
        for entry in args.display_profile:
            key,separator,path=entry.partition('=')
            if not separator or not key or not path or key in profile_paths:
                raise ValueError('Display profiles must be distinct CAMERA=PATH entries')
            profile_paths[key]=Path(path)
        if profile_paths:
            data=corrected_tracks_to_raw(data,{k:json.loads(p.read_text()) for k,p in profile_paths.items()},rig)
    except (ValueError,OSError,KeyError) as exc:
        parser.error(str(exc))
    if data.get('schema_version') != 1 or data.get('coordinate_space') != 'raw_distorted_pixels' or data.get('rig_id') != rig.rig_id:
        parser.error('Observation schema, coordinate space and rig_id must match the rig')
    inputs={args.rig.resolve(),args.observations.resolve()}
    inputs.update(p.resolve() for p in profile_paths.values())
    if args.ground_truth: inputs.add(args.ground_truth.resolve())
    if any((args.output_dir/name).resolve() in inputs for name in ('results.json','manifest.json','evaluation.json')):
        parser.error('Output would overwrite input evidence')
    tracks=[];ids=set()
    if not isinstance(data.get('tracks'),list) or not data['tracks']:
        parser.error('At least one annotated track is required')
    for track in data['tracks']:
        ident=track.get('track_id')
        if not isinstance(ident,str) or not ident.strip() or ident in ids:
            parser.error('Tracks need distinct nonempty track_id values')
        ids.add(ident)
        if 'components' in track:
            measured=measure_rig_passage(rig,track['components'],tolerance_m=args.tolerance_m,
                                        composition_verified=track.get('composition_verified') is True)
        else:
            measured=measure_rig_component(rig,track['pose_observations'],track['endpoints'],
                tolerance_m=args.tolerance_m,complete_vehicle=track.get('complete_vehicle') is True)
        tracks.append(dict(track_id=ident,**measured))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/'results.json',dict(tracks=tracks,accuracy_validated=False))
    if args.ground_truth:
        write_json(args.output_dir/'evaluation.json',evaluate(tracks,args.ground_truth,tolerance_m=args.tolerance_m))
    write_json(args.output_dir/'manifest.json',dict(rig_sha256=sha256_file(args.rig),
        observations_sha256=sha256_file(args.observations),tolerance_m=args.tolerance_m,
        ground_truth_sha256=sha256_file(args.ground_truth) if args.ground_truth else None,
        display_profile_sha256={k:sha256_file(p) for k,p in profile_paths.items()},
        note='Conditional reconstruction of supplied landmarks, not automatic detection or validated field accuracy.'))
    print(json.dumps(dict(tracks=len(tracks),measured=sum(t['length_m'] is not None for t in tracks),accuracy_validated=False)))


if __name__=='__main__': main()
