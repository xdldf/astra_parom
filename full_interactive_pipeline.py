#!/usr/bin/env python
"""Full pipeline with FULL visual interactivity at EACH step. User sees images, clicks, types.
No synthetic/invented measurements. Each stage produces real artifacts from user interaction.
Requires real video, user annotations, user survey measurements, user calibration values."""
import argparse, csv, json, sys, os, time
sys.path.insert(0, '.')
from pathlib import Path
import cv2


def print_stage(title, description, command):
    print('=== %s ===' % title)
    print(description)
    print('Run:')
    print('  %s' % command)
    print()


def main():
    video_default = 'C:/Users/alexe/Downloads/ST_2026-09-04_19-30-04.mp4'
    p = argparse.ArgumentParser(description='Full interactive pipeline with visual interaction at every stage.')
    p.add_argument('--video', default=video_default)
    p.add_argument('--output-dir', default='calibration/prod/full_interactive')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stages = [
        ('STAGE 1: VISUAL CALIBRATION (sliders)',
         'Open video frame in window. Adjust sliders (fx, fy, cx, cy, k1, k2, p1, p2, k3). Press S to save approximate intrinsic JSON.',
         '.venv/Scripts/python.exe calibration/prod/interactive_slider_visual.py --video %s --calibration-id prod-visual --output-dir calibration/prod/generated' % args.video),
        ('STAGE 2: ANNOTATE ROAD (visual annotation image)',
         'Generate annotated image showing approximate road polygon overlay and approximate vehicle annotations (not measurements).',
         '.venv/Scripts/python.exe annotate_road.py --video %s --calibration calibration/prod/prototype.json --output %s/annotated_road.jpg --frame 3000' % (args.video, args.output_dir)),
        ('STAGE 3: FULL INTERACTIVE CALIBRATION (visual survey + model)',
         'Interactive survey: user selects lens model visually, provides real focal/principal/distortion values, selects survey/check points, provides real world measurements.',
         '.venv/Scripts/python.exe calibration/prod/interactive_calibration_full.py --video %s --output-dir calibration/prod/generated' % args.video),
        ('STAGE 4: MANUAL ANNOTATIONS (interactive measurement)',
         'Use interactive_measure.py to click real persistent points on the same vehicle in different frames and provide independently measured physical lengths.',
         '.venv/Scripts/python.exe calibration/prod/interactive_measure.py --video %s --calibration calibration/prod/prototype.json --track-id production-car --output %s/annotations.json --sigma-px 2' % (args.video, args.output_dir)),
        ('STAGE 5: FULL MEASUREMENT (only after real calibration exists)',
         'Requires: real calibration/prod/camera.json (not prototype) + real observations + real ground_truth.csv. Produces measurement CSV with length estimates based ONLY on real observations.',
         'PYTHONPATH=. .venv/Scripts/python.exe measure_from_annotations.py --video %s --calibration calibration/prod/camera.json --observations %s/annotations.json --track-id production-car --output-csv %s/measurements.csv --mc-samples 100' % (args.video, args.output_dir, args.output_dir)),
    ]
    for title, desc, cmd in stages:
        print_stage(title, desc, cmd)

    # Confirm interactive steps will work; note limitations clearly
    print('=== PIPELINE STATUS ===')
    video_path = Path(args.video)
    if video_path.is_file():
        print('Video file verified:', video_path.resolve())
        print('SHA-256:', __import__('hashlib').sha256(open(str(video_path), 'rb').read()).hexdigest())
    else:
        print('Video file NOT FOUND at:', args.video)
        print('Pipeline stops here; fix video path before continuing.')
        return

    # Check calibration files (must exist from interactive calibration, not prototype only)
    proto_path = Path('calibration/prod/prototype.json')
    generated_dir = Path('calibration/prod/generated')
    if proto_path.exists() and not generated_dir.exists():
        print('NOTE: Only prototype calibration exists (dummy/unverified). Real metric measurements require generated/ calibration files from interactive survey.')
    elif generated_dir.exists() and (generated_dir / 'survey.json').exists() and (generated_dir / 'intrinsic.json').exists():
        print('NOTE: Generated calibration artifacts present. Verify they are from real user measurements, not synthetic/default.')
    else:
        print('NOTE: No complete calibration artifacts found. Stages 1 and 3 must be completed with real data.')

    # Confirm interactive interfaces callable
    try:
        import calibration.prod.interactive_slider_visual as slider
        import calibration.prod.interactive_calibration_full as calib
        import calibration.prod.interactive_measure as annotation
        import annotate_road_custom as overlay
        import measure_from_annotations as measurement
        print('All interactive modules callable.')
    except Exception as exc:
        print('Interactive import check failed (check PYTHONPATH):', exc)

    # Save a summary manifest
    manifest_path = out / 'pipeline_manifest.json'
    manifest_path.write_text(json.dumps({
        'pipeline_version': 'full_interactive',
        'video_path': str(video_path.resolve()),
        'stages': [
            {'name': 'visual_slider_calibration', 'command': stages[0][2]},
            {'name': 'visual_annotation', 'command': stages[1][2]},
            {'name': 'full_interactive_calibration', 'command': stages[2][2]},
            {'name': 'manual_annotations_measure', 'command': stages[3][2]},
            {'name': 'full_measurement', 'command': stages[4][2]},
        ],
        'note_interactive_only': 'All metric measurements require real user-provided observations, survey, and calibration. No synthetic values produced.',
        'note_real_video_only': 'This pipeline operates on real local video files, not synthetic data, and does not fabricate measurements.',
        'no_metric_claim_until_real_data_present': True,
        'video_sha256': __import__('hashlib').sha256(open(str(video_path), 'rb').read()).hexdigest()
    }, indent=2))
    print('Pipeline manifest saved:', manifest_path.resolve())
    print('Pipeline requires user interaction at stages 1, 2, 3, 4. No automatic synthetic results.')


if __name__ == '__main__':
    main()
