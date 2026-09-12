"""Offline recorded-video experiment, not a real-world accuracy certification."""
import argparse
import json
import sys

from vehicle_metrology.video import run_video


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video_positional', nargs='?', help='existing local recorded video file; no live input')
    parser.add_argument('--video', help='existing local recorded video file; no live input')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--output-video')
    parser.add_argument('--calibration')
    parser.add_argument('--observations')
    parser.add_argument('--ground-truth')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mc-samples', type=int, default=0)
    args = parser.parse_args(argv)
    positional = args.video_positional
    del args.video_positional
    if bool(positional) == bool(args.video):
        parser.error('Provide exactly one video path, positional or --video')
    args.video = args.video or positional
    try:
        result = run_video(**vars(args))
    except (ValueError, OSError) as exc:
        print(json.dumps({'status': 'error', 'reason': str(exc)}, allow_nan=False), file=sys.stderr)
        return 2
    print(json.dumps({'status': 'complete', 'decoded_frames': result['video']['decoded_frames'],
                      'tracks': len(result['tracks']), 'disclaimer': result['detector']['disclaimer']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
