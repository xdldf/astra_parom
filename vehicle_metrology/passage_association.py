"""Conservative identity checks for transferring a catalogue label between frames.

A sparse scout track is only a proposal. Each additional training view must be
connected to its labelled anchor by detections on every intervening video frame,
in both directions. Failure rejects the extra view, never the anchor passage.
"""
import math

from .detection import box_iou, suppress_duplicate_boxes


def _valid_box(box, image_size):
    width, height = image_size
    return (len(box) == 4 and all(math.isfinite(v) for v in box)
            and box[2] > 0 and box[3] > 0 and box[0] > 2 and box[1] > 2
            and box[0]+box[2] < width-2 and box[1]+box[3] < height-2)


def _walk(start, end, frames, image_size):
    step = 1 if end['frame'] >= start['frame'] else -1
    current = start['bbox']
    trace = []
    for number in range(start['frame'], end['frame']+step, step):
        if number not in frames:
            return dict(status='rejected', reason='missing_intermediate_frame', frame=number, trace=trace)
        detections = suppress_duplicate_boxes(frames[number])
        ranked = sorted([(box_iou(current, d['bbox']), i, d['bbox'])
                         for i, d in enumerate(detections)], reverse=True)
        threshold = .8 if number == start['frame'] else .65
        if not ranked or ranked[0][0] < threshold:
            return dict(status='rejected', reason='lost_vehicle', frame=number, trace=trace)
        overlap, index, box = ranked[0]
        if len(ranked) > 1 and ranked[1][0] > .4:
            return dict(status='rejected', reason='competing_vehicle', frame=number, trace=trace)
        if not _valid_box(box, image_size):
            return dict(status='rejected', reason='clipped_vehicle', frame=number, trace=trace)
        if number != start['frame']:
            movement = [abs(box[i]+box[i+2]/2-current[i]-current[i+2]/2)/current[i+2] for i in (0, 1)]
            ratios = [box[i]/current[i] for i in (2, 3)]
            if movement[0] > .15 or movement[1] > .2 or any(not .8 <= r <= 1.25 for r in ratios):
                return dict(status='rejected', reason='discontinuous_motion_or_size', frame=number, trace=trace)
        trace.append(dict(frame=number, detection=index, bbox=box, overlap=overlap))
        current = box
    if box_iou(current, end['bbox']) < .8:
        return dict(status='rejected', reason='different_endpoint_vehicle', frame=end['frame'], trace=trace)
    return dict(status='accepted', trace=trace)


def associate_view(anchor, candidate, frames, image_size):
    """Return auditable bidirectional continuity evidence; use no length labels."""
    if not _valid_box(anchor['bbox'], image_size) or not _valid_box(candidate['bbox'], image_size):
        return dict(status='rejected', reason='invalid_or_clipped_endpoint')
    forward = _walk(anchor, candidate, frames, image_size)
    backward = _walk(candidate, anchor, frames, image_size)
    if forward['status'] != 'accepted' or backward['status'] != 'accepted':
        reason = forward.get('reason', backward.get('reason'))
        return dict(status='rejected', reason=reason, forward=forward, backward=backward)
    a = [(s['frame'], s['detection']) for s in forward['trace']]
    b = [(s['frame'], s['detection']) for s in reversed(backward['trace'])]
    if a != b:
        return dict(status='rejected', reason='direction_disagreement', forward=forward, backward=backward)
    return dict(status='accepted', forward=forward, backward=backward)
