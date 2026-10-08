"""Conservative association across detector latency, including nonoverlapping boxes."""
from vehicle_metrology.detection import box_iou


def plausible_motion(a, b):
    x,y,w,h = a
    u,v,c,d = b
    return (min(w,c)>0 and min(h,d)>0 and .6 <= w/c <= 1.67 and .6 <= h/d <= 1.67
            and min(y+h,v+d)-max(y,v) >= .6*min(h,d)
            and abs(x+w/2-u-c/2) <= 2*max(w,c))


def match_track(tracks, detections, index, used, stamp):
    box = detections[index]['bbox']
    available = [t for t in tracks if t['id'] not in used]
    overlapping = [t for t in available if box_iou(t['box'],box) > .2]
    if overlapping:
        return max(overlapping,key=lambda t:box_iou(t['box'],box))
    candidates = [t for t in available if 0 < stamp-t['stamp'] <= 2 and plausible_motion(t['box'],box)]
    if len(candidates) != 1:
        return None
    track = candidates[0]
    # A weak motion match must be unique in both directions; no neighbour swaps.
    if sum(plausible_motion(track['box'],d['bbox']) for d in detections) != 1:
        return None
    return track
