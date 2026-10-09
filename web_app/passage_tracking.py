"""Conservative association across detector latency, including nonoverlapping boxes."""
from vehicle_metrology.detection import box_iou
import uuid
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def station_profile(profile):
    """Station capture always has a centre gate, including legacy profiles."""
    if profile.measurement_line_x is None:
        return profile.model_copy(update={'measurement_line_x':profile.image_size[0]/2})
    return profile


def contained_fragment(track, box):
    """Retain the identity of an already claimed car as its exit box shrinks."""
    if not (track.get('sent') or track.get('pending')):
        return False
    x,y,w,h=box; a,b,c,d=track['box']
    if min(w,h,c,d)<=0 or w>1.1*c or not .45*d<=h<=1.2*d:
        return False
    area=max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
    return area/(w*h)>=.8


def plausible_motion(a, b):
    x,y,w,h = a
    u,v,c,d = b
    return (min(w,c)>0 and min(h,d)>0 and .6 <= w/c <= 1.67 and .6 <= h/d <= 1.67
            and min(y+h,v+d)-max(y,v) >= .6*min(h,d)
            and abs(x+w/2-u-c/2) <= 2*max(w,c))


def associate(tracks, detections, stamp):
    """Assign the whole frame together so one car cannot steal its neighbour."""
    if not tracks or not detections:
        return {}
    # Each detection can remain unmatched via its own dummy column.
    costs=np.full((len(detections),len(tracks)+len(detections)),3.)
    for j,track in enumerate(tracks):
        a,b,c,d=track['box'];predicted=a+c/2
        previous=track.get('previous')
        if previous and track['stamp']>previous['time']:
            old=previous['box']
            shift=(predicted-old[0]-old[2]/2)/(track['stamp']-previous['time'])*min(2,max(0,stamp-track['stamp']))
            predicted+=max(-2*c,min(2*c,shift))
        for i,detection in enumerate(detections):
            box=detection['bbox'];x,y,w,h=box
            overlap=box_iou(track['box'],box)
            allowed=(overlap>.2 or contained_fragment(track,box) or
                     0<stamp-track['stamp']<=2 and plausible_motion(track['box'],box))
            if not allowed:
                costs[i,j]=1e6
                continue
            costs[i,j]=(abs(x+w/2-predicted)/max(w,c)+abs(y+h/2-b-d/2)/max(h,d)
                        +.5*abs(math.log(w/c))+.25*abs(math.log(h/d))+.2*(1-overlap))
    # Near-identical alternatives are ambiguous, unlike separated followers.
    blocked=[]
    for j in range(len(tracks)):
        ranked=sorted(range(len(detections)),key=lambda i:costs[i,j])
        if len(ranked)>1:
            i,k=ranked[:2]
            if costs[k,j]<3 and costs[k,j]-costs[i,j]<.15 and box_iou(detections[i]['bbox'],detections[k]['bbox'])>.5:
                blocked.extend([(i,j),(k,j)])
    for i in range(len(detections)):
        ranked=sorted(range(len(tracks)),key=lambda j:costs[i,j])
        if len(ranked)>1:
            j,k=ranked[:2]
            if costs[i,k]<3 and costs[i,k]-costs[i,j]<.15 and box_iou(tracks[j]['box'],tracks[k]['box'])>.5:
                blocked.extend([(i,j),(i,k)])
    for i,j in blocked:
        costs[i,j]=1e6
    rows,columns=linear_sum_assignment(costs)
    return {i:tracks[j] for i,j in zip(rows,columns) if j<len(tracks) and costs[i,j]<3}


def match_track(tracks, detections, index, used, stamp):
    """Compatibility for offline audit callers consuming a batch one row at a time."""
    track=associate(tracks,detections,stamp).get(index)
    return track if track is not None and track['id'] not in used else None


def update_tracks(tracks, detections, stamp):
    """One identity from approach through exit; inference delay is not absence."""
    active=[t for t in tracks if t.get('pending') or t.get('missing_since') is None
            or stamp-t['missing_since']<2]
    assigned=[None]*len(detections);used=set()
    matched=associate(active,detections,stamp)
    # A full car claims its track before a nested exit/duplicate proposal.
    for index in sorted(range(len(detections)),key=lambda i:-detections[i]['bbox'][2]*detections[i]['bbox'][3]):
        detection=detections[index];box=detection['bbox']
        track=matched.get(index)
        if track is None:
            fragments=[t for t in active if t['id'] in used and contained_fragment(t,box)]
            if len(fragments)==1:
                assigned[index]=fragments[0]
                continue
            track=dict(id=uuid.uuid4().hex,sent=False,history=[],observations=0,max_width=box[2],max_height=box[3])
            active.append(track)
        track['previous']=dict(box=track['box'],time=track['stamp']) if 'box' in track else None
        track.update(box=box,stamp=stamp,missing_since=None,
                     max_width=max(track.get('max_width',0),box[2]),max_height=max(track.get('max_height',0),box[3]),
                     observations=track.get('observations',0)+1)
        track.setdefault('history',[]).append(dict(time=stamp,box=list(box)))
        track['history']=track['history'][-128:]
        used.add(track['id']);assigned[index]=track
    for track in active:
        if track['id'] not in used and track.get('missing_since') is None:
            track['missing_since']=stamp
    return active,assigned


def capture_ready(track, detection):
    """Require an established whole-car track, not a new fragment at the line."""
    w,h=detection['bbox'][2:]
    return (not track.get('sent') and not track.get('pending')
            and track.get('observations',0)>=2
            and w>=.75*track['max_width'] and h>=.65*track['max_height'])


def observed_track(tracks, stamp, box):
    matches=[t for t in tracks if any(abs(row['time']-stamp)<1e-5 and box_iou(row['box'],box)>.8
                                     for row in t.get('history',[]))]
    return matches[0] if len(matches)==1 else None
