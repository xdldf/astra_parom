"""Fit a distorted planar ruler image and test a separate rail-to-road mapping.

Metre marks on raised barriers calibrate their plane, not the road underneath.
Vehicle dimensions never participate in these fits. All input pixels are raw.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def _points(value, size):
    points=np.asarray(value,float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx2 raw pixels')
    if np.any(points<0) or np.any(points>=size):
        raise ValueError('Survey points must be inside the original image')
    return points


def _matrix(p):
    return np.r_[p[:8],1.].reshape(3,3)


def undistort(pixels, parameters, width, focal=.47):
    k1,k2,cx,cy=parameters[8:12]
    K=np.array([[focal,0,cx],[0,focal,cy],[0,0,1.]])
    return cv2.undistortPointsIter(np.asarray(pixels,float).reshape(-1,1,2)/width,K,
        np.array([k1,k2,0,0,0]),None,K,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12))[:,0]


def plane_points(pixels, parameters, width):
    return cv2.perspectiveTransform(undistort(pixels,parameters,width)[None],np.linalg.inv(_matrix(parameters)))[0]


def _world(parameters, distances):
    a,b,angle=parameters[12:]
    return np.r_[np.c_[distances[0],np.zeros(len(distances[0]))],
                 np.c_[a+distances[1]*np.cos(angle),b+distances[1]*np.sin(angle)]]


def _project(parameters, distances):
    world=_world(parameters,distances)
    q=cv2.perspectiveTransform(world[None],_matrix(parameters))[0]
    k1,k2,cx,cy=parameters[8:12]
    c=np.array([cx,cy]);v=(q-c)/.47;r2=np.sum(v*v,axis=1)
    return c+.47*v*(1+k1*r2+k2*r2*r2)[:,None]


def fit_rulers(survey, *, heldout=(), initial=None):
    """Joint Brown radial correction + projective metric ruler fit.

Focal length fixes the distortion parameterization; it is not an independently
    calibrated physical focal length. Row 2 has a fitted origin and direction.
    Width endpoints are approximate digitizations of the user's distance arrows.
    """
    size=np.asarray(survey['image_size'],float)
    if size.shape!=(2,) or np.any(size<=0) or not np.isfinite(size).all():
        raise ValueError('Invalid image_size')
    width,height=size
    rows=survey['rulers']
    if len(rows)!=2:
        raise ValueError('Two spatially separated rulers are required')
    pixels=np.vstack([_points(r['points_px'],size) for r in rows])
    distances=[np.asarray(r['distances_m'],float) for r in rows]
    sigmas=[]
    for row,dist in zip(rows,distances):
        if len(dist)!=len(row['points_px']) or len(dist)<6 or not np.isfinite(dist).all() or np.any(np.diff(dist)<=0):
            raise ValueError('Each ruler needs >=6 strictly ordered metric positions')
        sigma=float(row.get('sigma_px',2.))
        if not np.isfinite(sigma) or sigma<=0:
            raise ValueError('Positive digitization uncertainty required')
        sigmas.extend([sigma]*len(dist))
    sigmas=np.asarray(sigmas)
    widths=survey['widths']
    if len(widths)<2:
        raise ValueError('At least two road-width constraints are required')
    width_pixels=np.vstack([_points(d['points_px'],size) for d in widths])
    if any(len(d['points_px'])!=2 for d in widths):
        raise ValueError('Width constraints need two endpoints')
    lengths=np.array([d['distance_m'] for d in widths],float)
    if not np.isfinite(lengths).all() or np.any(lengths<=0):
        raise ValueError('Positive measured widths required')
    heldout=np.asarray(heldout,dtype=int)
    if np.any(heldout<0) or np.any(heldout>=len(pixels)) or len(set(heldout))!=len(heldout):
        raise ValueError('Invalid held-out indices')
    keep=np.setdiff1d(np.arange(len(pixels)),heldout)
    if any(sum((keep>=start)&(keep<start+len(dist)))<6 for start,dist in
           [(0,distances[0]),(len(distances[0]),distances[1])]):
        raise ValueError('Need six training marks on each ruler after holding out points')
    if initial is None:
        world=np.r_[np.c_[distances[0],np.zeros(len(distances[0]))],
                    np.c_[2.3+distances[1],lengths[0]-.025*distances[1]]]
        H,_=cv2.findHomography(world[keep],pixels[keep]/width)
        initial=np.r_[H.ravel()[:8],-.2,.04,.5,height/(2*width),2.3,lengths[0],-.025]
    def residual(p):
        pred=_project(p,distances)
        q=plane_points(width_pixels,p,width).reshape(-1,2,2)
        measured=np.linalg.norm(q[:,0]-q[:,1],axis=1)
        # Weak principal-point regularization and the annotation-width uncertainty
        # are explicit. They do not imply surveyed endpoint precision.
        return np.r_[((pred[keep]-pixels[keep]/width)*width/sigmas[keep,None]).ravel(),
                     (measured-lengths)/.1,(p[10:12]-[.5,height/(2*width)])/.1]
    lower=np.r_[np.full(8,-np.inf),-.6,-.1,.4,.28,-2,3,-.15]
    upper=np.r_[np.full(8,np.inf),.1,.3,.6,.48,6,9,.15]
    fitted=least_squares(residual,initial,bounds=(lower,upper),max_nfev=1500,x_scale='jac',
                         ftol=1e-11,xtol=1e-11,gtol=1e-11)
    p=fitted.x
    world=_world(p,distances)
    recovered=plane_points(pixels,p,width)
    errors=np.linalg.norm(recovered-world,axis=1)
    k1,k2,cx,cy=p[8:12]
    radii=np.linspace(0,1.5,200)
    monotonic=bool(np.min(1+3*k1*radii**2+5*k2*radii**4)>0)
    return dict(parameters=p.tolist(),lens=dict(k1=k1,k2=k2,cx=cx,cy=cy*width/height,
                 focal=.47,zoom=.91,tilt_deg=3.6),world_points=world.tolist(),
                 point_errors_m=errors.tolist(),heldout_indices=heldout.tolist(),
                 heldout_max_error_m=float(errors[heldout].max()) if len(heldout) else None,
                 training_max_error_m=float(errors[keep].max()),optimizer_success=bool(fitted.success),
                 radial_monotonic=monotonic,accuracy_validated=False)


def fit_ground_transfer(survey, fit):
    """Test the common-height vertical-post hypothesis, including leave-one-out.

Never promote a failed fit into a surveyed ground plane. Feet can be hidden,
    posts can lean, and the road/barriers can have different grades or heights.
    """
    size=np.asarray(survey['image_size'],float)
    posts=survey.get('posts',[])
    if len(posts)<4:
        raise ValueError('At least four visible rail-top/foot pairs are required')
    pixels=np.array([_points(r['points_px'],size) for r in posts])
    if pixels.shape!=(len(posts),2,2):
        raise ValueError('Each post needs one top and one foot')
    parameters=np.array(fit['parameters']);H=_matrix(parameters)
    top=undistort(pixels[:,0],parameters,size[0]);bottom=undistort(pixels[:,1],parameters,size[0])
    world=cv2.perspectiveTransform(top[None],np.linalg.inv(H))[0]
    q=np.c_[world,np.ones(len(world))]@H.T
    def solve(indices):
        def residual(offset):
            pred=q[indices]+offset
            return ((pred[:,:2]/pred[:,2,None]-bottom[indices])*size[0]).ravel()
        found=least_squares(residual,[0,.03,-.01],max_nfev=1000)
        ground=H.copy();ground[:,2]+=found.x
        return ground
    ground=solve(np.arange(len(posts)))
    errors=[]
    for i in range(len(posts)):
        trial=solve(np.delete(np.arange(len(posts)),i))
        recovered=cv2.perspectiveTransform(bottom[i:i+1][None],np.linalg.inv(trial))[0,0]
        errors.append(float(np.linalg.norm(recovered-world[i])))
    return dict(normalized_world_to_ground_image=ground.tolist(),
                 post_heldout_errors_m=[dict(name=row['name'],error_m=e) for row,e in zip(posts,errors)],
                 max_post_heldout_error_m=max(errors),height_source='inferred_from_visible_posts',
                 barrier_height_m=survey.get('barrier_height_m'),accuracy_validated=False)


def corrected_projection(fit, ground, image_size):
    w,h=image_size;lens=fit['lens'];z=lens['zoom'];cx=lens['cx']*w;cy=lens['cy']*h
    zoom=np.array([[z,0,cx*(1-z)],[0,z,cy*(1-z)],[0,0,1.]])
    rotation=np.vstack([cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),-lens['tilt_deg'],1),[0,0,1]])
    return rotation@zoom@np.diag([w,w,1.])@np.asarray(ground['normalized_world_to_ground_image'])


def projective_span(bbox, projection):
    """Approximate aligned vehicle span at the depth of its bottom centre.

Correctly follows the longitudinal vanishing point instead of assuming that
    two equal image y coordinates have equal road depth. It is not a 3D bumper fit.
    """
    x,y,w,h=bbox;H=np.asarray(projection,float)
    bottom=np.array([[[x+w/2,y+h]]],float)
    point=cv2.perspectiveTransform(bottom,np.linalg.inv(H))[0,0];Y=point[1]
    denominators=np.array([H[0,0]-u*H[2,0] for u in (x,x+w)])
    if np.any(np.abs(denominators)<1e-9):
        raise ValueError('Vehicle span crosses the longitudinal vanishing point')
    ends=np.array([(u*(H[2,1]*Y+H[2,2])-H[0,1]*Y-H[0,2])/den
                   for u,den in zip((x,x+w),denominators)])
    if not np.isfinite(ends).all():
        raise ValueError('Invalid projected span')
    return float(abs(ends[1]-ends[0])),np.c_[ends,[Y,Y]]
