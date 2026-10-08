"""Joint pixel-domain fit of one rigid vehicle and its road poses.

Ground contacts are noisy observations, not exact poses. All frames share one
wheelbase and two physical endpoints; each frame has a road position and heading.
Camera calibration stays fixed. This is conditional geometry, not a field
accuracy certificate and not a landmark detector.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def refine_component(camera, cameras, times, translations, headings, contacts_uv,
                     contact_sigma_px, wheelbase, endpoints_body, endpoint_rows,
                     *, max_nfev=250):
    """endpoint_rows contain camera_id, landmark index (rear=0), uv, sigma, time.

    The caller must validate pixel bounds, timestamps and interpolation support.
    Timestamps are in the common rig clock and angles have been unwrapped.
    """
    n=len(times)
    initial=np.r_[wheelbase,np.asarray(endpoints_body).ravel(),
                  np.column_stack([translations[:,:2],headings]).ravel()]
    lower=np.full(len(initial),-np.inf);upper=np.full(len(initial),np.inf)
    lower[0]=.2
    lower[[3,6]]=0.  # Endpoints cannot be below the calibrated road plane.
    initial[[3,6]]=np.maximum(initial[[3,6]],1e-8)
    rows=[]
    for row in endpoint_rows:
        time=float(np.clip(row['time'],times[0],times[-1]))
        i=int(np.searchsorted(times,time))
        if i < n and abs(times[i]-time) <= 1e-9:
            lo=hi=i;alpha=0.
        else:
            if i==0 or i==n: raise ValueError('Endpoint time outside pose support')
            lo,hi=i-1,i
            alpha=(time-times[lo])/(times[hi]-times[lo])
        rows.append(dict(row,lo=lo,hi=hi,alpha=alpha))
    groups=[]
    for key in sorted({row['camera_id'] for row in rows}):
        selected=[r for r in rows if r['camera_id']==key]
        groups.append(dict(camera=cameras[key],camera_id=key,
            uv=np.asarray([r['uv'] for r in selected]),sigma=np.asarray([r['sigma'] for r in selected]),
            point=np.asarray([r['point'] for r in selected]),lo=np.asarray([r['lo'] for r in selected]),
            hi=np.asarray([r['hi'] for r in selected]),alpha=np.asarray([r['alpha'] for r in selected])))

    def errors(parameters):
        width=parameters[0];points=parameters[1:7].reshape(2,3)
        pose=parameters[7:].reshape(n,3)
        rear=np.c_[pose[:,:2],np.zeros(n)]
        forward=np.c_[np.cos(pose[:,2]),np.sin(pose[:,2]),np.zeros(n)]
        contacts=np.stack([rear,rear+width*forward],axis=1)
        residuals=[camera.project(contacts.reshape(-1,3)).reshape(n,2,2)-contacts_uv]
        for group in groups:
            alpha=group['alpha'][:,None]
            state=(1-alpha)*pose[group['lo']]+alpha*pose[group['hi']]
            body=points[group['point']]
            c,s=np.cos(state[:,2]),np.sin(state[:,2])
            world=np.column_stack([c*body[:,0]-s*body[:,1]+state[:,0],
                                   s*body[:,0]+c*body[:,1]+state[:,1],body[:,2]])
            residuals.append(group['camera'].project(world)-group['uv'])
        return residuals

    residual_count=4*n+2*len(rows)

    def residual(parameters):
        try:
            values=errors(parameters)
        except ValueError:
            return np.full(residual_count,1e6)
        return np.concatenate([(values[0]/contact_sigma_px[:,None,None]).ravel()]+
            [(v/g['sigma'][:,None]).ravel() for g,v in zip(groups,values[1:])])

    # Each observation only depends on its body point and neighbouring poses.
    # Sparse differences keep long passages from requiring a dense N-by-N matrix.
    sparsity=lil_matrix((residual_count,len(initial)),dtype=int)
    for i in range(n):
        sparsity[4*i:4*i+4,7+3*i:7+3*i+3]=1
        sparsity[4*i+2:4*i+4,0]=1
    offset=4*n
    for group in groups:
        for p,lo,hi in zip(group['point'],group['lo'],group['hi']):
            sparsity[offset:offset+2,1+3*p:4+3*p]=1
            sparsity[offset:offset+2,7+3*lo:10+3*lo]=1
            sparsity[offset:offset+2,7+3*hi:10+3*hi]=1
            offset+=2
    fit=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',
        jac_sparsity=sparsity.tocsr() if len(initial)>128 else None,
        x_scale='jac',max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    values=errors(fit.x)
    pose=fit.x[7:].reshape(n,3)
    endpoint_diagnostics={}
    for p,key in enumerate(('rear_extreme','front_extreme')):
        residuals=np.concatenate([v[g['point']==p] for g,v in zip(groups,values[1:])])
        endpoint_diagnostics[key]=dict(reprojection_rmse_px=float(np.sqrt(np.mean(residuals**2))))
    return dict(endpoints=fit.x[1:7].reshape(2,3),wheelbase_m=float(fit.x[0]),
        translations=np.c_[pose[:,:2],np.zeros(n)],headings=pose[:,2],
        endpoints_diagnostics=endpoint_diagnostics,
        diagnostics=dict(converged=bool(fit.success),evaluations=fit.nfev,
            contact_reprojection_rmse_px=float(np.sqrt(np.mean(values[0]**2))),
            contact_max_residual_px=float(np.max(np.linalg.norm(values[0],axis=2))),
            contact_sigma_px=contact_sigma_px.tolist(),
            model='Shared wheelbase and fixed physical endpoints; per-frame road pose; fixed calibrated cameras',
            accuracy_validated=False))
