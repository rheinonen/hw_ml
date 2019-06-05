# computes the (mean) hurst exponent for 1+1 dimensional spatiotemporal data.
# inputs are assumed to have shape (time, xpoints)

import numpy as np

# accumulated deviation from the mean, as function of time.
def accum_dev(v):
    v_mean=np.mean(v,axis=0)
    out=np.zeros(v.shape)
    out[0,:]=v[0,:]-v_mean
    for i in range(1,v.shape[0]):
        out[i,:]=out[i-1,:]+v[i-1,:]-v_mean
    return out

def res_range(v):
    x=accum_dev(v)
    r=np.amax(x,axis=0)-np.amin(x,axis=0)
    s=np.std(v,axis=0)
    return r/s
