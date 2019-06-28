# computes the (mean) hurst exponent for 1+1 dimensional spatiotemporal data.
# inputs are assumed to have shape (time, xpoints)

import numpy as np
import math as math

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

def h_exp(v):
    h=res_range(v)
    size=v.shape[0]
    xmax=v.shape[1]
    h=np.reshape(h,(xmax,1))
    curr_len=size
    lens=[curr_len]
    while curr_len>=8:
        curr_len=math.floor(curr_len/2)
        num=math.floor(size/curr_len)
        total=np.zeros(xmax)
        for i in range(0,num):
            total=total+res_range(v[i*curr_len:(i+1)*curr_len])
        total=total/num
        h=np.concatenate((h,np.reshape(total,(xmax,1))),axis=1)
        lens.append(curr_len)
    return lens,h
