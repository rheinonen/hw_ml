from boutdata.collect import collect
from boututils import calculus as calc
import math
import numpy as np
import bout_field as bf


phi=bf.Field(name='phi',dx=0.1)
n=bf.Field(name='n',dx=0.1)
vort=bf.Field(name='vort',dx=0.1)

print('collecting data')
phi.collect()
n.collect()
vort.collect()

print('computing mean fields')
#extract relevant mean fields
phi_zonal=phi.zonal()
phi_fluc=phi.fluc()
n_zonal=n.zonal()
n_fluc=n.fluc()
vort_zonal=vort.zonal()
vort_fluc=vort.fluc()

trange=n.dims[0]
#clear some memory
del phi
del n
del vort

q_fluc=n_fluc-vort_fluc

del n_fluc

ens_zonal=0.5*(q_fluc**2).zonal()
del q_fluc
vx=phi_fluc.deriv(2)
del phi_fluc
#ens_flux=0.5*((q_fluc**2)*vx).zonal()
#n_flux=(n_fluc*vx).zonal()
vort_flux=(vort_fluc*vx).zonal()
del vort_fluc
#build array corresponding to x points
x=[i for i in range(0,516)]
x=np.divide(x,515.)

#add in background density field

#beta=
#bg=(-0.5*beta*np.square(x)+0.5*beta)*515*0.1

print('adding in background density')
k=1
bg=k*515*0.1*x
bg=np.repeat(bg[np.newaxis,:,np.newaxis],trange,axis=0)
n_zonal.data=np.add(n_zonal.data,bg)

xpoints=16
tmin=25

print('coarse graining and packaging data')

#prepare data for analysis

#ens_flux_std=ens_flux.stddev(xpoints).clean(tmin)
#ens_flux=ens_flux.mean_x(xpoints).clean(tmin)
#n_flux_std=n_flux.stddev(xpoints).clean(tmin)
#n_flux=n_flux.mean_x(xpoints).clean(tmin)
#vort_flux_std=vort_flux.stddev(xpoints).clean(tmin)
vort_flux=vort_flux.mean_x(xpoints).clean(tmin)
ens=ens_zonal.mean_x(xpoints).clean(tmin)
#print(ens.shape)
#ens_x=ens_zonal.secants(xpoints)
#print(ens_x.dims)
#ens_x=ens_x.clean(tmin)
#print(ens_x.shape)
vort=vort_zonal.mean_x(xpoints).clean(tmin)
vort_x=vort_zonal.secants(xpoints).clean(tmin)
#n=n_zonal.mean_x(xpoints).clean(tmin)
#phi=phi_zonal.mean_x(xpoints).clean(tmin)

vort_xx=vort_zonal.mean_d2(xpoints).clean(tmin)
vort_xxx=vort_zonal.mean_d3(xpoints).clean(tmin)

#todo: compute averages over windows

n_x=n_zonal.secants(xpoints)
n_x=n_x.clean(tmin)
print(n_x.size)

#phi_x=phi_zonal.secants(xpoints).clean()

#save the data
np.savez('cleaned_data_vort.npz',vort_flux=vort_flux,ens=ens,vort=vort,n_x=n_x,vort_x=vort_x,vort_xx=vort_xx,vort_xxx=vort_xxx)
