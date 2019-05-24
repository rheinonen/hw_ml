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

ens_zonal=0.5*(q_fluc**2).zonal()

vx=phi_fluc.deriv(2)
del phi_fluc

ens_flux=0.5*((q_fluc**2)*vx).zonal()
del q_fluc

n_flux=(n_fluc*vx).zonal()
del n_fluc

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

tmin=25
n=n_zonal.clean(tmin,full_x=True)
vort=vort_zonal.clean(tmin,full_x=True)
ens=ens_zonal.clean(tmin,full_x=True)
ens_flux=ens_flux.clean(tmin,full_x=True)
n_flux=n_flux.clean(tmin,full_x=True)
vort_flux=vort_flux.clean(tmin,full_x=True)

print(n.size)
print(vort.size)
print(ens.size)
print(vort_flux.size)
print(ens_flux.size)
print(n_flux.size)

np.savez('cleaned_data_nonlocal.npz',n=n,ens=ens,vort=vort,ens_flux=ens_flux,vort_flux=vort_flux,n_flux=n_flux)
