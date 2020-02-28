
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
vy=-1*phi_fluc.deriv(1)
vx=phi_fluc.deriv(2)
energy_zonal=(vx**2+vy**2+n_fluc**2).zonal()
phi2_zonal=(phi_fluc**2).zonal()
del phi_fluc
ens_flux_zonal=0.5*((q_fluc**2)*vx).zonal()
del q_fluc
n_flux_zonal=(n_fluc*vx).zonal()
del n_fluc
#vort_flux=(vort_fluc*vx).zonal()
reynolds_stress_zonal=(vx*vy).zonal()
del vort_fluc

#build array corresponding to x points
x=[i for i in range(0,516)]
x=np.divide(x,515.)

#add in background density field

#beta=
#bg=(-0.5*beta*np.square(x)+0.5*beta)*515*0.1

print('adding in background density')
k=2
bg=k*515*0.1*x
bg=np.repeat(bg[np.newaxis,:,np.newaxis],trange,axis=0)
n_zonal.data=np.add(n_zonal.data,bg)

xpoints=128
tmin=10
tmax=100

print('coarse graining and packaging data')

#prepare data for analysis

#ens_flux_std=ens_flux.stddev(xpoints).clean(tmin)
ens_flux=ens_flux_zonal.mean_x(xpoints).clean(tmin,tmax)
#n_flux_std=n_flux.stddev(xpoints).clean(tmin)
n_flux=n_flux_zonal.mean_x(xpoints).clean(tmin,tmax)
#vort_flux_std=vort_flux.stddev(xpoints).clean(tmin)
#vort_flux=vort_flux.mean_x(xpoints).clean(tmin)
ens=ens_zonal.mean_x(xpoints).clean(tmin,tmax)
#print(ens.shape)
ens_x=ens_zonal.secants(xpoints)
#print(ens_x.dims)
ens_x=ens_x.clean(tmin,tmax)
#print(ens_x.shape)
vort=vort_zonal.mean_x(xpoints).clean(tmin,tmax)
vort_x=vort_zonal.secants(xpoints).clean(tmin,tmax)
#n=n_zonal.mean_x(xpoints).clean(tmin,tmax)
#phi=phi_zonal.mean_x(xpoints).clean(tmin,tmax)
reynolds_stress=reynolds_stress_zonal.mean_x(xpoints).clean(tmin,tmax)
energy=energy_zonal.mean_x(xpoints).clean(tmin,tmax)
phi2=phi2_zonal.mean_x(xpoints).clean(tmin,tmax)

vort_xx=vort_zonal.mean_d2(xpoints).clean(tmin,tmax)
#vort_xxx=vort_zonal.mean_d3(xpoints).clean(tmin,tmax)
n_xx=n_zonal.mean_d2(xpoints).clean(tmin,tmax)

#todo: compute averages over windows

n_x=n_zonal.secants(xpoints)
n_x=n_x.clean(tmin,tmax)
print(n_x.size)

#save the data
np.savez('cleaned_data_early.npz',n_flux=n_flux,energy=energy,phi2=phi2,n_xx=n_xx,vort_xx=vort_xx,reynolds_stress=reynolds_stress,ens=ens,ens_x=ens_x,vort=vort,n_x=n_x,vort_x=vort_x)

tmax=2000

print('coarse graining and packaging data')

#prepare data for analysis

#ens_flux_std=ens_flux.stddev(xpoints).clean(tmin)
ens_flux=ens_flux_zonal.mean_x(xpoints).clean(tmin,tmax)
#n_flux_std=n_flux.stddev(xpoints).clean(tmin)
n_flux=n_flux_zonal.mean_x(xpoints).clean(tmin,tmax)
#vort_flux_std=vort_flux.stddev(xpoints).clean(tmin)
#vort_flux=vort_flux.mean_x(xpoints).clean(tmin)
ens=ens_zonal.mean_x(xpoints).clean(tmin,tmax)
#print(ens.shape)
ens_x=ens_zonal.secants(xpoints)
#print(ens_x.dims)
ens_x=ens_x.clean(tmin,tmax)
#print(ens_x.shape)
vort=vort_zonal.mean_x(xpoints).clean(tmin,tmax)
vort_x=vort_zonal.secants(xpoints).clean(tmin,tmax)
#n=n_zonal.mean_x(xpoints).clean(tmin,tmax)
#phi=phi_zonal.mean_x(xpoints).clean(tmin,tmax)
reynolds_stress=reynolds_stress_zonal.mean_x(xpoints).clean(tmin,tmax)
energy=energy_zonal.mean_x(xpoints).clean(tmin,tmax)
phi2=phi2_zonal.mean_x(xpoints).clean(tmin,tmax)

vort_xx=vort_zonal.mean_d2(xpoints).clean(tmin,tmax)
#vort_xxx=vort_zonal.mean_d3(xpoints).clean(tmin,tmax)
n_xx=n_zonal.mean_d2(xpoints).clean(tmin,tmax)

#todo: compute averages over windows

n_x=n_zonal.secants(xpoints)
n_x=n_x.clean(tmin,tmax)
print(n_x.size)
np.savez('cleaned_data_all.npz',ens_flux=ens_flux,n_flux=n_flux,energy=energy,phi2=phi2,n_xx=n_xx,vort_xx=vort_xx,reynolds_stress=reynolds_stress,ens=ens,ens_x=ens_x,vort=vort,n_x=n_x,vort_x=vort_x)




