from boutdata.collect import collect
from boututils import calculus as calc
import math
import numpy as np

class Field:
    def __init__(self,*args,**kwargs):
        self.name=kwargs.get('name',None)
        self.data=[]
        self.dims=None
        if(len(args)!=0):
            self.data=args[0]
            self.dims=self.data.shape
        self.dx=kwargs.get('dx',0.1)

    def collect(self):
        self.data=np.squeeze(collect(self.name))
        self.dims=self.data.shape

    def zonal(self,ax=2):
        y=np.mean(self.data,axis=ax,keepdims=True)
        x=Field(y,dx=self.dx)
        return x;

    def fluc(self,ax=2):
        return self-self.zonal()


    def __add__(self,other):
        y=Field(np.add(self.data,other.data),dx=self.dx)
        return y;

    def __sub__(self,other):
        y=Field(np.subtract(self.data,other.data),dx=self.dx)
        return y;

    def __mul__(self,other):
        try:
            y=Field(np.multiply(self.data,other.data),dx=self.dx)
            return y;
        except:
            y=Field(np.multiply(self.data,other),dx=self.dx)
            return y;

    def __rmul__(self,other):
            y=Field(np.multiply(self.data,other),dx=self.dx)
            return y;

    def deriv(self,ax):
        #z=a
        #for i in range(a.shape[0]):
            #z[i,:,:]=calc.deriv2D(np.squeeze(a[i,:,:]),axis=ax,dx=dx,noise_suppression=false)
        #return z
        x=np.subtract(np.roll(self.data,-1,axis=ax),self.data)/self.dx
        y=Field(x,dx=self.dx)
        return y;

    # returns a version of data averaged over xpoints rectangular regions in x
    # future: add window function?
    def mean_x(self,xpoints=5,guards=2):
        nx=self.dims[1]-2*guards
        xstep=math.floor((float(nx)-1)/float(xpoints))
        x=np.asarray([np.mean(self.data[:,int(guards+i*xstep):int(guards+(i+1)*xstep),:],axis=1) for i in range(0,xpoints)])
        y=np.moveaxis(x,0,1)
        z=Field(y,dx=xstep*self.dx)
        return z;

    def stddev(self,xpoints=5,guards=2):
        nx=self.dims[1]-2*guards
        xstep=math.floor((float(nx)-1)/float(xpoints))
        x=np.asarray([np.std(self.data[:,int(guards+i*xstep):int(guards+(i+1)*xstep),:],axis=1) for i in range(0,xpoints)])
        y=np.moveaxis(x,0,1)
        z=Field(y,dx=xstep*self.dx)
        return z;

    #  window average over N points along axis
    #def rolling_average(a,N,ax):
        #return np.apply_along_axis(lambda m: np.convolve(m, np.ones((N,))/N, mode='same'), axis=ax,arr=a)

    def clean(self,tmin=10,guards=2):
        y=self.data[tmin:,:,:]
        z=y.flatten()
        return z;

    def secants(self,xpoints=5,guards=2):
        nx=self.dims[1]-2*guards
        xstep=math.floor((float(nx)-1)/float(xpoints))
        x=np.asarray([np.subtract(self.data[:,int(guards+(i+1)*xstep),:],self.data[:,int(guards+i*xstep),:])/(self.dx*xstep) for i in range(0,xpoints)])
        y=np.moveaxis(x,0,1)
        z=Field(y,dx=xstep*self.dx)
        return z;

    def __pow__(self,other):
        y=Field(np.power(self.data,other),dx=self.dx)
        return y;

#get simulation data

phi=Field(name='phi',dx=0.1)
n=Field(name='n',dx=0.1)
vort=Field(name='vort',dx=0.1)

phi.collect()
n.collect()
vort.collect()

#extract relevant mean fields
phi_zonal=phi.zonal()
phi_fluc=phi.fluc()
n_zonal=n.zonal()
n_fluc=n.fluc()
vort_zonal=vort.zonal()
vort_fluc=vort.fluc()
q_fluc=n_fluc-vort_fluc
ens_zonal=0.5*(q_fluc**2).zonal()
vx=phi_fluc.deriv(2)
ens_flux=0.5*((q_fluc**2)*vx).zonal()
n_flux=(n_fluc*vx).zonal()
vort_flux=(vort_fluc*vx).zonal()

#build array corresponding to x points
x=[i for i in range(0,516)]
x=np.divide(x,515.)

#add in background density field#
#beta=
#bg=(-0.5*beta*np.square(x)+0.5*beta)*515*0.1


k=1
bg=k*515*0.1*x
bg=np.repeat(bg[np.newaxis,:,np.newaxis],n.dims[0],axis=0)
n_zonal.data=np.add(n_zonal.data,bg)

xpoints=16
tmin=10
#prepare data for analysis

ens_flux_std=ens_flux.stddev(xpoints).clean(tmin)
ens_flux=ens_flux.mean_x(xpoints).clean(tmin)
n_flux_std=n_flux.stddev(xpoints).clean(tmin)
n_flux=n_flux.mean_x(xpoints).clean(tmin)
vort_flux_std=vort_flux.stddev(xpoints).clean(tmin)
vort_flux=vort_flux.mean_x(xpoints).clean(tmin)
ens=ens_zonal.mean_x(xpoints).clean(tmin)
#print(ens.shape)
ens_x=ens_zonal.secants(xpoints)
#print(ens_x.dims)
ens_x=ens_x.clean(tmin)
#print(ens_x.shape)
vort=vort_zonal.mean_x(xpoints).clean(tmin)
vort_x=vort_zonal.secants(xpoints).clean(tmin)
n=n_zonal.mean_x(xpoints).clean(tmin)
phi=phi_zonal.mean_x(xpoints).clean(tmin)
n_x=n_zonal.secants(xpoints)
n_x=n_x.clean(tmin)
print(n_x.size)

#phi_x=phi_zonal.secants(xpoints).clean()

#save the data
np.savez('cleaned_data.npz',n_flux_std=n_flux_std,vort_flux_std=vort_flux_std,n_flux=n_flux,vort_flux=vort_flux,ens_flux_std=ens_flux_std,ens_flux=ens_flux,ens=ens,vort=vort,n=n,phi=phi,n_x=n_x,ens_x=ens_x,vort_x=vort_x)
