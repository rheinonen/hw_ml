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

    #get data from BOUT output files
    def collect(self):
        self.data=np.squeeze(collect(self.name))
        self.dims=self.data.shape

    #average over symmetry direction
    def zonal(self,ax=2):
        y=np.mean(self.data,axis=ax,keepdims=True)
        x=Field(y,dx=self.dx)
        return x;

    #fluctuation from zonal average
    def fluc(self,ax=2):
        return self-self.zonal()

    #overloading binary operators for elmementwise arithmetic on fields
    def __add__(self,other):
        y=Field(np.add(self.data,other.data),dx=self.dx)
        return y;

    def __sub__(self,other):
        y=Field(np.subtract(self.data,other.data),dx=self.dx)
        return y;

    #handles Field*scalar and Field*Field
    def __mul__(self,other):
        try:
            y=Field(np.multiply(self.data,other.data),dx=self.dx)
            return y;
        except:
            y=Field(np.multiply(self.data,other),dx=self.dx)
            return y;

    #handles scalar*Field
    def __rmul__(self,other):
            y=Field(np.multiply(self.data,other),dx=self.dx)
            return y;

    # discrete derivative along chosen axis
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
    def mean_x(self,xpoints=16,guards=2):
        nx=self.dims[1]-2*guards
        xstep=math.floor((float(nx)-1)/float(xpoints))
        x=np.asarray([np.mean(self.data[:,int(guards+i*xstep):int(guards+(i+1)*xstep),:],axis=1) for i in range(0,xpoints)])
        y=np.moveaxis(x,0,1)
        z=Field(y,dx=xstep*self.dx)
        return z;

    # returns the stddev from above coarse graining
    def stddev(self,xpoints=16,guards=2):
        nx=self.dims[1]-2*guards
        xstep=math.floor((float(nx)-1)/float(xpoints))
        x=np.asarray([np.std(self.data[:,int(guards+i*xstep):int(guards+(i+1)*xstep),:],axis=1) for i in range(0,xpoints)])
        y=np.moveaxis(x,0,1)
        z=Field(y,dx=xstep*self.dx)
        return z;

    #  window average over N points along axis
    #def rolling_average(a,N,ax):
        #return np.apply_along_axis(lambda m: np.convolve(m, np.ones((N,))/N, mode='same'), axis=ax,arr=a)

    # remove early time data and flatten vector to prepare for nn.py
    def clean(self,tmin=10,guards=2):
        y=self.data[tmin:,:,:]
        z=y.flatten()
        return z;

    # mean derivative over windows in x
    def secants(self,xpoints=16,guards=2):
        nx=self.dims[1]-2*guards
        xstep=math.floor((float(nx)-1)/float(xpoints))
        x=np.asarray([np.subtract(self.data[:,int(guards+(i+1)*xstep),:],self.data[:,int(guards+i*xstep),:])/(self.dx*xstep) for i in range(0,xpoints)])
        y=np.moveaxis(x,0,1)
        z=Field(y,dx=xstep*self.dx)
        return z;

    def __pow__(self,other):
        y=Field(np.power(self.data,other),dx=self.dx)
        return y;

    # mean second derivative using 5 point method
    def mean_d2(self,xpoints=16,guards=2):
        nx=self.dims[1]-2*guards
        xstep=math.floor((float(nx)-1)/float(xpoints))
        x=np.asarray([(-self.data[:,int(guards+(i+1)*xstep)+2,:]+8*self.data[:,int(guards+(i+1)*xstep)+1,:]-8*self.data[:,int(guards+(i+1)*xstep)-1,:]
            +self.data[:,int(guards+(i+1)*xstep)-2,:]+self.data[:,int(guards+i*xstep)+2,:]-8*self.data[:,int(guards+i*xstep)+1,:]
            +8*self.data[:,int(guards+i*xstep)-1,:]-self.data[:,int(guards+i*xstep)-2,:])/(12*self.dx)/(xstep*self.dx) for i in range(0,xpoints)])
        y=np.moveaxis(x,0,1)
        z=Field(y,dx=xstep*self.dx)
        return z;

    # mean third derivative using 5 point method
    def mean_d3(self,xpoints=16,guards=2):
        nx=self.dims[1]-2*guards
        xstep=math.floor((float(nx)-1)/float(xpoints))
        x=np.asarray([(-self.data[:,int(guards+(i+1)*xstep)+2,:]+16*self.data[:,int(guards+(i+1)*xstep)+1,:]-30*self.data[:,int(guards+(i+1)*xstep),:]+16*self.data[:,int(guards+(i+1)*xstep)-1,:]
            -self.data[:,int(guards+(i+1)*xstep)-2,:]+self.data[:,int(guards+i*xstep)+2,:]-16*self.data[:,int(guards+i*xstep)+1,:]+30*self.data[:,int(guards+i*xstep),:]
            -16*self.data[:,int(guards+i*xstep)-1,:]+self.data[:,int(guards+i*xstep)-2,:])/(12*self.dx**2)/(xstep*self.dx) for i in range(0,xpoints)])
        y=np.moveaxis(x,0,1)
        z=Field(y,dx=xstep*self.dx)
        return z;

    #returns cubic fit parameters for windows in x
    #def fit_cubic(self,xpoints=16,guards=2,tol=10e-4):
     #   nx=self.dims[1]-2*guards
      #  xstep=math.floor((float(nx)-1)/float(xpoints))
      #  params=np.zeros(self.dims[0],xpoints,4)
       # for t in range(0,self.dims[0]):
        #    for i in range(0,xpoints):
                #todo: appropriate library function for fit. probably needs parallelization as well
                #params[t,i,:]=fit(np.squeeze(self.data[t,int(guards+i*xstep):int(guards+(i+1)*xstep),:]),tol)
        #z=Field(params,dx=xstep*self.dx)
        #return z;
