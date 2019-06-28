# coding: utf-8
import hurst
import bout_field as bf
n=bf.Field(name='vort',dx=0.1)
n.collect()
n=n.zonal()
total=0
for i in range(0,1):
	l,h=hurst.h_exp(n.data[50:50+128,2:514,0])
	import numpy as np
	h_mean=np.mean(h,axis=0)
	h_std=np.std(h,axis=0)
	params=np.polyfit(np.log(l),np.log(h_mean),1)
	print(params[1])
	total=params[1]+total
total=total/7
print(total)

