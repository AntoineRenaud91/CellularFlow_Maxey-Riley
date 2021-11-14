import numpy as np
import multiprocessing as mp
from numpy import newaxis as nx
import matplotlib.pyplot as plt
from scipy.special import ellipk,ellipe
from tqdm import tqdm
import joblib as jl
import os
from matplotlib import rc
rc('text', usetex=True)

def u(x):
	return np.sin(np.sqrt(2)*x[1])/np.sqrt(2)
def v(x):
	return np.sin(np.sqrt(2)*x[0])/np.sqrt(2)
def Au(x):
	return np.sin(np.sqrt(2)*x[0])*np.cos(np.sqrt(2)*x[1])/np.sqrt(2)	
def Av(x):
	return np.cos(np.sqrt(2)*x[0])*np.sin(np.sqrt(2)*x[1])/np.sqrt(2)
def GradU(x):
	return np.cos(np.sqrt(2)*x)

def Overdamped_Step(x,dt,St,Pe,i):
	if St>0:
		if i%2==0:
			x[0]+=(u(x)-St*Au(x))*dt
			x[1]+=(v(x)-St*Av(x))*dt
		else:
			x[1]+=(v(x)-St*Av(x))*dt
			x[0]+=(u(x)-St*Au(x))*dt
	else:
		if i%2==0:
			x[0]+=u(x)*dt
			x[1]+=v(x)*dt
		else:
			x[1]+=v(x)*dt
			x[0]+=u(x)*dt
	x+=np.sqrt(2*dt/Pe)*np.random.normal(0,1,x.shape)
	return x

def Langevin_Step(x,dt,St,Pe,i):
	if np.round(St/dt)>10:
		if i%2==0:
			x[0]+=(u(x[:2])+x[2]/St)*dt
			x[1]+=(v(x[:2])+x[3]/St)*dt
		else:
			x[1]+=(v(x[:2])+x[3]/St)*dt
			x[0]+=(u(x[:2])+x[2]/St)*dt
		ux=np.array([u(x[:2])[:],v(x[:2])[:]])
		gx=GradU(x[:2])
		aux=np.array([Au(x[:2])[:],Av(x[:2])[:]])
		x[2:]+=-x[2:]*dt/St+gx*x[2:][::-1]*dt-St*aux*dt
		#x[2:]+=np.sqrt(2*dt/Pe)*np.random.normal(0,1,x[2:].shape)
	else: 
		Ni=np.int_(np.round(dt*10/St))+1
		dti=dt/Ni
		for it in np.arange(Ni)+i:
			if i%2==0:
				x[0]+=(u(x[:2])+x[2]/St)*dti
				x[1]+=(v(x[:2])+x[3]/St)*dti
			else:
				x[1]+=(v(x[:2])+x[3]/St)*dti
				x[0]+=(u(x[:2])+x[2]/St)*dti
			ux=np.array([u(x[:2])[:],v(x[:2])[:]])
			gx=GradU(x[:2])
			aux=np.array([Au(x[:2])[:],Av(x[:2])[:]])
			x[2:]+=-x[2:]*dti/St+gx*x[2:][::-1]*dti-St*aux*dti		
			x[2:]+=np.sqrt(2*dti/Pe)*np.random.normal(0,1,x[2:].shape)
	return x


Pe=1000
St=0.0001
tmax=2*10**4		

rotmat=np.array([[1,-1],[1,1]])/np.sqrt(2)
nreal=10**4
dt	= 0.05
fid='./data/Comp_Pe-%d' %Pe
fid+='_St-%01.04f.npz' %St
if os.path.isfile(fid):
	z=np.load(fid)['z']
	t=np.load(fid)['t']
else:
	z=rotmat.dot(np.random.random((2,nreal))*2*np.pi-np.pi)
	z=np.concatenate((z,np.zeros((2,nreal))),axis=0)
	t=0
it=0
while t<tmax:
	it+=1
	t+=dt
	z=Langevin_Step(z,dt,St,Pe,it)
	if it%1000==0:
		np.savez(fid,z=z,t=t)
np.savez(fid,z=z,t=t)
