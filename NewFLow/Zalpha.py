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
	return np.sin(x[1])/2
def v(x):
	return -np.sin(x[0])/2
def GradU(x):
	return -np.sin(x)/2
def Au(x):
	return v(x)*np.cos(x[1])/2
def Av(x):
	return -u(x)*np.cos(x[0])/2

def abc(x,t):
	xm=(x[:,1:]+x[:,:-1])/2
	u2=u(xm)**2+v(xm)**2
	grdPsiuGrdU=v(xm)*Au(xm)-u(xm)*Av(xm)
	dt=np.diff(t)
	return np.sum(u2*dt),np.sum(grdPsiuGrdU*dt),t[-1]

def Orbital_Step(x,dt,i):
	if i%2==0:
		x[0]+=u(x)*dt
		x[1]+=v(x)*dt
	else:
		x[1]+=v(x)*dt
		x[0]+=u(x)*dt
	return x

def Comp(Psi,Toplot=False,output=None):
	dt	= 0.001
	fid='./data_Za/Psi-%01.05f.npz' %Psi
	if not os.path.isfile(fid):
		x=np.zeros((2,1))
		x[0]=0
		x[1]=np.arccos(2*Psi-1)
		t=np.array([0])
		it=-1
		if Toplot:
			plt.ion()
			fig=plt.figure()
			ax=fig.add_subplot(111)
			Plot,=ax.plot(x[0],x[1],'-',linewidth=0.6)
			ax.set_xlim(-np.pi,np.pi)
			ax.set_ylim(-np.pi,np.pi)
			fig.canvas.draw()
			fig.canvas.flush_events()
		while x[1,-1]>0:
			it+=1
			t=np.append(t,t[-1]+dt)
			x=np.concatenate((x,Orbital_Step(x[:,-1],dt,it)[:,np.newaxis]),axis=-1)
			if Toplot&(it%50==0):
				Plot.set_xdata(x[0])
				Plot.set_ydata(x[1])
				fig.canvas.draw()
				fig.canvas.flush_events()
		x=x[:,:-1]
		t=t[:-1]
		a,b,c=abc(x,t)
		a,b,c=4*a,4*b,4*c
		np.savez(fid,a=a,b=b,c=c)
	if output is None:
		return 
	else:
		return np.load(fid)[output]

nPsi=100
Psi=np.linspace(0,1,nPsi+2)[1:-1]
a=np.array([Comp(Psi[ipsi],output='a') for ipsi in tqdm(np.arange(nPsi))])
b=np.array([Comp(Psi[ipsi],output='b') for ipsi in tqdm(np.arange(nPsi))])
c=np.array([Comp(Psi[ipsi],output='c') for ipsi in tqdm(np.arange(nPsi))])

plt.plot(Psi,c/2)
plt.plot(Psi,4*ellipk(1-Psi**2))
plt.show()

plt.plot(Psi,a)
plt.plot(Psi,8*ellipe(1-Psi**2)-8*Psi**2*ellipk(1-Psi**2))
plt.show()

plt.plot(Psi,2*b)
plt.plot(Psi,8*Psi*(ellipk(1-Psi**2)-ellipe(1-Psi**2)))
plt.show()


