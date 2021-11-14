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


q=20

x=np.linspace(-np.pi,np.pi,1000)
xx,yy=np.meshgrid(x,x,indexing='ij')
psi=(np.cos(xx)/(q-(q-1)*np.cos(xx)**2)+np.cos(yy)/(q-(q-1)*np.cos(yy)**2))/2
plt.figure(1)
plt.contourf(psi)
x=np.linspace(0,2*np.pi,20)
xx,yy=np.meshgrid(x,x,indexing='ij')
u= np.sin(yy)*(q+(q-1)*np.cos(yy)**2)/(q-(q-1)*np.cos(yy)**2)/2
v=-np.sin(xx)*(q+(q-1)*np.cos(xx)**2)/(q-(q-1)*np.cos(xx)**2)/2
plt.figure(2)
plt.quiver(xx,yy,u,v)
plt.show()



def u(x):
	return np.sin(x[1])*(q+(q-1)*np.cos(x[1])**2)/(q-(q-1)*np.cos(x[1])**2)/2
def v(x):
	return -np.sin(x[0])*(q+(q-1)*np.cos(x[0])**2)/(q-(q-1)*np.cos(x[0])**2)/2
def Au(x):
	return -v(x)*np.cos(x[1])*(-5+(34-21*q)*q+4*(-1+q)*(1+5*q)*np.cos(2*x[1])+(-1+q)**2*np.cos(4*x[1]))/16/(-q+(-1+q)*np.cos(x[1])**2)**3
def Av(x):
	return u(x)*np.cos(x[0])*(-5+(34-21*q)*q+4*(-1+q)*(1+5*q)*np.cos(2*x[0])+(-1+q)**2*np.cos(4*x[0]))/16/(-q+(-1+q)*np.cos(x[0])**2)**3

def Overdamped_Step(x,dt,St,Pe,i):
	if St!=0:
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


def Comp(Pe,a,tmax,Toplot=False):
	Pe=np.int_(Pe)
	St=a/Pe
	nreal=10**4
	dt	= 0.01
	fid='./data/x_Pe-%d' %Pe
	fid+='_a-%d.npz' %a
	if os.path.isfile(fid):
		y=np.load(fid)['y']
		t=np.load(fid)['t']
	else:
		y=(np.random.random((2,nreal))*2*np.pi-np.pi)
		t=0
	if Toplot:
		Y=y/2/np.pi
		plt.ion()
		fig=plt.figure()
		ax=fig.add_subplot(111)
		PlotB,=ax.plot(Y[0],Y[1],'ro',Markersize=0.1,alpha=0.5)
		xb=np.floor(np.abs(Y).max())+1
		fig.canvas.draw()
		fig.canvas.flush_events()
	it=0
	while t<tmax:
		it+=1
		t+=dt
		y=Overdamped_Step(y,dt,St,Pe,it)
		if it%1000==0:
			np.savez(fid,y=y,t=t)
			if Toplot:
				Y=y/2/np.pi
				PlotB.set_xdata(Y[0])
				PlotB.set_ydata(Y[1])
				xb=np.floor(np.abs(Y).max())+1
				ax.set_xlim(-xb,xb)
				ax.set_ylim(-xb,xb)
				fig.canvas.draw()
				fig.canvas.flush_events()
	np.savez(fid,y=y,t=t)
	return 


def CompSimple(St,tmax,Toplot=False):
	nreal=10**4
	dt	= 0.01
	y=(np.random.random((2,nreal))*2*np.pi-np.pi)
	ind=np.abs(y[0])+np.abs(y[1])>np.pi
	while np.any(ind):
		y[:,ind]=(np.random.random((2,ind.sum()))*2*np.pi-np.pi)
		ind=np.abs(y[0])+np.abs(y[1])>np.pi
	t=0
	if Toplot:
		plt.ion()
		fig=plt.figure()
		ax=fig.add_subplot(111)
		PlotB,=ax.plot(y[0],y[1],'ro',Markersize=0.1,alpha=0.5)
		fig.canvas.draw()
		fig.canvas.flush_events()
	it=0
	while t<tmax:
		it+=1
		t+=dt
		y=Overdamped_Step(y,dt,St,np.inf,it)
		if it%10==0:
			if Toplot:
				PlotB.set_xdata(y[0])
				PlotB.set_ydata(y[1])
				fig.canvas.draw()
				fig.canvas.flush_events()
	return 



Pe=np.array([10**3,10**4,10**5])
alpha=np.array([0,20,40,60])
nPe=Pe.size
na=alpha.size

tmax=10**4

jl.Parallel(n_jobs=8)(jl.delayed(Comp)(Pe[iPe],alpha[ia],tmax) for ia in tqdm(np.arange(na)) for iPe in np.arange(nPe))

