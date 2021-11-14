import numpy as np
import multiprocessing as mp
from numpy import newaxis as nx
import matplotlib.pyplot as plt
from scipy.special import ellipk,ellipe,lambertw
from scipy.interpolate import interp1d
from scipy.integrate import quad
from tqdm import tqdm
import joblib as jl
import os
from matplotlib import rc
rc('text', usetex=True)

q=10

x=np.linspace(-np.pi,np.pi,1000)
xx,yy=np.meshgrid(x,x,indexing='ij')
psi=(np.cos(xx)*np.exp(-q*np.sin(xx)**2)+np.cos(yy)*np.exp(-q*np.sin(yy)**2))/2
plt.figure(1)
plt.contourf(psi)
plt.show()
x=np.linspace(0,2*np.pi,20)
xx,yy=np.meshgrid(x,x,indexing='ij')
u=np.sin(yy)*(1+q+q*np.cos(2*yy))*np.exp(-q*np.sin(yy)**2)/2
v=-np.sin(xx)*(1+q+q*np.cos(2*xx))*np.exp(-q*np.sin(xx)**2)/2
plt.figure(2)
plt.quiver(xx,yy,u,v)
plt.show()



def u(x):
	return np.sin(x[1])*(1+q+q*np.cos(2*x[1]))*np.exp(-q*np.sin(x[1])**2)/2
def v(x):
	return -np.sin(x[0])*(1+q+q*np.cos(2*x[0]))*np.exp(-q*np.sin(x[0])**2)/2
def Au(x):
	return v(x)*np.cos(x[1])*(2-q*(q+4)+8*q*np.cos(2*x[1])+q**2*np.cos(4*x[1]))*np.exp(-q*np.sin(x[1])**2)/4
def Av(x):
	return -u(x)*np.cos(x[0])*(2-q*(q+4)+8*q*np.cos(2*x[0])+q**2*np.cos(4*x[0]))*np.exp(-q*np.sin(x[0])**2)/4

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

def Comp(Psi,Toplot=False):
	dt	= 0.001
	x=np.zeros((2,1))
	x[0]=np.arccos(np.sqrt(lambertw(2*np.exp(2*q)*q*Psi**2).real)/np.sqrt(2*q))
	x[1]=np.arccos(np.sqrt(lambertw(2*np.exp(2*q)*q*Psi**2).real)/np.sqrt(2*q))
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
	while x[1,-1]>=0:
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
	return a,b,c


dpsi=10**-2
Psi=np.arange(dpsi,1,dpsi)
nPsi=Psi.size
Output=np.array([Comp(Psi[ipsi]) for ipsi in tqdm(np.arange(nPsi))])

plt.plot(Psi,Output[:,0])
plt.show()
plt.plot(Psi,Output[:,1])
plt.show()
plt.plot(Psi,Output[:,2])
plt.show()
plt.plot(Psi,Output[:,1]/Output[:,0])
plt.show()
plt.plot(Psi,np.cumsum(Output[:,1]/Output[:,0])*dpsi)
plt.show()




# def Z(alpha,a,b,c,psi):
# 	gg=interp1d(np.hstack((0,psi)),np.hstack((0,b/a)),fill_value="extrapolate")
# 	cc=interp1d(psi,c,fill_value="extrapolate")
# 	bigG=lambda y: quad(gg,0,y)[0]
# 	bg=np.array([bigG(Psi[10*i]) for i in tqdm(np.arange(Psi.size//10))])




# def g(x):
# 	gg=np.zeros((x.shape))
# 	gg[x==1]=1
# 	y=x[(x>0)&(x<1)]
# 	gg[(x>0)&(x<1)]=y*(ellipk(1-y**2)-ellipe(1-y**2))/(ellipe(1-y**2)-y**2*ellipk(1-y**2))
# 	gg[x==1]=1
# 	return gg
# def G(x):
# 	nx=x.size
# 	n=100
# 	xx,yy=np.meshgrid(x,np.linspace(0,1,n,endpoint=False),indexing='ij')
# 	gg=np.sum(g(xx*yy)*xx/n,axis=-1)
# 	return gg

# def Z(a):
# 	n=100+np.int_(5*a)
# 	x=np.linspace(0,1,n+1)[1:]
# 	dx=x[0]
# 	Gt=G(x)
# 	return np.sum(ellipk(1-x**2)*np.exp(-Gt*a))*dx