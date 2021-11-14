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

q=21

x=np.linspace(-np.pi,np.pi,1000)
xx,yy=np.meshgrid(x,x,indexing='ij')
psi=(np.cos(xx)/(q+1-q*np.cos(xx)**2)+np.cos(yy)/(q+1-q*np.cos(yy)**2))/2
plt.figure(1)
plt.contourf(psi)
x=np.linspace(0,2*np.pi,20)
xx,yy=np.meshgrid(x,x,indexing='ij')
u= np.sin(yy)*(q+1+q*np.cos(yy)**2)/(q+1-q*np.cos(yy)**2)/2
v=-np.sin(xx)*(q+1+q*np.cos(xx)**2)/(q+1-q*np.cos(xx)**2)/2
plt.figure(2)
plt.quiver(xx,yy,u,v)
plt.show()

def u(y):
	return (1+q+q*np.cos(y)**2)*np.sin(y)/2/(1+q-q*np.cos(y)**2)
def v(x):
	return -(1+q+q*np.cos(x)**2)*np.sin(x)/2/(1+q-q*np.cos(x)**2)
def Au(x,y):
	return v(x)*np.cos(y)*(8-q*(8+21*q)+4*q*(6+5*q)*np.cos(2*y)+q**2*np.cos(4*y))/2/(2+q-q*np.cos(2*y))**3
def Av(x,y):
	return -u(y)*np.cos(x)*(8-q*(8+21*q)+4*q*(6+5*q)*np.cos(2*x)+q**2*np.cos(4*x))/2/(2+q-q*np.cos(2*x))**3

def Y(psi,x):
	def Yt(psi):
		return np.arccos((-1+np.sqrt(1+16*q*psi**2+16*q**2*psi**2))/4/psi/q)
	return Yt(psi-np.cos(x)/2/(q+1-q*np.cos(x)**2))

def abc(x,y):
	xm=(x[1:]+x[:-1])/2
	ym=(y[1:]+y[:-1])/2
	dx=np.sqrt(np.diff(x)**2+np.diff(y)**2)
	u2=u(ym)**2+v(xm)**2
	dt=dx/u2**0.5
	grdPsiuGrdU=v(xm)*Au(xm,ym)-u(ym)*Av(xm,ym)
	return np.sum(u2*dt),np.sum(grdPsiuGrdU*dt),np.sum(dt)

def Comp(psi):
	nx=10000
	xmax=Y(psi,0)
	x=np.linspace(0,xmax,nx)
	y=Y(psi,x)
	a,b,c=abc(x,y)
	a,b,c=4*a,4*b,4*c
	return a,b,c

nPsi=1005
Psi=np.linspace(0,1,nPsi+2)[1:-1]
Integrals=np.array([Comp(Psi[ipsi]) for ipsi in tqdm(np.arange(nPsi))]).swapaxes(0,1)
a=interp1d(Psi[np.logical_not(np.isnan(Integrals[0,:]))],Integrals[0,np.logical_not(np.isnan(Integrals[0,:]))],fill_value='extrapolate')
b=interp1d(Psi[np.logical_not(np.isnan(Integrals[1,:]))],Integrals[1,np.logical_not(np.isnan(Integrals[1,:]))],fill_value='extrapolate')
c=interp1d(Psi[np.logical_not(np.isnan(Integrals[2,:]))],Integrals[2,np.logical_not(np.isnan(Integrals[2,:]))],fill_value='extrapolate')
g=np.cumsum(b(Psi)/a(Psi))*Psi[0]

psilim=Psi[np.argmin(g)]
nx=10000
xmax=Y(psilim,0)
x=np.linspace(0,xmax,nx)
y=Y(psilim,x)
plt.loglog(x,y,'k-')
plt.show()