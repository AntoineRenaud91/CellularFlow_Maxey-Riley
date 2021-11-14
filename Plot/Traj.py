import numpy as np
import multiprocessing as mp
from numpy import newaxis as nx
import matplotlib.pyplot as plt
from scipy.special import ellipk,ellipe
from tqdm import tqdm
import joblib as jl
import os
from matplotlib import rc
from matplotlib.collections import LineCollection
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

def Overdamped_Step(x,dt,St,i):
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
	return x

St=0.1
Pe=1000
tmax=500

rotmat=np.array([[1,-1],[1,1]])/np.sqrt(2)
nreal=24
dt	= 0.05
nt=np.int_(tmax/dt)
x1=np.zeros((2,nt+1))
x2=np.zeros((2,nt+1))
x3=np.zeros((2,nt+1))
y1=np.random.random(2)*np.pi
y1=rotmat.dot(y1)
x1[:,0]=rotmat.transpose().dot(y1)/2/np.pi
y2=np.copy(y1)
x2[:,0]=rotmat.transpose().dot(y2)/2/np.pi
y3=np.copy(y1)
x3[:,0]=rotmat.transpose().dot(y3)/2/np.pi
t=0
it=0
for it in np.arange(nt):
	t+=dt
	rand=np.sqrt(2/Pe)*np.random.normal(0,1,y1.shape)
	y1=Overdamped_Step(y1,dt,-St,it)+rand
	y2=Overdamped_Step(y2,dt,0,it)+rand
	y3=Overdamped_Step(y3,dt,St,it)+rand
	x1[:,it+1]=rotmat.transpose().dot(y1)/2/np.pi
	x2[:,it+1]=rotmat.transpose().dot(y2)/2/np.pi
	x3[:,it+1]=rotmat.transpose().dot(y3)/2/np.pi

xmean1=np.round(x1[0].mean()/0.5)*0.5
ymean1=np.round(x1[1].mean()/0.5)*0.5
xmean2=np.round(x2[0].mean()/0.5)*0.5
ymean2=np.round(x2[1].mean()/0.5)*0.5
xmean3=np.round(x3[0].mean()/0.5)*0.5
ymean3=np.round(x3[1].mean()/0.5)*0.5

x1[0]-=xmean1
x2[0]-=xmean2
x3[0]-=xmean3
x1[1]-=ymean1
x2[1]-=ymean2
x3[1]-=ymean3

xmin1=np.round(x1.min()/0.5)*0.5-0.5
xmax1=np.round(x1.max()/0.5)*0.5+0.5
xmin2=np.round(x2.min()/0.5)*0.5-0.5
xmax2=np.round(x2.max()/0.5)*0.5+0.5
xmin3=np.round(x3.min()/0.5)*0.5-0.5
xmax3=np.round(x3.max()/0.5)*0.5+0.5

t=np.arange(nt+1)*dt

plt.close('all')
fig=plt.figure(figsize=(12,6))
ax1=fig.add_subplot(131,aspect=1)
for i in np.arange(xmin1,xmax1+0.5,0.5):
	ax1.plot([i,i],[xmin1,xmax1],'k-',Linewidth=0.5,alpha=0.5)
	ax1.plot([xmin1,xmax1],[i,i],'k-',Linewidth=0.5,alpha=0.5)
	for j in np.arange(xmin1,xmax1+0.5,0.5):
		Psi=0.5
		x=np.linspace(np.arcsin(Psi),np.pi-np.arcsin(Psi),1000)
		y=np.arcsin(Psi/np.sin(x))
		ax1.plot(i+x/2/np.pi,j+y/2/np.pi-0.005,'k-',Linewidth=0.5,alpha=0.5)
		ax1.plot(i+x/2/np.pi,j+0.5-y/2/np.pi-0.005,'k-',Linewidth=0.5,alpha=0.5)
ax1.plot(x1[0],x1[1],color='k')
ax1.set_xlim(xmin1,xmax1)
ax1.set_ylim(xmin1,xmax1)
ax1.set_xticks([])
ax1.set_yticks([])

ax2=fig.add_subplot(132,aspect=1)
for i in np.arange(xmin2,xmax2+0.5,0.5):
	ax2.plot([i,i],[xmin2,xmax2],'k-',Linewidth=0.5,alpha=0.5)
	ax2.plot([xmin2,xmax2],[i,i],'k-',Linewidth=0.5,alpha=0.5)
	for j in np.arange(xmin2,xmax2+0.5,0.5):
		Psi=0.5
		x=np.linspace(np.arcsin(Psi),np.pi-np.arcsin(Psi),1000)
		y=np.arcsin(Psi/np.sin(x))
		ax2.plot(i+x/2/np.pi,j+y/2/np.pi-0.005,'k-',Linewidth=0.5,alpha=0.5)
		ax2.plot(i+x/2/np.pi,j+0.5-y/2/np.pi-0.005,'k-',Linewidth=0.5,alpha=0.5)
ax2.plot(x2[0],x2[1],color='k')
ax2.set_xlim(xmin2,xmax2)
ax2.set_ylim(xmin2,xmax2)
ax2.set_xticks([])
ax2.set_yticks([])

ax3=fig.add_subplot(133,aspect=1)
for i in np.arange(xmin3,xmax3+0.5,0.5):
	ax3.plot([i,i],[xmin3,xmax3],'k-',Linewidth=0.5,alpha=0.5)
	ax3.plot([xmin3,xmax3],[i,i],'k-',Linewidth=0.5,alpha=0.5)
	for j in np.arange(xmin3,xmax3+0.5,0.5):
		Psi=0.5
		x=np.linspace(np.arcsin(Psi),np.pi-np.arcsin(Psi),1000)
		y=np.arcsin(Psi/np.sin(x))
		ax3.plot(i+x/2/np.pi,j+y/2/np.pi-0.005,'k-',Linewidth=0.5,alpha=0.5)
		ax3.plot(i+x/2/np.pi,j+0.5-y/2/np.pi-0.005,'k-',Linewidth=0.5,alpha=0.5)
ax3.plot(x3[0],x3[1],color='k')
ax3.set_xlim(xmin3,xmax3)
ax3.set_ylim(xmin3,xmax3)
ax3.set_xticks([])
ax3.set_yticks([])


plt.savefig('Traj_%01.03f.png' %St,dpi=200,Linewidth=0.5)
plt.show()