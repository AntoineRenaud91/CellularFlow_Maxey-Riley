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
import sys
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

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

Pe=1000
St=0.1
tmax=33

rotmat=np.array([[1,-1],[1,1]])/np.sqrt(2)
nreal=10**5
dt	= 0.05
y1=rotmat.dot(np.random.random((2,nreal))*np.pi*0.95+np.pi*0.025)
y2=rotmat.dot(np.random.random((2,nreal))*np.pi*0.95+np.pi*0.025)
y3=rotmat.dot(np.random.random((2,nreal))*np.pi*0.95+np.pi*0.025)
t=0
Y1=rotmat.transpose().dot(y1)/2/np.pi
Y2=rotmat.transpose().dot(y2)/2/np.pi
Y3=rotmat.transpose().dot(y3)/2/np.pi
plt.ion()
fig=plt.figure()
ax=fig.add_subplot(111)
PlotA,=ax.plot(Y1[0],Y1[1],'ro',Markersize=0.1,alpha=0.2)
PlotB,=ax.plot(Y2[0],Y2[1],'ro',Markersize=0.1,alpha=0.2)
PlotC,=ax.plot(Y3[0],Y3[1],'ro',Markersize=0.1,alpha=0.2)
xb=np.floor(np.abs(Y3).max())+1
ax.set_xlim(-xb,xb)
ax.set_ylim(-xb,xb)
fig.canvas.draw()
fig.canvas.flush_events()
it=0
while t<tmax:
	it+=1
	progress(it, np.int_(tmax/dt), status='Computing...')
	t+=dt
	y1=Overdamped_Step(y1,dt,-St,Pe,it)
	y2=Overdamped_Step(y2,dt,0,Pe,it)
	y3=Overdamped_Step(y3,dt,St,Pe,it)
	if it%10==0:
		Y1=rotmat.transpose().dot(y1)/2/np.pi
		Y2=rotmat.transpose().dot(y2)/2/np.pi
		Y3=rotmat.transpose().dot(y3)/2/np.pi
		PlotA.set_xdata(Y1[0])
		PlotA.set_ydata(Y1[1])
		PlotB.set_xdata(Y2[0])
		PlotB.set_ydata(Y2[1])
		PlotC.set_xdata(Y3[0])
		PlotC.set_ydata(Y3[1])
		xb=np.floor(np.abs(Y3).max())+1
		ax.set_xlim(-xb,xb)
		ax.set_ylim(-xb,xb)
		fig.canvas.draw()
		fig.canvas.flush_events()

xmin1=np.round(Y1.min()/0.5)*0.5-0.5
xmax1=np.round(Y1.max()/0.5)*0.5+0.5
xmin2=np.round(Y2.min()/0.5)*0.5-0.5
xmax2=np.round(Y2.max()/0.5)*0.5+0.5
xmin3=np.round(Y3.min()/0.5)*0.5-0.5
xmax3=np.round(Y3.max()/0.5)*0.5+0.5


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
ax1.plot(Y1[0],Y1[1],'o',color='dodgerblue',Markersize=0.2,alpha=0.4)
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
ax2.plot(Y2[0],Y2[1],'o',color='dodgerblue',Markersize=0.2,alpha=0.4)
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
ax3.plot(Y3[0],Y3[1],'o',color='dodgerblue',Markersize=0.2,alpha=0.5)
ax3.set_xlim(xmin3,xmax3)
ax3.set_ylim(xmin3,xmax3)
ax3.set_xticks([])
ax3.set_yticks([])


plt.savefig('Ensemble_%01.03f.png' %St,dpi=200,Linewidth=0.5)
plt.show()