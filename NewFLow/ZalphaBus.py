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
import cmocean
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
rc('text', usetex=True)

q=1

def Psi(x):
	return np.cos(x)**(2*q+1)/2

def dPsi(x):
	return -(1/2)*(1+2*q)*np.cos(x)**(2*q)*np.sin(x)

def ddPsi(x):
	return -(1/2)*(1+2*q)*np.cos(x)**(-1+2*q)*(np.cos(x)**2-2*q*np.sin(x)**2)


def u(y):
	return -dPsi(y)
def v(x):
	return dPsi(x)
def Au(x,y):
	return -v(x)*ddPsi(y)
def Av(x,y):
	return u(y)*ddPsi(x)

def Yg(psi,x):
	def Yt(psi):
		return np.arccos(np.sign(psi)*2**(1/(1+2*q))*np.abs(psi)**(1/(1+2*q)))
	return Yt(psi-Psi(x))

def abc(x,y):
	xm=(x[1:]+x[:-1])/2
	ym=(y[1:]+y[:-1])/2
	dx=np.sqrt(np.diff(x)**2+np.diff(y)**2)
	u2=u(ym)**2+v(xm)**2
	dt=dx/u2**0.5
	grdPsiuGrdU=v(xm)*Au(xm,ym)-u(ym)*Av(xm,ym)
	return np.sum(u2*dt),np.sum(grdPsiuGrdU*dt),np.sum(dt)

def Comp(psi,toplot=False):
	nx=10000
	xmax=Yg(psi,0)
	x=np.linspace(0,xmax,nx)
	y=Yg(psi,x)
	if toplot:
		plt.plot(x,y)
		plt.show()
	else:
		a,b,c=abc(x,y)
		a,b,c=4*a,4*b,4*c
		return a,b,c

npsi=3000
psi=np.linspace(0,1,npsi+2)[1:-1]
fid= './data/Int_q%d.npy' %q
if os.path.isfile(fid):
	Integrals=np.load(fid)
else:
	Integrals=np.array([Comp(psi[ipsi]) for ipsi in tqdm(np.arange(npsi))]).swapaxes(0,1)
	np.save(fid,Integrals)
a=interp1d(psi[np.logical_not(np.isnan(Integrals[0,:]))],Integrals[0,np.logical_not(np.isnan(Integrals[0,:]))],fill_value='extrapolate')
b=interp1d(psi[np.logical_not(np.isnan(Integrals[1,:]))],Integrals[1,np.logical_not(np.isnan(Integrals[1,:]))],fill_value='extrapolate')
c=interp1d(psi[np.logical_not(np.isnan(Integrals[2,:]))],Integrals[2,np.logical_not(np.isnan(Integrals[2,:]))],fill_value='extrapolate')
g=lambda psi: b(psi)/a(psi)
Gt=lambda psi: quad(g,0,psi,full_output=1)[0]
fid= './data/G_q%d.npy' %q
if os.path.isfile(fid):
	G=np.load(fid)
else:
	G=np.array(jl.Parallel(n_jobs=8)(jl.delayed(Gt)(psi[ipsi]) for ipsi in tqdm(np.arange(npsi))))
	np.save(fid,G)
G=interp1d(psi,G,fill_value='extrapolate')
def Z(alpha):
	phidag=lambda psi: c(psi)*np.exp(-alpha*G(psi))
	return quad(phidag,0,1,full_output=1)[0]/np.pi**2/2
psi1=0.417
psi2=0.515
psiii=np.copy(psi)
# fig = plt.figure(figsize=[9,6])
# ax1=fig.add_subplot(111)
# # ax2=fig.add_subplot(212)
# ax1.plot(psi,b(psi),'k-')
# ax1.plot([psi1,psi1],[-1,3.5],'k--')
# ax1.plot([psi2,psi2],[-1,3.5],'k--')
# ax1.set_ylabel(r'$b(\psi)$',fontsize=24,rotation=0)
# ax1.yaxis.set_label_coords(-0.1,0.43)
# ax1.set_xticks([0,psi1,psi2,1])
# ax1.set_yticks([-1,0,1,2,3])
# ax1.plot(psi,0*psi,'k--')
# ax1.set_ylim(-1,3.5)
# # ax2.plot(psi,np.exp(-20*G(psi)),'k-')
# ax1.set_xlabel(r'$\psi$',fontsize=24,labelpad=0.9)
# ax1.tick_params(axis='both', which='both', labelsize=20)
# ax1.set_xlim(-0.002,1.001)
# ax1.set_xticklabels([0,r'$\psi_1$',r'$\psi_2$',1])
# # ax2.set_ylabel(r'$g(\psi)$',fontsize=24,rotation=0)
# plt.savefig('Bpsi.png',dpi=200)
# plt.show()

# alpha=np.linspace(-5,5,50)
# Za=np.array(jl.Parallel(n_jobs=8)(jl.delayed(Z)(alpha[ia]) for ia in tqdm(np.arange(alpha.size))))


def Overdamped_Step(x,y,dt,St,Pe,i):
	if St!=0:
		if i%2==0:
			x+=(u(y)-St*Au(x,y))*dt
			y+=(v(x)-St*Av(x,y))*dt
		else:
			y+=(v(x)-St*Av(x,y))*dt
			x+=(u(y)-St*Au(x,y))*dt
	else:
		if i%2==0:
			x+=u(y)*dt
			y+=v(x)*dt
		else:
			y+=v(x)*dt
			x+=u(y)*dt
	x+=np.sqrt(2*dt/Pe)*np.random.normal(0,1,x.size)
	y+=np.sqrt(2*dt/Pe)*np.random.normal(0,1,y.size)
	return x,y



def CompDirect(Pe,St,tmax,Toplot=False):
	alpha=Pe*St
	nreal=10**4
	dt	= 0.01
	fid='./data/x_Pe-%d' %Pe
	fid+='_St-%01.05f.npz' %St
	if os.path.isfile(fid):
		x=np.load(fid)['x']
		y=np.load(fid)['y']
		t=np.load(fid)['t']
	else:
		psi=np.linspace(0,1,nreal*100)[1:-1]
		phidag=c(psi)*np.exp(-alpha*G(psi))
		phidag/=phidag.sum()
		psi=np.random.choice(psi,nreal,p=phidag)
		x=np.zeros(nreal)
		y=Yg(psi,0)
		t=0
	if Toplot:
		X=x/2/np.pi
		Y=y/2/np.pi
		plt.ion()
		fig=plt.figure()
		ax=fig.add_subplot(111)
		PlotB,=ax.plot(X,Y,'ro',Markersize=0.1,alpha=0.5)
		fig.canvas.draw()
		fig.canvas.flush_events()
	it=0
	while t<tmax:
		it+=1
		t+=dt
		x,y=Overdamped_Step(x,y,dt,St,Pe,it)
		if it%10==0:
			np.savez(fid,x=x,y=y,t=t)
			if Toplot:
				X=x/2/np.pi
				Y=y/2/np.pi
				PlotB.set_xdata(X)
				PlotB.set_ydata(Y)
				xb=np.floor(np.maximum(np.abs(X).max()+1,np.abs(Y).max())+1)
				ax.set_xlim(-xb,xb)
				ax.set_ylim(-xb,xb)
				fig.canvas.draw()
				fig.canvas.flush_events()
	np.savez(fid,x=x,y=y,t=t)
	return 


Pe=np.array([10**3,10**4,10**5])
alpha=np.array([-10,0,10,20,60])
alphab=np.linspace(-10,60,100)
Za=np.array(jl.Parallel(n_jobs=8)(jl.delayed(Z)(alphab[ia]) for ia in tqdm(np.arange(alphab.size))))
nPe=Pe.size
na=alpha.size

tmax=10**3

jl.Parallel(n_jobs=7)(jl.delayed(CompDirect)(Pe[iPe],alpha[ia]/Pe[iPe],tmax) for ia in tqdm(np.arange(na)) for iPe in np.arange(nPe))


def Kappa(Pe,St):
	fid='./data/x_Pe-%d' %Pe
	fid+='_St-%01.05f.npz' %St
	if os.path.isfile(fid):
		return np.mean(np.load(fid)['x']**2+np.load(fid)['y']**2)/np.load(fid)['t']/2
	else:
		return 0 

k=np.array([[Kappa(Pe[iPe],alpha[ia]/Pe[iPe]) for ia in np.arange(na)] for iPe in tqdm(np.arange(nPe))])



fig = plt.figure(figsize=[13,11])
gs = fig.add_gridspec(2, 10)
axes=[]
axes.append(fig.add_subplot(gs[0, 0:4]))
axes.append(fig.add_subplot(gs[1,:]))
axes.append(fig.add_subplot(gs[0, 5:]))

axes[0].tick_params(axis='both', which='both', labelsize=24)
x=np.linspace(-np.pi,np.pi,100)
xx,yy=np.meshgrid(x,x,indexing='ij')
psi=3/4*np.sin(xx)*np.sin(yy)+np.sin(3*xx)*np.sin(3*yy)/4
OO=axes[0].pcolor(xx,yy,psi,cmap=cmocean.cm.balance,vmin=-1,vmax=1)
X=np.linspace(0,Yg(psi2,0),1000)
Y=Yg(psi2,X)
X,Y=(X-Y)/2,(X+Y)/2
axes[0].plot(X[:]+np.pi/2,Y[:]+np.pi/2,'k-',linewidth=1)
#axes[0].text(X[50]+np.pi/2,Y[0]+np.pi/2-0.05,r'$\psi_2$',fontsize=10)
axes[0].plot(X[:]+np.pi/2,Y[:]-np.pi/2,'k-',linewidth=1)
#axes[0].text(X[0]+np.pi/2,Y[0]-np.pi/2-0.05,r'-$\psi_2$',fontsize=10)
axes[0].plot(X[:]-np.pi/2,Y[:]+np.pi/2,'k-',linewidth=1)
#axes[0].text(X[0]-np.pi/2,Y[0]+np.pi/2-0.05,r'-$\psi_2$',fontsize=10)
axes[0].plot(X[:]-np.pi/2,Y[:]-np.pi/2,'k-',linewidth=1)
#axes[0].text(X[50]-np.pi/2,Y[0]-np.pi/2-0.05,r'$\psi_2$',fontsize=10)
axes[0].plot(X+np.pi/2,-Y+np.pi/2,'k-',linewidth=1)
axes[0].plot(X+np.pi/2,-Y-np.pi/2,'k-',linewidth=1)
axes[0].plot(X-np.pi/2,-Y+np.pi/2,'k-',linewidth=1)
axes[0].plot(X-np.pi/2,-Y-np.pi/2,'k-',linewidth=1)
axes[0].plot(Y+np.pi/2,X+np.pi/2,'k-',linewidth=1)
axes[0].plot(Y+np.pi/2,X-np.pi/2,'k-',linewidth=1)
axes[0].plot(Y-np.pi/2,X+np.pi/2,'k-',linewidth=1)
axes[0].plot(Y-np.pi/2,X-np.pi/2,'k-',linewidth=1)
axes[0].plot(-Y+np.pi/2,X+np.pi/2,'k-',linewidth=1)
axes[0].plot(-Y+np.pi/2,X-np.pi/2,'k-',linewidth=1)
axes[0].plot(-Y-np.pi/2,X+np.pi/2,'k-',linewidth=1)
axes[0].plot(-Y-np.pi/2,X-np.pi/2,'k-',linewidth=1)
X=np.linspace(0,Yg(psi1,0),1000)
Y=Yg(psi1,X)
X,Y=(X-Y)/2,(X+Y)/2
axes[0].plot(X[:]+np.pi/2,Y[:]+np.pi/2,'k-',linewidth=1)
#axes[0].text(X[30]+np.pi/2,Y[0]+np.pi/2-0.05,r'$\psi_1$',fontsize=10)
axes[0].plot(X[:]+np.pi/2,Y[:]-np.pi/2,'k-',linewidth=1)
#axes[0].text(X[0]+np.pi/2,Y[0]-np.pi/2-0.05,r'-$\psi_1$',fontsize=10)
axes[0].plot(X[:]-np.pi/2,Y[:]+np.pi/2,'k-',linewidth=1)
#axes[0].text(X[0]-np.pi/2,Y[0]+np.pi/2-0.05,r'-$\psi_1$',fontsize=10)
axes[0].plot(X[:]-np.pi/2,Y[:]-np.pi/2,'k-',linewidth=1)
#axes[0].text(X[30]-np.pi/2,Y[0]-np.pi/2-0.05,r'$\psi_1$',fontsize=10)
axes[0].plot(X+np.pi/2,-Y+np.pi/2,'k-',linewidth=1)
axes[0].plot(X+np.pi/2,-Y-np.pi/2,'k-',linewidth=1)
axes[0].plot(X-np.pi/2,-Y+np.pi/2,'k-',linewidth=1)
axes[0].plot(X-np.pi/2,-Y-np.pi/2,'k-',linewidth=1)
axes[0].plot(Y+np.pi/2,X+np.pi/2,'k-',linewidth=1)
axes[0].plot(Y+np.pi/2,X-np.pi/2,'k-',linewidth=1)
axes[0].plot(Y-np.pi/2,X+np.pi/2,'k-',linewidth=1)
axes[0].plot(Y-np.pi/2,X-np.pi/2,'k-',linewidth=1)
axes[0].plot(-Y+np.pi/2,X+np.pi/2,'k-',linewidth=1)
axes[0].plot(-Y+np.pi/2,X-np.pi/2,'k-',linewidth=1)
axes[0].plot(-Y-np.pi/2,X+np.pi/2,'k-',linewidth=1)
axes[0].plot(-Y-np.pi/2,X-np.pi/2,'k-',linewidth=1)
axes[0].plot([-np.pi,np.pi],[0,0],'k-',linewidth=1)
axes[0].plot([0,0],[-np.pi,np.pi],'k-',linewidth=1)
# x=np.linspace(-np.pi,np.pi,27)
# xx,yy=np.meshgrid(x,x,indexing='ij')
# u= -3/4*np.sin(xx)*np.cos(yy)-3*np.sin(3*xx)*np.cos(3*yy)/4
# v= 3/4*np.sin(yy)*np.cos(xx)+3*np.sin(3*yy)*np.cos(3*xx)/4
# axes[0].quiver(xx,yy,u,v)
axes[0].set_aspect('equal')
axes[0].set_xticks([-np.pi,0,np.pi])
axes[0].set_yticks([-np.pi,0,np.pi])
axes[0].set_xticklabels([r'-$\pi$',r'$0$',r'$\pi$'])
axes[0].set_yticklabels([r'-$\pi$',r'$0$',r'$\pi$'])
axes[0].set_xlabel(r'$x$',fontsize=24,labelpad=0.9)
axes[0].set_ylabel(r'$y$',fontsize=24,rotation=0)
axes[0].yaxis.set_label_coords(-0.13,0.472)

axins=inset_axes(axes[0],width="80%",
					height="2%",
					loc=3,
					bbox_to_anchor=(0.1, 1.01, 1, 1),
					bbox_transform=axes[0].transAxes,
					borderpad=0)
c=fig.colorbar(OO,cax=axins,ticks=[-1,1],orientation='horizontal')
axins.tick_params(axis='both', which='both', labelsize=20)
axins.xaxis.set_ticks_position('top')
c.set_ticklabels([r'-$1$',r'$1$'])
c.set_label(r'$\psi$',fontsize=24)
c.ax.xaxis.set_label_coords(0.5,5)


axes[1].tick_params(axis='both', which='both', labelsize=24)
axes[1].plot(alphab,1/Za,'k-',label=r'$1/Z(\alpha)$')
axes[1].plot(alpha,k[0]/k[0,1],'ro',markersize=10,label=r'$\mathrm{Pe}=10^3$')
axes[1].plot(alpha,k[1]/k[1,1],'bv',markersize=10,label=r'$\mathrm{Pe}=10^4$')
axes[1].plot(alpha,k[2]/k[2,1],'*',color='brown',markersize=10,label=r'$\mathrm{Pe}=10^5$')
axes[1].set_xlabel(r'$\alpha$',fontsize=24,labelpad=0.9)
axes[1].set_ylabel(r'$\frac{\overline{D}(\mathrm{Pe},\alpha)}{\overline{D}(\mathrm{Pe},0)}$',fontsize=32,rotation=0)
axes[1].yaxis.set_label_coords(-0.1,0.42)
axes[1].legend(fontsize=22)

# ax2=fig.add_subplot(212)
axes[2].plot(psiii,b(psiii),'k-')
axes[2].plot([psi1,psi1],[-1,3.5],'k--')
axes[2].plot([psi2,psi2],[-1,3.5],'k--')
axes[2].set_ylabel(r'$b(\psi)$',fontsize=24,rotation=0)
axes[2].yaxis.set_label_coords(-0.13,0.46)
axes[2].set_xticks([0,psi1,psi2,1])
axes[2].set_yticks([-1,0,1,2,3])
axes[2].plot(psiii,0*psiii,'k--')
axes[2].set_ylim(-1,3.5)
# ax2.plot(psi,np.exp(-20*G(psi)),'k-')
axes[2].set_xlabel(r'$\psi$',fontsize=24,labelpad=0.9)
axes[2].tick_params(axis='both', which='both', labelsize=20)
axes[2].set_xlim(-0.002,1.001)
axes[2].set_xticklabels([0,r'$\psi_1$',r'$\psi_2$',1])
# ax2.set_ylabel(r'$g(\psi)$',fontsize=24,rotation=0)


plt.savefig('SecondFlow.png',dpi=400)


#plt.show()
plt.close('all')