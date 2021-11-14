import numpy as np
import multiprocessing as mp
from numpy import newaxis as nx
import matplotlib.pyplot as plt
from scipy.special import ellipk,ellipe
from scipy.interpolate import interp1d
from tqdm import tqdm
import joblib as jl
import os
from matplotlib import rc
import matplotlib.colors
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","deepskyblue","royalblue","navy","black"])
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import cmocean
#cmap2=cmocean.cm.balance
cmap2 = 'RdGy_r'# matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","deepskyblue","royalblue","navy","black"])
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{bm}')
def G(psi):
	def g(x):
		return x*(ellipk(1-x**2)-ellipe(1-x**2))/(ellipe(1-x**2)-x**2*ellipk(1-x**2))
	x=np.linspace(0,psi,10000+np.int_(1000*psi))[1:-1]
	dx=x[0]
	return np.sum(g(x))*dx


def Z(a):
	def G(x):
		def g(x):
			gg=np.zeros((x.shape))
			gg[x==1]=1
			y=x[(x>0)&(x<1)]
			gg[(x>0)&(x<1)]=y*(ellipk(1-y**2)-ellipe(1-y**2))/(ellipe(1-y**2)-y**2*ellipk(1-y**2))
			gg[x==1]=1
			return gg
		nx=x.size
		n=100
		xx,yy=np.meshgrid(x,np.linspace(0,1,n,endpoint=False),indexing='ij')
		gg=np.sum(g(xx*yy)*xx/n,axis=-1)
		return gg
	n=100+np.int_(5*np.abs(a))
	x=np.linspace(0,1,n+1)[1:]
	dx=x[0]
	Gt=G(x)
	return (np.sum(ellipk(1-x**2)*np.exp(-Gt*a))*dx)*4/np.pi**2



alpham=10
alphap=100
Psi=np.linspace(0,1,10000)[1:]
Gt=np.hstack((np.array([0]),np.array([G(Psi[i]) for i in tqdm(np.arange(Psi.size))])))
Psi=np.hstack((np.array([0]),Psi))
Gt=interp1d(Psi,Gt)

n=1000
x=np.linspace(0,np.pi,n+1)
x=(x[1:]+x[:-1])/2
dx=x[2]-x[1]
xx,yy=np.meshgrid(x,x,indexing='ij')
Psi=np.sin(xx)*np.sin(yy)
Da=-(np.cos(2*xx)+np.cos(2*yy))/4
Phidag=np.exp(-alphap*Gt(Psi))
Phidagb=np.exp(alpham*Gt(Psi))
Phidag/=Phidag.sum()*dx
Phidagb/=Phidagb.sum()*dx
Phi0=np.ones((n,n))
Phi0/=Phi0.sum()*dx

psi1=0.25
x1=np.linspace(np.arcsin(np.sqrt(psi1)),np.pi-np.arcsin(np.sqrt(psi1)),1000)
y1=np.arcsin(psi1/np.sin(x1))
psi2=0.75
x2=np.linspace(np.arcsin(np.sqrt(psi2)),np.pi-np.arcsin(np.sqrt(psi2)),1000)
y2=np.arcsin(psi2/np.sin(x2))


fig = plt.figure(figsize=[10,6])
gs = fig.add_gridspec(1, 17)
axes=[]
axes.append(fig.add_subplot(gs[0, 5:9]))
axes.append(fig.add_subplot(gs[0, 9:13]))
axes.append(fig.add_subplot(gs[0, 13:]))
axes.append(fig.add_subplot(gs[0, :4]))
# fig, axes = plt.subplots(1, 4,figsize=[12,6])
axes[0].tick_params(axis='both', which='both', labelsize=22)
axes[1].tick_params(axis='both', which='both', labelsize=22)
axes[2].tick_params(axis='both', which='both', labelsize=22)
axes[3].tick_params(axis='both', which='both', labelsize=22)
v=axes[0].imshow(Phidagb,cmap=cmap,extent=[0,1,0,1],vmin=0,vmax=0.005)
axes[1].imshow(Phi0,cmap=cmap,extent=[0,1,0,1],vmin=0,vmax=0.005)
axes[2].imshow(Phidag,cmap=cmap,extent=[0,1,0,1],vmin=0,vmax=0.005)
v2=axes[3].imshow(Da,cmap=cmap2,extent=[0,1,0,1],vmin=-1/2,vmax=1/2)
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
axes[2].set_aspect('equal')
axes[3].set_aspect('equal')
axes[0].plot(x1/np.pi,y1/np.pi,'k-',Linewidth=0.5)
axes[0].plot(x1/np.pi,1-y1/np.pi,'k-',Linewidth=0.5)
axes[0].plot(y1/np.pi,x1/np.pi,'k-',Linewidth=0.5)
axes[0].plot(1-y1/np.pi,x1/np.pi,'k-',Linewidth=0.5)
axes[0].plot(x2/np.pi,y2/np.pi,'k-',Linewidth=0.5)
axes[0].plot(x2/np.pi,1-y2/np.pi,'k-',Linewidth=0.5)
axes[0].plot(y2/np.pi,x2/np.pi,'k-',Linewidth=0.5)
axes[0].plot(1-y2/np.pi,x2/np.pi,'k-',Linewidth=0.5)
axes[1].plot(x1/np.pi,y1/np.pi,'k-',Linewidth=0.5)
axes[1].plot(x1/np.pi,1-y1/np.pi,'k-',Linewidth=0.5)
axes[1].plot(y1/np.pi,x1/np.pi,'k-',Linewidth=0.5)
axes[1].plot(1-y1/np.pi,x1/np.pi,'k-',Linewidth=0.5)
axes[1].plot(x2/np.pi,y2/np.pi,'k-',Linewidth=0.5)
axes[1].plot(x2/np.pi,1-y2/np.pi,'k-',Linewidth=0.5)
axes[1].plot(y2/np.pi,x2/np.pi,'k-',Linewidth=0.5)
axes[1].plot(1-y2/np.pi,x2/np.pi,'k-',Linewidth=0.5)
axes[2].plot(x1/np.pi,y1/np.pi,'k-',Linewidth=0.5)
axes[2].plot(x1/np.pi,1-y1/np.pi,'k-',Linewidth=0.5)
axes[2].plot(y1/np.pi,x1/np.pi,'k-',Linewidth=0.5)
axes[2].plot(1-y1/np.pi,x1/np.pi,'k-',Linewidth=0.5)
axes[2].plot(x2/np.pi,y2/np.pi,'k-',Linewidth=0.5)
axes[2].plot(x2/np.pi,1-y2/np.pi,'k-',Linewidth=0.5)
axes[2].plot(y2/np.pi,x2/np.pi,'k-',Linewidth=0.5)
axes[2].plot(1-y2/np.pi,x2/np.pi,'k-',Linewidth=0.5)
axes[3].plot(x1/np.pi,y1/np.pi,'k-',Linewidth=0.5)
axes[3].plot(x1/np.pi,1-y1/np.pi,'k-',Linewidth=0.5)
axes[3].plot(y1/np.pi,x1/np.pi,'k-',Linewidth=0.5)
axes[3].plot(1-y1/np.pi,x1/np.pi,'k-',Linewidth=0.5)
axes[3].plot(x2/np.pi,y2/np.pi,'k-',Linewidth=0.5)
axes[3].plot(x2/np.pi,1-y2/np.pi,'k-',Linewidth=0.5)
axes[3].plot(y2/np.pi,x2/np.pi,'k-',Linewidth=0.5)
axes[3].plot(1-y2/np.pi,x2/np.pi,'k-',Linewidth=0.5)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[3].set_xticks([])
axes[3].set_yticks([])

axins=inset_axes(	axes[1],
					width="150%",
					height="5%",
					loc=3,
					bbox_to_anchor=(-0.25, -0.1, 1, 1),
					bbox_transform=axes[1].transAxes,
					borderpad=0)
axins2=inset_axes(	axes[3],
					width="100%",
					height="5%",
					loc=3,
					bbox_to_anchor=(0, -0.1, 1, 1),
					bbox_transform=axes[3].transAxes,
					borderpad=0)
c=fig.colorbar(		v,
					cax=axins,ticks=[0,0.005], orientation='horizontal')
c2=fig.colorbar(	v2,
					cax=axins2,ticks=[-0.5,0,0.5], orientation='horizontal')
# axins.xaxis.set_ticks_position('top')
# axins.xaxis.set_label_position('top')
axins.tick_params(axis='both', which='both', labelsize=17)
axins2.tick_params(axis='both', which='both', labelsize=17)
c.set_ticklabels([r'$0$',r'$5\times 10^{-3}$'])
c.set_label(r'$\phi^{\dagger}$',fontsize=22,labelpad=-3)
c2.set_ticklabels([r'-$0.5$',r'$0$',r'$0.5$'])
c2.set_label(r'$\Phi$',fontsize=22,labelpad=2)
axes[0].set_title(r'$\alpha=-10$',fontsize=22)
axes[1].set_title(r'$\alpha=0$',fontsize=22)
axes[2].set_title(r'$\alpha=100$',fontsize=22)	
#axes[3].set_title(r'$\nabla\bm{u}_{\rm e}$',fontsize=22)	
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2,wspace=0.05)
plt.savefig('phidag.png',dpi=200)
plt.show()
