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
def Z_assymp_p(a):		
	gamma=0.577
	return 2/np.pi**(3/2)*np.sqrt(np.log(a)/a)*(1+(3+gamma+4*np.log(2)+np.log(np.log(a)))/2/np.log(a))
def Z_assymp_pp(a):		
	gamma=0.577
	return 2/np.pi**(3/2)*np.sqrt(np.log(a)/a)
def Z_assymp_m(a):		
	upsilon=0.655
	return 2/np.pi/np.abs(a)*np.exp(upsilon*np.abs(a))

na=2000
a=np.hstack((-10**(np.linspace(1.2,-1,na)),10**(np.linspace(-1,4.2,na))))
am=-10**np.linspace(1.2,0.5,na)
ap=10**np.linspace(1,4.2,na)
Za=np.array([Z(a[ia]) for ia in tqdm(np.arange(2*na))])
Zap=Z_assymp_p(ap)
Zapp=Z_assymp_pp(ap)
Zam=Z_assymp_m(am)



fig, axes = plt.subplots(1, 1,figsize=[12,6])
axes.tick_params(axis='both', which='both', labelsize=22)
axes.set_ylabel(r'$Z(\alpha)$',fontsize=22,rotation=0,)
axes.set_xlabel(r'$\alpha$',fontsize=22)
axes.semilogy(a,Za,label=r'$ {Z(\alpha)}$')
axes.semilogy(ap,Zapp,'k:',label=r'$\frac{2\sqrt{\log\alpha}}{\pi^{3/2}\sqrt{\alpha}}$')
axes.semilogy(ap,Zap,'k-.',label=r'$\frac{2\sqrt{\log\alpha}}{\pi^{3/2}\sqrt{\alpha}}\left(1+\frac{3+\gamma_e+\log(16\log\alpha)}{2\log\alpha}\right)$')
axes.semilogy(am,Zam,'k--',label=r'$\frac{2}{\pi|\alpha|}\mathrm{e}^{\Upsilon|\alpha|}$')
axes.semilogy([a.min(),a.max()],[1,1],'k-',Linewidth=0.5)
axes.semilogy([0,0],[0.01,100],'k-',Linewidth=0.5)
axes.set_xscale('symlog')
axes.set_xlim(a.min(),a.max())
axes.set_ylim(0.01,100)
axes.set_xticks([-10,-1,0,1,10,100,1000,10000])
axes.set_yticks([0.01,0.1,1,10,100])
axes.set_xticklabels([r'-$10$',r'-$1$',r'$0$',r'$1$',r'$10$',r'$10^2$',r'$10^3$',r'$10^4$'])
axes.set_yticklabels([r'$10^{\mathrm{-}2}$',r'$10^{\mathrm{-}1}$',r'$1$',r'$10$',r'$10^2$'])
axes.yaxis.set_label_coords(-0.1,0.46)
axes.legend(fontsize=20,bbox_to_anchor=(0.541,0.477))
plt.savefig('Za.png',dpi=200)
plt.show()

# #axes00
# axes[0].loglog(Stb,KSowU*np.ones(Stb.size),'k--')
# axes[0].loglog(Stb,KSowU/ZSt,'k--')
# axes[0].loglog(Stb,KSowU/ZMSt,'k--')
# axes[0].loglog(St,KPSU*np.ones(St.size),'ro',markersize=8)
# axes[0].loglog(St,K1_St,'bv',markersize=7)
# axes[0].loglog(St[K2_St>0],K2_St[K2_St>0],'*',markersize=10,color='purple')
# axes[0].loglog(St[:5],K1_MSt[:5],'^',color='maroon',markersize=8)
# axes[0].set_ylim(0.0006,1)
# axes[0].text(0.002,0.6,r'$\mathrm{Pe}=1000$',fontsize=22,bbox=dict(facecolor='none'))

# #axes01
# axes[1].loglog(Pe,KPS,'ro',markersize=8,label=r'Non-inertial particles')
# axes[1].loglog(Pe,K1_Pe,'bv',markersize=8,label=r'Heavy particles (overdamped)')
# axes[1].loglog(Pe[K2_Pe>0],K2_Pe[K2_Pe>0],'*',markersize=10,color='purple',label=r'Heavy particles (Langevin)')
# axes[1].loglog(Pe[:4],K1_MPe[:4],'^',markersize=8,color='maroon',label=r'Light particles (overdamped)')
# axes[1].loglog(Peb,KSow,'k--',label=r'Asymptotic')
# axes[1].loglog(Peb,KSow/ZPe,'k--')
# axes[1].loglog(Peb,KSow/ZMPe,'k--')
# axes[1].loglog(Pe,KPS,'ro',markersize=8)
# axes[1].loglog(Pe,K1_Pe,'bv',markersize=8)
# axes[1].loglog(Pe[K2_Pe>0],K2_Pe[K2_Pe>0],'*',markersize=10,color='purple')
# axes[1].loglog(Pe[:4],K1_MPe[:4],'^',markersize=8,color='maroon')
# axes[1].set_ylim(0.0006,1)
# axes[1].set_yticklabels([])
# axes[1].text(1000,0.6,r'$\mathrm{St}=0.1$',fontsize=22,bbox=dict(facecolor='none'))
# axes[1].legend(bbox_to_anchor=(0.27, 0.47),fontsize=18)
# plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.11,wspace=0.05)
# plt.savefig('AsympPlot.png',dpi=200)

# plt.show()