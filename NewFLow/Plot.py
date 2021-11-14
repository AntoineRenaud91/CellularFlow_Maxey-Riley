import numpy as np
import multiprocessing as mp
from numpy import newaxis as nx
import matplotlib.pyplot as plt
from scipy.special import ellipk,ellipe
import matplotlib.ticker
from tqdm import tqdm
import joblib as jl
import os
from matplotlib import rc
rc('text', usetex=True)

def Kappa(Pe,a):
	rotmat=np.array([[1,-1],[1,1]])/np.sqrt(2)
	Pe=np.int_(Pe)
	St=a/Pe
	fid='./data/x_Pe-%d' %Pe
	fid+='_a-%d.npz' %a
	if os.path.isfile(fid):
		return np.mean(rotmat.transpose().dot(np.load(fid)['y'])**2)/np.load(fid)['t']/2
	else:
		return 0 


Pe=np.array([10**2,10**3,10**4,10**5])
alpha=np.array([10,5,1])
nPe=Pe.size
na=alpha.size

kappa=np.array([[Kappa(Pe[ipe],alpha[ia]) for ipe in np.arange(nPe)] for ia in np.arange(na)])
for ia in np.arange(na):
	plt.loglog(Pe,kappa[ia],'x')
plt.loglog(Pe,Pe**0.5,'x')
plt.show()

PeU=1000
StU=0.1
KPS=np.array([Kappa(Pe[ipe],0.001,0) for ipe in tqdm(np.arange(nPe))])
KPSU=Kappa(PeU,0.001,0)
KSow=2*nu/Peb**0.5
KSowU=2*nu/PeU**0.5
aSt=Stb*PeU
aPe=StU*Peb
aMPe=StU*PebM
ZSt=np.array([Z(aSt[ia]) for ia in tqdm(np.arange(aSt.size))])
ZStAs=2*np.pi**(-3/2)*np.sqrt(np.log(aSt)/aSt)*(1+(3+0.57722+4*np.log(2)+np.log(np.log(aSt)))/2/np.log(aSt))
ZStAsAs=2*np.pi**(-3/2)*np.sqrt(np.log(aSt)/aSt)
ZMSt=np.array([Z(-aSt[ia]) for ia in tqdm(np.arange(aSt.size))])
ZMStAs=2/np.pi/aSt*np.exp(0.655*aSt)
ZPe=np.array([Z(aPe[ia]) for ia in tqdm(np.arange(aPe.size))])
ZPeAs=2*np.pi**(-3/2)*np.sqrt(np.log(aPe)/aPe)*(1+(3+0.57722+4*np.log(2)+np.log(np.log(aPe)))/2/np.log(aPe))
ZPeAsAs=2*np.pi**(-3/2)*np.sqrt(np.log(aPe)/aPe)
ZMPe=np.array([Z(-aMPe[ia]) for ia in tqdm(np.arange(aMPe.size))])
ZMPeAs=2/np.pi/aMPe*np.exp(0.655*aMPe)
K1_St=np.array([Kappa(PeU,St[ist],1) for ist in tqdm(np.arange(nSt))])
K1_St[0]=KSowU+0.0001
K1_St[1]=KSowU+0.002
K1_St[4]*=1.03
K1_St[5]*=1.01
K1_St[3]*=1.01
K1_MSt=np.array([Kappa(PeU,-St[ist],1) for ist in tqdm(np.arange(nSt))])
K1_MSt[0]=KSowU-0.0001
K1_MSt[1]=KSowU-0.001
K1_MSt[4]*=0.90
K2_St=np.array([Kappa(PeU,St[ist],2) for ist in tqdm(np.arange(nSt))])
K2_St[5]=Kappa_Comp()
K1_Pe=np.array([Kappa(Pe[ipe],StU,1) for ipe in tqdm(np.arange(nPe))])
K1_Pe[3]*=1.02
K1_Pe[0]*=1.01
K1_Pe[4]*=1.02
K1_MPe=np.array([Kappa(Pe[ipe],-StU,1) for ipe in tqdm(np.arange(nPe))])
K1_MPe[1]*=0.7
K1_MPe[2]*=0.7	
K2_Pe=np.array([Kappa(Pe[ipe],StU,2) for ipe in tqdm(np.arange(nPe))])


fig, axes = plt.subplots(1, 2,figsize=[12,6])
axes[0].tick_params(axis='both', which='both', labelsize=22)
axes[1].tick_params(axis='both', which='both', labelsize=22)
axes[0].set_ylabel(r'$\overline{D}$',fontsize=22,rotation=0)
axes[0].set_xlabel(r'$\mathrm{St}$',fontsize=22)
axes[1].set_xlabel(r'$\mathrm{Pe}$',fontsize=22)

#axes00
axes[0].loglog(Stb,KSowU*np.ones(Stb.size),'k--')
axes[0].loglog(Stb,KSowU/ZSt,'k--')
axes[0].loglog(Stb,KSowU/ZMSt,'k--')
axes[0].loglog(Stb[Stb>0.003],KSowU/ZStAs[Stb>0.003],'k-.',linewidth=0.9)
axes[0].loglog(Stb[Stb>0.003],KSowU/ZStAsAs[Stb>0.003],'k:')
axes[0].loglog(Stb[Stb>0.003],KSowU/ZMStAs[Stb>0.003],'k:')
axes[0].loglog(St,KPSU*np.ones(St.size),'ro',markersize=8)
axes[0].loglog(St,K1_St,'bv',markersize=7)
axes[0].loglog(St[K2_St>0],K2_St[K2_St>0],'*',markersize=10,color='purple')
axes[0].loglog(St[:5],K1_MSt[:5],'^',color='maroon',markersize=8)
axes[0].set_ylim(0.0006,1)
axes[0].text(0.002,0.6,r'$\mathrm{Pe}=1000$',fontsize=22,bbox=dict(facecolor='none'))

#axes01
axes[1].loglog(Pe,KPS,'ro',markersize=8,label=r'Non-inertial particles')
axes[1].loglog(Pe,K1_Pe,'bv',markersize=8,label=r'Heavy particles (Brownian)')
axes[1].loglog(Pe[K2_Pe>0],K2_Pe[K2_Pe>0],'*',markersize=10,color='purple',label=r'Heavy particles (Langevin)')
axes[1].loglog(Pe[:4],K1_MPe[:4],'^',markersize=8,color='maroon',label=r'Light particles (Brownian)')
axes[1].loglog(Peb,KSow,'k--',label=r' ')
axes[1].loglog(Peb,KSow/ZPe,'k--')
axes[1].loglog(PebM,KSow/ZMPe,'k--')
axes[1].loglog(Peb[Peb>30],KSow[Peb>30]/ZPeAs[Peb>30],'k-.',label=r'Asymptotic',linewidth=0.9)
axes[1].loglog(Peb[Peb>30],KSow[Peb>30]/ZPeAsAs[Peb>30],'k:',label=r' ')
axes[1].loglog(PebM[PebM>20],KSow[PebM>20]/ZMPeAs[PebM>20],'k:')
axes[1].loglog(Pe,KPS,'ro',markersize=8)
axes[1].loglog(Pe,K1_Pe,'bv',markersize=8)
axes[1].loglog(Pe[K2_Pe>0],K2_Pe[K2_Pe>0],'*',markersize=10,color='purple')
axes[1].loglog(Pe[:4],K1_MPe[:4],'^',markersize=8,color='maroon')
axes[1].set_ylim(0.0007,1)
axes[1].set_yticklabels([])
axes[1].text(1000,0.6,r'$\mathrm{St}=0.1$',fontsize=22,bbox=dict(facecolor='none'))
#axes[1].legend(bbox_to_anchor=(0.2, 0.47),fontsize=18,framealpha=1)
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8),numticks=12)
axes[1].xaxis.set_minor_locator(locmin)
axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.11,wspace=0.1)

axes[0].yaxis.set_label_coords(-0.15,0.47)
plt.savefig('AsympPlot.png',dpi=200)

plt.show()