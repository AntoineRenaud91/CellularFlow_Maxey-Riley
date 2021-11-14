import numpy as np
from numpy import newaxis as nx
from scipy.linalg import det
import joblib as jl
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from tqdm import tqdm
import os
from scipy.special import ellipk


#Grid
n=6000
x=np.linspace(0,2*np.pi,n,endpoint=False)
dx=x[1]-x[0]
xx,yy=np.meshgrid(x,x,indexing='ij')
k=np.squeeze(np.array([np.meshgrid(np.fft.fftfreq(n,dx),np.fft.rfftfreq(n,dx),indexing='ij')]))
k*=2*np.pi

psi=np.sin(xx)*np.sin(yy)


#Errortmax
errmax1=2*10**(-2)
errmax2=10**(-1)

#Flow
u=np.zeros((2,n,n))
acc=np.zeros((2,n,n))
u[0]=np.sin(xx)*np.cos(yy)
acc[0]=np.sin(2*xx)/2
u[1]=-np.cos(xx)*np.sin(yy)
acc[1]=np.sin(2*yy)/2


def Kappa(ieps):
	U=u-eps[ieps]*acc
	fid='data_inertia_%d.npz' %ieps
	phidag=np.load(fid)['phidag']
	phidag/=phidag.mean()
	c=np.mean(np.mean(U*phidag[nx,:,:],axis=-1),axis=-1)
	phi=np.load(fid)['phi']
	kappa1=nu*eps[ieps]**2*np.mean(np.mean(phidag[nx,nx,:,:]*(np.fft.irfft2(1.J*k[:,nx,:,:]*np.fft.rfft2(phi[nx,:,:,:]))+np.fft.irfft2(1.J*k[nx,:,:,:]*np.fft.rfft2(phi[:,nx,:,:]))),axis=-1),axis=-1)
	kappa2=np.mean(np.mean(phidag[nx,nx,:,:]*(U[:,nx,:,:]-c[:,nx,nx,nx])*phi[nx,:,:,:],axis=-1),axis=-1)	
	kappa3=nu*eps[ieps]**2*np.mean(np.mean(phidag[nx,nx,:,:]*np.sum(np.fft.irfft2(1.J*k[nx,nx,:,:,:]*np.fft.rfft2(phi[:,nx,nx,:,:]))*np.fft.irfft2(1.J*k[nx,nx,:,:,:]*np.fft.rfft2(phi[nx,:,nx,:,:])),axis=2),axis=-1),axis=-1)
	return kappa1,kappa2,kappa3

neps=10
eps=10**(np.linspace(-2,0,neps)[::-1])
nu=1
kap1,kap2=np.zeros(neps),np.zeros(neps)
for ieps in tqdm(np.arange(neps)):
	kappa1,kappa2,kappa3=Kappa(ieps)
	kap1[ieps]=np.linalg.det(kappa1+kappa2+nu*eps[ieps]**2*np.identity(2))**0.5
	kap2[ieps]=np.linalg.det(kappa1+kappa3+nu*eps[ieps]**2*np.identity(2))**0.5

def ff(eps):
	n=10000
	psi=np.linspace(0,1,n+1)[1:]
	dpsi=psi[0]
	tosum=np.exp(-np.sqrt(2/eps[:,nx])*psi[nx,:])*ellipk(np.sqrt(1-psi[nx,:]**2))
	return np.sum(tosum,axis=-1)*dpsi*4/np.pi**2

kappa=2*nu*eps
kap_init=kappa/ff(eps)
gamma=0.57721
kap_init_2=np.pi*np.pi*np.sqrt(2*eps)/(2*gamma+5*np.log(2)-np.log(eps))*nu



def ComputeKappa(nu,eps,phidag=None,phi=None,Inertia=True,nt=None):
	if Inertia:
		U=u-eps*acc
	else:
		U=u
	dt=np.min([dx**2/nu/eps**2,dx])/40
	if phidag is None:
		phidag=np.ones((n,n))
	if phi is None:
		phi=np.ones((2,n,n))
	# plt.ion()
	# fig=plt.figure()
	# ax=fig.add_subplot(111)
	# im=ax.imshow(phidag,vmin=0,vmax=10)
	# plt.colorbar(im)
	# fig.canvas.draw()
	# fig.canvas.flush_events()
	i=1
	err=1
	RHS=np.ones((n,n))
	if nt is None:
		try: 
			while True:
				RHS=np.fft.irfft2(-nu*eps**2*np.sum(k**2,axis=0)*np.fft.rfft2(phidag)-np.sum(1.J*k*np.fft.rfft2(U*phidag[nx,:,:]),axis=0))
				err=np.max(np.abs(RHS))
				phidag+=RHS*dt
				print(err, end="\r")
				# i+=1
				# if i%100==0:
				# 	im.set_array(phidag)
				# 	im.set_clim(phidag.min(),phidag.max())
				# 	fig.canvas.draw()
				# 	fig.canvas.flush_events()
		except KeyboardInterrupt:
			pass
	else:
		for it in tqdm(np.arange(nt)):
			RHS=np.fft.irfft2(-nu*eps**2*np.sum(k**2,axis=0)*np.fft.rfft2(phidag)-np.sum(1.J*k*np.fft.rfft2(U*phidag[nx,:,:]),axis=0))
			err=np.max(np.abs(RHS))
			phidag+=RHS*dt
			i+=1
			# if i%100==0:
			# 	im.set_array(phidag)
			# 	im.set_clim(phidag.min(),phidag.max())
			# 	fig.canvas.draw()
			# 	fig.canvas.flush_events()
	c=np.mean(np.mean(U*phidag[nx,:,:],axis=-1),axis=-1)
	RHS=np.ones((2,n,n))
	err=1
	# plt.close('all')
	# plt.ion()
	# fig=plt.figure()
	# ax=fig.add_subplot(111)
	# im=ax.imshow(phi[0],vmin=0,vmax=10)
	# plt.colorbar(im)
	# fig.canvas.draw()
	# fig.canvas.flush_events()
	# i=1
	if nt is None:
		try:
			while True: 
				RHS=np.fft.irfft2(-nu*eps**2*np.sum(k**2,axis=0)[nx,:,:]*np.fft.rfft2(phi))
				RHS+=np.sum(U[:,nx,:,:]*np.fft.irfft2(1.J*k[:,nx,:,:]*np.fft.rfft2(*phi[nx,:,:,:])),axis=0)
				RHS+=U-c[:,nx,nx]
				err=np.max(np.abs(RHS))
				phi+=RHS*dt
				print(err, end="\r")
				# i+=1
				# if i%100==0:
				# 	im.set_array(phi[0])
				# 	im.set_clim(phi.min(),phi.max())
				# 	fig.canvas.draw()
				# 	fig.canvas.flush_events()
		except KeyboardInterrupt:
			pass
	else:
		for it in tqdm(np.arange(nt)):
			RHS=np.fft.irfft2(-nu*eps**2*np.sum(k**2,axis=0)[nx,:,:]*np.fft.rfft2(phi))
			RHS+=np.sum(U[:,nx,:,:]*np.fft.irfft2(1.J*k[:,nx,:,:]*np.fft.rfft2(*phi[nx,:,:,:])),axis=0)
			RHS+=U-c[:,nx,nx]
			err=np.max(np.abs(RHS))
			phi+=RHS*dt
			# i+=1
			# if i%100==0:
			# 	im.set_array(phi[0])
			# 	im.set_clim(phi.min(),phi.max())
			# 	fig.canvas.draw()
			# 	fig.canvas.flush_events()
	# plt.close('all')
	kappa1=2*nu*eps**2*np.mean(np.mean(phidag[nx,nx,:,:]*np.fft.irfft2(1.J*k[:,nx,:,:]*np.fft.rfft2(phi[nx,:,:,:])),axis=-1),axis=-1)
	kappa2=np.mean(np.mean(phidag[nx,nx,:,:]*(U[:,nx,:,:]-c[:,nx,nx,nx])*phi[nx,:,:,:],axis=-1),axis=-1)	
	return kappa1,kappa2,phidag,phi

neps=10
eps=10**(np.linspace(-2,0,neps)[::-1])
nu=1

nt=1000

kap1=np.zeros(neps)
kap2=np.zeros(neps)

for ieps in tqdm(np.arange(neps-6)+6):
	fid='data_inertia_%d.npz' %ieps
	phidag=np.load(fid)['phidag']
	phi=np.load(fid)['phi']
	if phidag.size!=n**2:
		f1=interp2d(np.linspace(0,2*np.pi,np.int_(phidag.size**0.5),endpoint=False),np.linspace(0,2*np.pi,np.int_(phidag.size**0.5),endpoint=False),phidag)
		f2=interp2d(np.linspace(0,2*np.pi,np.int_(phidag.size**0.5),endpoint=False),np.linspace(0,2*np.pi,np.int_(phidag.size**0.5),endpoint=False),phi[0])
		f3=interp2d(np.linspace(0,2*np.pi,np.int_(phidag.size**0.5),endpoint=False),np.linspace(0,2*np.pi,np.int_(phidag.size**0.5),endpoint=False),phi[1])
		phi=np.ones((2,n,n))
		phidag=f1(x,x)
		phi[0]=f2(x,x)
		phi[1]=f3(x,x)
	kappa1,kappa2,phidag,phi=ComputeKappa(nu,eps[ieps],phidag=phidag,phi=phi,nt=nt)
	kap1[ieps]=det(kappa1)**0.5
	kap2[ieps]=det(kappa2)**0.5
	if not np.isnan(kap1[ieps]):
		np.savez(fid,kappa=kappa1+kappa2,phidag=phidag,phi=phi)
