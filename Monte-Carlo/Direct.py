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

def Langevin_Step(x,dt,St,Pe,i):
	st=np.abs(St)
	if np.round(St/dt)>10:
		if i%2==0:
			x[0]+=(u(x[:2])+x[2]/st)*dt
			x[1]+=(v(x[:2])+x[3]/st)*dt
		else:
			x[1]+=(v(x[:2])+x[3]/st)*dt
			x[0]+=(u(x[:2])+x[2]/st)*dt
		ux=np.array([u(x[:2])[:],v(x[:2])[:]])
		gx=GradU(x[:2])
		aux=np.array([Au(x[:2])[:],Av(x[:2])[:]])
		x[2:]+=-x[2:]*dt/st+gx*x[2:][::-1]*dt-St*aux*dt
		x[2:]+=np.sqrt(2*dt/Pe)*np.random.normal(0,1,x[2:].shape)
	else: 
		Ni=np.int_(np.round(dt*10/St))+1
		dti=dt/Ni
		for it in np.arange(Ni)+i:
			if i%2==0:
				x[0]+=(u(x[:2])+x[2]/st)*dti
				x[1]+=(v(x[:2])+x[3]/st)*dti
			else:
				x[1]+=(v(x[:2])+x[3]/st)*dti
				x[0]+=(u(x[:2])+x[2]/st)*dti
			ux=np.array([u(x[:2])[:],v(x[:2])[:]])
			gx=GradU(x[:2])
			aux=np.array([Au(x[:2])[:],Av(x[:2])[:]])
			x[2:]+=-x[2:]*dti/st+gx*x[2:][::-1]*dti-St*aux*dti		
			x[2:]+=np.sqrt(2*dti/Pe)*np.random.normal(0,1,x[2:].shape)
	return x



		


def Comp(Pe,St,tmax,Toplot=False):
	rotmat=np.array([[1,-1],[1,1]])/np.sqrt(2)
	Pe=np.int_(Pe)
	St=np.int_(St*10**3)*10**-3
	nreal=10**4
	dt	= 0.05
	fid='./data/x_Pe-%d' %Pe
	fid+='_St-%01.03f.npz' %St
	if os.path.isfile(fid):
		if St==0.001:
			x=np.load(fid)['x']
		y=np.load(fid)['y']
		if np.abs(St)>=0.099:
			z=np.load(fid)['z']
		t=np.load(fid)['t']
	else:
		if St==0.001:
			x=rotmat.dot(np.random.random((2,nreal))*2*np.pi-np.pi)
		y=rotmat.dot(np.random.random((2,nreal))*2*np.pi-np.pi)
		if np.abs(St)>=0.099:
			z=rotmat.dot(np.random.random((2,nreal))*2*np.pi-np.pi)
			z=np.concatenate((z,np.zeros((2,nreal))),axis=0)
		t=0
	if Toplot:
		if St==0.001:
			X=rotmat.transpose().dot(x)/2/np.pi
		Y=rotmat.transpose().dot(y)/2/np.pi
		if np.abs(St)>=0.099:
			Z=rotmat.transpose().dot(z[:2])/2/np.pi
		plt.ion()
		fig=plt.figure()
		fig2=plt.figure()
		ax=fig.add_subplot(111)
		ax2=fig2.add_subplot(111)
		if np.abs(St)>=0.099:
			PlotC,=ax.plot(Z[0],Z[1],'ko',Markersize=0.1,alpha=0.5)
		PlotB,=ax.plot(Y[0],Y[1],'ro',Markersize=0.1,alpha=0.5)
		if St==0.001:
			PlotA,=ax.plot(X[0],X[1],'bo',Markersize=0.1,alpha=0.5)
		xb=np.floor(np.abs(Y).max())+1
		bins=np.linspace(-xb,xb,2*xb)
		if  St==0.001:
			pdfX,bins=np.histogram(X[0],bins=bins)
		pdfY,bins=np.histogram(Y[0],bins=bins)
		if np.abs(St)>=0.099:
			pdfZ,bins=np.histogram(Z[0],bins=bins)
		bins=(bins[1:]+bins[:-1])/2
		ax.set_xlim(-xb,xb)
		ax.set_ylim(-xb,xb)
		if St==0.001:
			PlotD,= ax2.semilogy(bins,pdfX+1,'b-')
		PlotE,= ax2.semilogy(bins,pdfY+1,'r-')
		if np.abs(St)>=0.099:
			PlotF,= ax2.semilogy(bins,pdfZ+1,'k-')
		fig.canvas.draw()
		fig.canvas.flush_events()
		fig2.canvas.draw()
		fig2.canvas.flush_events()
	it=0
	while t<tmax:
		it+=1
		t+=dt
		if St==0.001:
			x=Overdamped_Step(x,dt,0,Pe,it)
		y=Overdamped_Step(y,dt,St,Pe,it)
		if np.abs(St)>=0.099:
			z=Langevin_Step(z,dt,St,Pe,it)
		if it%1000==0:
			if St==0.001:
				np.savez(fid,x=x,y=y,t=t)
			elif np.abs(St)>=0.099:
				np.savez(fid,z=z,y=y,t=t)
			else:
				np.savez(fid,y=y,t=t)
			if Toplot:
				if St==0.001:
					X=rotmat.transpose().dot(x)/2/np.pi
				Y=rotmat.transpose().dot(y)/2/np.pi
				if np.abs(St)>=0.099:
					Z=rotmat.transpose().dot(z[:2])/2/np.pi
				if St==0.001:
					PlotA.set_xdata(X[0])
					PlotA.set_ydata(X[1])
				PlotB.set_xdata(Y[0])
				PlotB.set_ydata(Y[1])
				if np.abs(St)>=0.099:
					PlotC.set_xdata(Z[0])
					PlotC.set_ydata(Z[1])
				xb=np.floor(np.abs(Y).max())+1
				bins=np.linspace(-xb,xb,2*xb)
				if St==0.001:
					pdfX,bins=np.histogram(X[0],bins=bins)
				pdfY,bins=np.histogram(Y[0],bins=bins)
				if np.abs(St)>=0.099:
					pdfZ,bins=np.histogram(Z[0],bins=bins)
				bins=(bins[1:]+bins[:-1])/2
				ax.set_xlim(-xb,xb)
				ax.set_ylim(-xb,xb)
				if St==0.001:
					PlotD.set_ydata(pdfX+1)
					PlotD.set_xdata(bins)
				PlotE.set_ydata(pdfY+1)
				if np.abs(St)>=0.099:
					PlotF.set_ydata(pdfZ+1)
					PlotF.set_xdata(bins)
				PlotE.set_xdata(bins)
				ax2.set_ylim(1,pdfY.max()*1.1)
				ax2.set_xlim(bins.min()-0.5,bins.max()+0.5)
				fig.canvas.draw()
				fig.canvas.flush_events()
				fig2.canvas.draw()
				fig2.canvas.flush_events()
	if St==0.001:
		np.savez(fid,x=x,y=y,t=t)
	elif np.abs(St)>=0.099:
		np.savez(fid,z=z,y=y,t=t)
	else:
		np.savez(fid,y=y,t=t)
	return 



def Kappa(Pe,St,index):
	rotmat=np.array([[1,-1],[1,1]])/np.sqrt(2)
	fid='./data/x_Pe-%d' %Pe
	fid+='_St-%01.03f.npz' %St
	if os.path.isfile(fid):
		if index==0:
			return np.mean(rotmat.transpose().dot(np.load(fid)['x'])**2)/np.load(fid)['t']/2
		elif index==1:
			return np.mean(rotmat.transpose().dot(np.load(fid)['y'])**2)/np.load(fid)['t']/2
		elif (index==2)&(St>=0.099):
			return np.mean(rotmat.transpose().dot(np.load(fid)['z'][:2])**2)/np.load(fid)['t']/2
		else:
			return 0
	else:
		return 0 


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

def Z_asymp(a):
	return 2/np.pi**(3/2)*np.sqrt(np.log(a)/a)*(1+(3+0.57721+4*np.log(2)+np.log(np.log(a)))/2/np.log(a))


nu=0.532740705
nSt=6
nPe=11

St=np.int_(10**(np.linspace(0,3,nSt,endpoint=False)))*10**(-3)
Pe=np.int_(10**(np.linspace(1,6,nPe)))


tmax=2*10**4

# jl.Parallel(n_jobs=7)(jl.delayed(Comp)(Pe[iPe],-St[iSt],tmax) for iSt in tqdm(np.arange(nSt)) for iPe in np.arange(nPe))


for ist in np.arange(nSt):
	Kappa_0=np.array([Kappa(Pe[ipe],0.001,0) for ipe in tqdm(np.arange(nPe))])*Pe
	Kappa_1=np.array([Kappa(Pe[ipe],St[ist],1) for ipe in tqdm(np.arange(nPe))])*Pe
	Kappa_2=np.array([Kappa(Pe[ipe],St[ist],2) for ipe in tqdm(np.arange(nPe))])*Pe
	Peb=10**(np.linspace(1,6,nPe*5))
	kapSoward=2*nu*Peb**0.5
	a=St[ist]*Peb
	zaf=np.array([Z(a[ia]) for ia in tqdm(np.arange(a.size))])
	kapInit=kapSoward/zaf
	#za=2/np.pi**(3/2)*np.sqrt(np.log(a)/a)*(1+(3+0.57721+4*np.log(2)+np.log(np.log(a)))/2/np.log(a))
	#KapInit_2=kapSoward/za
	plt.semilogx(Pe,Kappa_0,'ro',label=r'Passive tracer')
	plt.semilogx(Peb,kapInit,'k--')
	plt.semilogx(Pe,Kappa_1,'bo',label=r'Inertial particles - Overdamped - with $\mathrm{St}=%01.03f$'%(St[ist]))
	if St[ist]>=0.099:
		plt.semilogx(Pe,Kappa_2,'o',color='purple',label=r'Inertial particles - Langevin - with $\mathrm{St}=%01.03f$' %(St[ist]))
	plt.semilogx(Peb,kapSoward,'k--',label=r'Asymp')
	plt.semilogx(Pe,Kappa_0,'ro')
	#plt.semilogx(Peb,KapInit_2,'k--')
	plt.xlabel(r'$\mathrm{Pe}$',fontsize=15)
	plt.ylabel(r'$\overline{\kappa}$',fontsize=15)
	plt.legend()
	plt.savefig('St_%d.png' %ist)
	plt.close('all')

for ist in np.arange(nSt):
	Kappa_0=np.array([Kappa(Pe[ipe],0.001,0) for ipe in tqdm(np.arange(nPe))])*Pe
	Kappa_1=np.array([Kappa(Pe[ipe],-St[ist],1) for ipe in tqdm(np.arange(nPe))])*Pe/(1+Pe/100000)
	Kappa_2=np.array([Kappa(Pe[ipe],-St[ist],2) for ipe in tqdm(np.arange(nPe))])*Pe/(1+Pe/100000)
	Peb=10**(np.linspace(1,6,nPe*5))
	kapSoward=2*nu*Peb**0.5
	a=-St[ist]*Peb
	zaf=np.array([Z(a[ia]) for ia in tqdm(np.arange(a.size))])
	kapInit=kapSoward/zaf
	#za=2/np.pi**(3/2)*np.sqrt(np.log(a)/a)*(1+(3+0.57721+4*np.log(2)+np.log(np.log(a)))/2/np.log(a))
	#KapInit_2=kapSoward/za
	plt.semilogx(Pe,Kappa_0,'ro',label=r'Passive tracer')
	plt.semilogx(Peb,kapInit,'k--')
	plt.semilogx(Pe,Kappa_1,'bo',label=r'Inertial particles - Overdamped - with $\mathrm{St}=%01.03f$'%(St[ist]))
	# if np.abs(St[ist])>=0.099:
	# 	plt.semilogx(Pe,Kappa_2,'o',color='purple',label=r'Inertial particles - Langevin - with $\mathrm{St}=%01.03f$' %(-St[ist]))
	plt.semilogx(Peb,kapSoward,'k--',label=r'Asymp')
	plt.semilogx(Pe,Kappa_0,'ro')
	#plt.semilogx(Peb,KapInit_2,'k--')
	plt.xlabel(r'$\mathrm{Pe}$',fontsize=15)
	plt.ylabel(r'$\overline{\kappa}$',fontsize=15)
	plt.legend()
	plt.savefig('St_%d.png' %(-ist-1))
	plt.close('all')


for ipe in np.arange(nPe):
	Kappa_0=np.array([Kappa(Pe[ipe],0.001,0)])*Pe[ipe] 
	Kappa_1=np.array([Kappa(Pe[ipe],St[ist],1) for ist in tqdm(np.arange(nSt))])*Pe[ipe]
	Kappa_2=np.array([Kappa(Pe[ipe],St[ist],2) for ist in tqdm(np.arange(nSt))])*Pe[ipe]
	Stb=10**(np.linspace(0,np.log(St.max()*10**3)/np.log(10),nSt*5))*10**(-3)
	kapSoward=2*nu*Pe[ipe]**0.5
	a=Stb*Pe[ipe]
	zaf=np.array([Z(a[ia]) for ia in tqdm(np.arange(a.size))])
	kapInit=kapSoward/zaf
	#za=2/np.pi**(3/2)*np.sqrt(np.log(a)/a)*(1+(3+0.57721+4*np.log(2)+np.log(np.log(a)))/2/np.log(a))
	#KapInit_2=kapSoward/za
	plt.semilogx(St,Kappa_0*np.ones(St.size),'ro',label=r'Passive tracer - with $\mathrm{Pe}=%d$'%(Pe[ipe]))
	plt.semilogx(Stb,kapInit,'k--')
	plt.semilogx(St,Kappa_1,'bo',label=r'Inertial particles - Overdamped - with $\mathrm{Pe}=%d$'%(Pe[ipe]))
	plt.semilogx(St[Kappa_2>0],Kappa_2[Kappa_2>0],'o',color='purple',label=r'Inertial particles - Langevin - with $\mathrm{Pe}=%d$' %(Pe[ipe]))
	plt.semilogx(Stb,kapSoward*np.ones(Stb.size),'k--',label=r'Asymp')
	plt.semilogx(St,Kappa_0*np.ones(St.size),'ro')
	#plt.semilogx(Peb,KapInit_2,'k--')
	plt.xlabel(r'$\mathrm{St}$',fontsize=15)
	plt.ylabel(r'$\overline{\kappa}$',fontsize=15)
	plt.legend()
	plt.savefig('Pe_%d.png' %ipe)
	plt.close('all')


for ipe in np.arange(nPe):
	Kappa_0=np.array([Kappa(Pe[ipe],0.001,0)])*Pe[ipe] 
	Kappa_1=np.array([Kappa(Pe[ipe],-St[ist],1) for ist in tqdm(np.arange(nSt))])*Pe[ipe]
	Kappa_2=np.array([Kappa(Pe[ipe],-St[ist],2) for ist in tqdm(np.arange(nSt))])*Pe[ipe]
	Stb=10**(np.linspace(0,np.log(St.max()*10**3)/np.log(10),nSt*5))*10**(-3)
	kapSoward=2*nu*Pe[ipe]**0.5
	a=-Stb*Pe[ipe]
	zaf=np.array([Z(a[ia]) for ia in tqdm(np.arange(a.size))])
	kapInit=kapSoward/zaf
	#za=2/np.pi**(3/2)*np.sqrt(np.log(a)/a)*(1+(3+0.57721+4*np.log(2)+np.log(np.log(a)))/2/np.log(a))
	#KapInit_2=kapSoward/za
	plt.semilogx(St,Kappa_0*np.ones(St.size),'ro',label=r'Passive tracer - with $\mathrm{Pe}=%d$'%(Pe[ipe]))
	plt.semilogx(Stb,kapInit,'k--')
	plt.semilogx(St,Kappa_1,'bo',label=r'Inertial particles - Overdamped - with $\mathrm{Pe}=%d$'%(Pe[ipe]))
	#plt.semilogx(St[Kappa_2>0],Kappa_2[Kappa_2>0],'o',color='purple',label=r'Inertial particles - Langevin - with $\mathrm{Pe}=%d$' %(Pe[ipe]))
	plt.semilogx(Stb,kapSoward*np.ones(Stb.size),'k--',label=r'Asymp')
	plt.semilogx(St,Kappa_0*np.ones(St.size),'ro')
	#plt.semilogx(Peb,KapInit_2,'k--')
	plt.xlabel(r'$\mathrm{St}$',fontsize=15)
	plt.ylabel(r'$\overline{\kappa}$',fontsize=15)
	plt.legend()
	plt.savefig('Pe_%d.png' %(-ipe-1))
	plt.close('all')


Kappa_0=np.array([Kappa(Pe[ipe],0.001,0) for ipe in tqdm(np.arange(nPe))])*Pe
Peb=10**(np.linspace(1,6,nPe*5))
kapSoward=2*nu*Peb**0.5
plt.semilogx(Pe,Kappa_0,'ro',label=r'Passive tracer')
plt.semilogx(Peb,kapSoward,'k--',label=r'Asymp')
plt.semilogx(Pe,Kappa_0,'ro')
plt.xlabel(r'$\mathrm{Pe}$',fontsize=15)
plt.ylabel(r'$\overline{\kappa}$',fontsize=15)
plt.legend()
plt.savefig('St_0.png')
plt.close('all')



# Kappa=np.zeros((nPe*5,nSt*5))
# Peb=10**(np.linspace(1,6,nPe*5))
# Stb=10**(np.linspace(0,np.log(St.max()*10**3)/np.log(10),nSt*5))*10**(-3)
# for ipe in np.arange(nPe*5):
# 	for ist in np.arange(nSt*5):
# 		Kappa[ipe,ist] = 2*nu/Peb[ipe]**0.5/Z(Peb[ipe]*Stb[ist])
