import numpy as np
from numpy import newaxis as nx
import joblib as jl
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import eigs,eigsh,inv
from scipy.linalg import det
import scipy.sparse.linalg as sl
from tqdm import tqdm
import os

def Laplacian(n):
	r,c,d=np.arange(n),np.arange(n),np.ones(n)
	r=np.concatenate((r,(r+1)%n,r))
	c=np.concatenate((c,c,(c+1)%n))
	d=np.concatenate((-4*d,d,d))
	for i in np.arange(n):
		if i==0:
			row=np.copy(r)
			col=np.copy(c)
			data=np.copy(d)
		else:
			row=np.concatenate((row,r+i*n))
			col=np.concatenate((col,c+i*n))
			data=np.concatenate((data,d))
	laplacian=sp.csr_matrix(sp.coo_matrix((data,(row,col)),shape=(n**2,n**2)))
	up_down_diag = np.ones(n**2-n)
	boundary=np.ones(n)
	diagonals = [up_down_diag,up_down_diag,boundary,boundary]
	laplacian += sp.diags(diagonals, [n,-n,n**2-n,-n**2+n], format="csc")
	return laplacian
def Gradient(n):
	r,c,d=np.arange(n),np.arange(n),np.ones(n)
	r=np.concatenate((r,(r+1)%n))
	c=np.concatenate(((c+1)%n,c))
	d=np.concatenate((d,-d))/2
	for i in np.arange(n):
		if i==0:
			row=np.copy(r)
			col=np.copy(c)
			data=np.copy(d)
		else:
			row=np.concatenate((row,r+i*n))
			col=np.concatenate((col,c+i*n))
			data=np.concatenate((data,d))
	Grad_x=sp.csr_matrix(sp.coo_matrix((data,(row,col)),shape=(n**2,n**2)))
	up_down_diag = np.ones(n**2-n)
	boundary=np.ones(n)
	diagonals = [up_down_diag/2,-up_down_diag/2,-boundary/2,boundary/2]
	Grad_y = sp.diags(diagonals, [n,-n,n**2-n,-n**2+n], format="csc")
	return Grad_x,Grad_y

def CellularFlow(n):
	x=np.linspace(0,2*np.pi,n,endpoint=False)
	dx=x[1]-x[0]
	xx,yy=np.meshgrid(x,x,indexing='ij')
	u=np.zeros((2,n,n))
	u[0]=np.sin(xx)*np.cos(yy)
	u[1]=-np.cos(xx)*np.sin(yy)
	Acc=np.zeros((2,n,n))
	Acc[0]=np.sin(2*xx)/2
	Acc[1]=np.sin(2*yy)/2
	return u,Acc,dx

def SolveStat(L,A=None,Phi=None,nt=100000):
	if Phi is None:
		Phi=np.ones((L.shape[0]))
	plt.ion()
	fig=plt.figure()
	ax=fig.add_subplot(111)
	im=ax.imshow(Phi.reshape((n,n)),vmin=0,vmax=10)
	c=plt.colorbar(im)
	fig.canvas.draw()
	fig.canvas.flush_events()
	dt=1/L.max()/10
	Op=inv(sp.csc_matrix(sp.diags([np.ones(n**2)],[0])-dt*L))
	if A is None:
		for it in tqdm(np.arange(nt)):
			Phi=Op.dot(Phi)
			if it%10==0:
				# print("Progress {:1.02%}".format(eps/err), end="\r")
				im.set_array(Phi.reshape((n,n)))
				im.set_clim(Phi.min(),Phi.max())
				fig.canvas.draw()
				fig.canvas.flush_events()
	else:
		for it in tqdm(np.arange(nt)):
			Phi+=Op.dot(Phi+A*dt)
	return Phi



# y=np.load('../f_q/data/x_Pe10000_St0.01.npz')['Y']
# t=np.load('../f_q/data/x_Pe10000_St0.01.npz')['t']
# kappa_d=np.mean(y[:,nx,:]*y[nx,:,:],axis=-1)/t

eps=0.5
nu= 1

n=100

u,Acc,dx=CellularFlow(n)
u=u.reshape((2,n**2))-eps*Acc.reshape((2,n**2))
Lap=Laplacian(n)/dx**2
Gy,Gx=Gradient(n)
Gx/=dx
Gy/=dx
UGx=Gx.dot(sp.diags([u[0]],[0]))
UGy=Gy.dot(sp.diags([u[1]],[0]))
Ldag=sp.csc_matrix(nu*eps**2*Lap-UGx-UGy)
phidag=SolveStat(Ldag)

c=np.mean(np.mean(u.reshape((2,n,n))*phidag[nx,:,:],axis=-1),axis=-1)
L=sp.csc_matrix(Ldag.transpose())

Linv=sp.linalg.inv(L)
varphi=np.squeeze(np.array([Linv.dot(c[0]-u[0]),Linv.dot(c[1]-u[1])]))
JacVarphi=np.array([[Gx.dot(varphi[0]),Gy.dot(varphi[0])],[Gx.dot(varphi[0]),Gy.dot(varphi[0])]])
uvarphi=(u[:,nx]-c[:,nx,nx])*varphi[nx,:,:]
kappa=np.mean((2*eps**2*nu*JacVarphi+uvarphi)*phi.reshape((1,1,n**2)),axis=-1)
kappa+=nu*eps**2*np.identity(2)
varphi=varphi.reshape((2,n,n))
plt.figure(1)
plt.imshow(phi)
plt.colorbar()
plt.figure(2)
plt.imshow(varphi[0])
plt.colorbar()
plt.figure(3)
plt.imshow(varphi[1])
plt.colorbar()
plt.show()
