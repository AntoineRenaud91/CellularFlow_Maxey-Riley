import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk,ellipe
from tqdm import tqdm


def g(x):
	gg=np.zeros((x.shape))
	gg[x==1]=1
	y=x[(x>0)&(x<1)]
	gg[(x>0)&(x<1)]=y*(ellipk(1-y**2)-ellipe(1-y**2))/(ellipe(1-y**2)-y**2*ellipk(1-y**2))
	gg[x==1]=1
	return gg
def G(x):
	nx=x.size
	n=100
	xx,yy=np.meshgrid(x,np.linspace(0,1,n,endpoint=False),indexing='ij')
	gg=np.sum(g(xx*yy)*xx/n,axis=-1)
	return gg

def Z(a):
	n=100+np.int_(5*a)
	x=np.linspace(0,1,n+1)[1:]
	dx=x[0]
	Gt=G(x)
	return np.sum(ellipk(1-x**2)*np.exp(-Gt*a))*dx

a=10**(np.linspace(-5,5.5,105))
z=np.array([Z(a[ia]) for ia in tqdm(np.arange(105))])

z*=4/np.pi**2
ztest=2/np.pi**(3/2)*np.sqrt(np.log(a)/a)*(1+(3+0.57721+4*np.log(2)+np.log(np.log(a)))/2/np.log(a))
ztest1=2/np.pi**(3/2)*np.sqrt(np.log(a)/a)*(1+(np.log(np.log(a)))/2/np.log(a))
ztest2=2/np.pi**(3/2)*np.sqrt(np.log(a)/a)*(1+(3+0.57721+4*np.log(2))/2/np.log(a))

C=1+0.57721+4*np.log(2)
z1=2/np.pi**(3/2)*np.sqrt(np.log(a)/a)
z2=(2+C+np.log(np.log(a)))/2/np.log(a)
z3=((C+np.log(np.log(a)))**2-2*(np.log(np.log(a))-1)**2-np.pi**2-32)/(4*np.log(a))**2

