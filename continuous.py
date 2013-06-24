#!/usr/bin/env python
"""
continuous.py -- evaluate the covariance part of the optimal cost-to-go for LQG in continuous time

See git repository alexsuse/Thesis for more information.
"""
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
T = 10
dt = 0.001
q = 0.01
QT = 0.1
r = 0.1
eta = 1.0
a = -0.1
b = 0.2
alpha = 0.1
dtheta = 0.05
phi = 0.01

def solve_riccatti(N,dt,QT,a,b,q,r):
	s = np.zeros(N)
	s[-1] = QT
	for i in range(N-1):
		s[N-i-2] = s[N-i-1]+dt*(-s[N-i-1]**2*b**2/(2*r)+2*a*s[N-i-1]+0.5*q)
	return s

def mf_sigma(sigma0,dt,N,a,sigma,alpha,la):
	s = np.zeros(N)
	s[0] = sigma0
	for i in range(N)[1:]:
		s[i] =s[i-1]+dt*(2*a*s[i-1]+sigma**2-la*s[i-1]**2/(alpha**2+s[i-1]))
	return s

def mf_f(sigma0,S,dt,a,sigma,alpha,b,q,r,la):
	f = sigma0*S[0]
	f+= dt*np.sum((b**2*S**2/r)*mf_sigma(sigma0,dt,S.size,a,sigma,alpha,la))
	return f

def full_stoc_sigma(sigma0,dt,N,a,sigma,alpha,la,NSamples,rands=None):
	sigmas = np.zeros((NSamples,N))
	sigmas[:,0] = sigma0
	if rands==None:
		rng = np.random.RandomState(12345)
		rands = (rng.uniform(size=(NSamples,N))<la*dt).astype('int')
	else:
		rands = (rands<la*dt).astype('int')
	#for i in xrange(1,NSamples):
	#	for j in xrange(N):
	#		if rands[i,j]>la*dt:
	#			sigmas[i,j] = sigmas[i-1,j] + dt*(2*a*sigmas[i-1,j]+sigma**2)
	#		else:
	#			sigmas[i-1,j] = alpha**2*sigmas[i-1,j]/(alpha**2+sigmas[i-1,j])
	for i in xrange(1,N):
		splus1 = np.asarray([sigmas[:,i-1]+dt*(2*a*sigmas[:,i-1]+sigma**2),alpha**2*sigmas[:,i-1]/(alpha**2+sigmas[:,i-1])])
		sigmas[:,i] = splus1[rands[:,i],range(NSamples)]
	return np.mean(sigmas,axis=0)


def full_stoc_f(sigma0,S,dt,a,sigma,alpha,b,q,r,la,NSamples,rands=None):
	f = sigma0*S[0]
	sigmas = full_stoc_sigma(sigma0,dt,N,a,sigma,alpha,la,NSamples,rands)
	f += dt*np.sum((b**2*S**2/r)*sigmas)
	return f

if __name__=='__main__':
	N = int(T/dt)
	S = solve_riccatti(N,dt,QT,a,b,q,r)
	alphas = np.arange(0.001,4.0,0.1)
	s = 2.0
	fs = np.zeros_like(alphas)
	full_fs = np.zeros_like(alphas)
	Nsamples = 2000
	print 'running '+str(alphas.size)+' runs'
	rands = np.random.uniform(size=(Nsamples,N))
	for i,alpha in enumerate(alphas):
		print '---->\n.....Alpha = %s' % str(alpha) 
		la = np.sqrt(2*np.pi*alpha**2)*phi/dtheta
		fs[i] = mf_f(s,S,dt,a,eta,alpha,b,q,r,la)
		print '---->\n.....Mean field %s' % str(fs[i])
		full_fs[i] = full_stoc_f(s,S,dt,a,eta,alpha,b,q,r,la,Nsamples,rands=rands)
		print '---->\n.....Stochastic %s' % str(full_fs[i])

	rc('text',usetex='true')
	plt.plot(alphas,fs,alphas,full_fs)
	plt.legend(['Mean Field', 'Stochastic Average'])
	plt.xlabel(r'$\alpha$')
	plt.ylabel(r'$f(\Sigma)$')
	plt.show()
	try:
		a = raw_input('file name?')
		print 'saving to %s'%a
		plt.savefig(a)
	except SyntaxError,NameError:
		print 'saving to params.png'
		plt.savefig('params.png')
	#print "don't run this as a script, come on!"
