#!usr/bin/env python
"""
continuous.py -- evaluate the covariance part of the optimal cost-to-go for LQG in continuous time

See git repository alexsuse/Thesis for more information.
"""
import numpy as np
import matplotlib.pyplot as plt
T = 10
dt = 0.001
q = 0.1
QT = .40
r = 0.01
sigma = 1.0
a = .10
b = 1.0
alpha = 0.1
la = 2

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

if __name__=='__main__':
	N = int(T/dt)
	S = solve_riccatti(N,dt,QT,a,b,q,r)
	alphas = np.arange(0.001,5.0,0.01)
	s = 0.1
	fs = np.zeros_like(alphas)
	print 'running '+str(alphas.size)+' runs'
	for i,alpha in enumerate(alphas):
		print i
		fs[i] = mf_f(s,S,dt,a,sigma,alpha,b,q,r,la*alpha)

	plt.plot(alphas,fs)
	plt.show()
	input()
	#print "don't run this as a script, come on!"
