#!/usr/bin/env python
'''
finds the optimal encoder for the filtering problem.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

def d_eps_dt(eps,gamma,eta,alpha,lamb):
	return -2*gamma*eps+eta**2-lamb*eps**2/(alpha**2+eps)
	#return -np.dot(gamma,eps)-np.dot(eps,gamma.T)+np.dot(eta.T,eta)-lamb*np.linalg.solve(alpha+eps,np.dot(eps,eps))

def get_eq_eps(gamma,eta,alpha,lamb):
	f = lambda e : d_eps_dt(e,gamma,eta,alpha,lamb)
	return opt.fsolve(f,1.0)

if __name__=='__main__':

    alphas = np.arange(0.001,4.0,0.01)
    eps = np.zeros_like(alphas)

    gamma = 0.1
    eta = 1.0
    phi = 0.1
    for n,alpha in enumerate(alphas):
        lamb = phi*np.sqrt(2*np.pi*alpha)
        eps[n] = get_eq_eps( gamma, eta, alpha, lamb )

    plt.plot( alphas, eps )
    plt.savefig('estimation_uni.png')
