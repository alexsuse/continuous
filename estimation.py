#!/usr/bin/env python
'''
finds the optimal encoder for the filtering problem.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import os

def d_eps_dt(eps,gamma,eta,alpha,lamb):
	return -2*gamma*eps+eta**2-lamb*eps**2/(alpha**2+eps)
	#return -np.dot(gamma,eps)-np.dot(eps,gamma.T)+np.dot(eta.T,eta)-lamb*np.linalg.solve(alpha+eps,np.dot(eps,eps))

def get_eq_eps(gamma,eta,alpha,lamb):
	f = lambda e : d_eps_dt(e,gamma,eta,alpha,lamb)
	return opt.fsolve(f,1.0)

def full_stoc_sigma(sigma0,dt,N,a,eta,alpha,la,NSamples,rands=None, discard=0):
    sigmas = np.zeros((NSamples,N))
    sigmas[:,0] = sigma0
    if rands==None:
        rng = np.random.RandomState(12345)
        rands = (rng.uniform(size=(NSamples,N))<la*dt).astype('int')
    else:
        assert rands.shape == (NSamples, N)
        rands = (rands<la*dt).astype('int')
    for i in xrange(0, discard):
        rand_sample = (rng.uniform(size=(NSamples,1))<la*dt).astype('int')
        splus1 = np.asarray([sigmas[:,0]+dt*(2*a*sigmas[:,0]+eta**2),
                             alpha**2*sigmas[:,0]/(alpha**2+sigmas[:,0])])
        sigmas[:,0] = splus1[rand_sample[:,0],range(NSamples)]
    
    for i in xrange(1,N):
        splus1 = np.asarray([sigmas[:,i-1]+dt*(2*a*sigmas[:,i-1]+eta**2),
                             alpha**2*sigmas[:,i-1]/(alpha**2+sigmas[:,i-1])])
        sigmas[:,i] = splus1[rands[:,i],range(NSamples)]
    return np.mean(sigmas, axis = 0)

def replica_eps(gamma, eta, alpha, lamb, tol=1e-6):
    eps = eta**2/(2.0*gamma)
    U = lamb/(alpha**2+eps)
    phi = (np.sqrt(gamma**2+U*eta**2)-gamma) + lamb*np.log(1.0+eps/alpha**2)-U*eps
    phi = 0.5*phi
    for i in range(1000):
        eps = eta**2/2 *(1.0/np.sqrt(gamma**2+U*eta**2))
        U = lamb/(alpha**2+eps)
        phi = (np.sqrt(gamma**2+U*eta**2)-gamma) + lamb*np.log(1.0+eps/alpha**2)-U*eps
        phi = 0.5*phi
        if np.abs(eps -  eta**2/2 *(1.0/np.sqrt(gamma**2+U*eta**2))) < tol:
            break
    return alpha**2*(np.exp(2.0*phi/lamb) - 1)

if __name__=='__main__':

    alphas = np.arange(0.001,4.0,0.1)
    eps = np.zeros_like(alphas)
    rep_eps = np.zeros_like(alphas)
    stoc_eps = np.zeros_like(alphas)

    gamma = 0.1
    eta = 1.0
    phi = 0.1
    N = 100000
    dt = 0.0001
    discard = 5*int(1.0/(2*gamma*dt))
    print discard
    for n,alpha in enumerate(alphas):
        print n, alphas.size
        lamb = phi*np.sqrt(2*np.pi*alpha)
        eps[n] = get_eq_eps( gamma, eta, alpha, lamb )
        rep_eps[n] = replica_eps(gamma, eta, alpha, lamb )
        stoc_eps[n] =  np.mean(full_stoc_sigma(0.01, dt, N, -gamma,
                                       eta, alpha, lamb, 1000,
                                       discard=discard))
    plt.plot( alphas, eps, 'r', label='mean-field')
    plt.plot( alphas, rep_eps, 'g.-', label='replica')
    plt.plot( alphas, stoc_eps, 'b.', label='stochastic average')
    plt.legend()
    plt.show()
    plt.savefig('estimation_uni.png')
    os.system("mutt -a \"estimation_uni.png\" -s \"Plot\" --recipient=alexsusemihl@gmail.com")
