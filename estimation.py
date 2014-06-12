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
	return -2*gamma*eps+eta**2-(lamb*eps**2)/(alpha**2+eps)
	#return -np.dot(gamma,eps)-np.dot(eps,gamma.T)+np.dot(eta.T,eta)-lamb*np.linalg.solve(alpha+eps,np.dot(eps,eps))

def d_eps_kalman(eps, gamma, eta, alpha):
    return -2*gamma*eps+eta**2-(eps**2)/alpha**2

def get_eq_eps(gamma,eta,alpha,lamb):
	f = lambda e, gamma=gamma, eta=eta, alpha=alpha, lamb=lamb :\
             d_eps_dt(e,gamma,eta,alpha,lamb)
	return opt.fsolve(f,1.0)

def get_eq_kalman(gamma,eta,alpha):
    f = lambda e, gamma=gamma,eta=eta,alpha=alpha :\
            d_eps_kalman(e,gamma,eta,alpha)
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
    return sigmas

def replica_eps(gamma, eta, alpha, lamb, tol=1e-9):
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

    alphas = np.arange(0.01,4.0,0.1)
    eps = np.zeros_like(alphas)
    ka_eps = np.zeros_like(alphas)
    rep_eps = np.zeros_like(alphas)
    stoc_eps = np.zeros_like(alphas)

    gamma = 1.0
    eta = 1.0
    phi = 1.0
    dtheta = 0.5
    for n,alpha in enumerate(alphas):
        print n, alphas.size
        lamb = np.sqrt(2*np.pi*alpha**2)*phi/dtheta
        eps[n] = get_eq_eps( gamma, eta, alpha, lamb )
        ka_eps[n] = get_eq_kalman( gamma,eta,alpha)
    #    rep_eps[n] = replica_eps(gamma, eta, alpha, lamb )
    #    stoc_eps[n] =  np.mean(full_stoc_sigma(0.01, dt, N, -gamma,
    #                                   eta, alpha, lamb, 1000,
    #                                   discard=discard))
    plt.plot( alphas, eps, 'r', label='mean-field')
    plt.plot( alphas, ka_eps, 'k', label='kalman')
    #plt.plot( alphas, rep_eps, 'g.-', label='replica')
    #plt.plot( alphas, stoc_eps, 'b.', label='stochastic average')
    plt.legend()
    plt.show()
    plt.savefig('estimation_uni.png')
    os.system("echo \"file\" | mutt -a \"estimation_uni.png\" -s \"Plot\" -- alexsusemihl@gmail.com")
