#!/usr/bin/env python
'''
finds the optimal encoder for the filtering problem.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as numpy
import scipy.optimize

def d_eps_dt(eps,a,eta,alpha,lamb):
    return numpy.dot( a, eps) +\
           numpy.dot( eps, a.T) +\
           numpy.dot(eta, eta.T) -\
           lamb*numpy.dot(eps,numpy.linalg.solve( alpha+eps ,eps))
    #return -numpy.dot(gamma,eps)-numpy.dot(eps,gamma.T)+numpy.dot(eta.T,eta)-lamb*numpy.linalg.solve(alpha+eps,numpy.dot(eps,eps))

def d_eps_kalman( eps, a, eta, alpha ):
    return numpy.dot( a, eps) +\
           numpy.dot( eps, a.T) +\
           numpy.dot(eta, eta.T) -\
           numpy.dot( eps, numpy.linalg.solve( alpha, eps))

def get_eq_kalman(gamma,eta,alpha, N=2):
    f = lambda e, g=gamma,et=eta,a=alpha :\
            d_eps_kalman(e.reshape((N,N)),g,et,a).reshape((N**2,))
    ret =  scipy.optimize.fsolve(f,numpy.eye(N).reshape((N**2,))).reshape((N,N))
    return ret

def get_eq_eps(gamma,eta,alpha,lamb, N=2):
    f = lambda e, g=gamma,et=eta,a=alpha,l=lamb :\
            d_eps_dt(e.reshape((N,N)),g,et,a,l).reshape((N**2,))
    ret =  scipy.optimize.fsolve(f,numpy.eye(N).reshape((N**2,))).reshape((N,N))
    return ret

if __name__=='__main__':
    a = -0.1*numpy.eye(2)
    eta = .40*numpy.eye(2)
    dtheta = 0.1
    phi = .05
    sigma = numpy.eye(2)
    thetas = numpy.arange(0.0001,numpy.pi/2-0.0001,0.01)
#    eps = numpy.zeros_like(alphas)
    const_eps = numpy.zeros_like(thetas)
    kalman_eps = numpy.zeros_like(thetas)
    for n,i in enumerate(thetas):
        alpha = numpy.array([[numpy.tan(i),0.0],[0.0,1.0/numpy.tan(i)]])
        lamb = phi*numpy.sqrt((2*numpy.pi)**2*numpy.linalg.det(alpha))/dtheta**2
        print n
        const_eps[n] = numpy.trace( get_eq_eps( a, eta, alpha,lamb ) )
        kalman_eps[n] = numpy.trace( get_eq_kalman( a, eta, alpha ) )
        """
    for n,i in enumerate(alphas):
        for m,j in enumerate(alphas):
            print m,n
            alpha = numpy.array([[i**2,0.0],[0.0,j**2]])
            lamb = phi*numpy.sqrt(2*numpy.pi*numpy.linalg.det(alpha))
            eps[n,m] = numpy.trace(get_eq_eps( a, eta, alpha, lamb ))
"""
    ax1 = plt.subplot(122)
    ax2 = plt.subplot(321)
    ax3 = plt.subplot(323, sharex=ax2)
    ax4 = plt.subplot(325, sharex=ax2)

    ax1.plot( thetas, const_eps )
    ax1.plot( thetas, kalman_eps )
    ax1.legend(['Poisson'])
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'MMSE')
    plt.savefig('estimation_const.png')

   # plt.imshow(eps)
   # plt.colorbar()
   # plt.savefig('estimation_multi_hm.png')
