#!/usr/bin/env python
'''
finds the optimal encoder for the filtering problem.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as numpy
import scipy.optimize

def d_eps_dt(eps,a,eta, inva,lamb,N=2):
    return numpy.dot( a, eps) +\
           numpy.dot( eps, a.T) +\
           numpy.dot(eta, eta.T) -\
           lamb*(eps - numpy.linalg.solve((numpy.eye(N)+numpy.dot(inva,eps)).T,eps))

def d_eps_kalman( eps, a, eta, inva ):
    return numpy.dot( a, eps) +\
           numpy.dot( eps, a.T) +\
           numpy.dot(eta, eta.T) -\
           numpy.dot( eps, numpy.dot( inva, eps))

def get_eq_kalman(gamma,eta,alpha, N=2):
    inva = numpy.linalg.pinv(alpha)
    f = lambda e, g=gamma,et=eta,inva=inva,N=N :\
            d_eps_kalman(e.reshape((N,N)),g,et,inva).reshape((N**2,))
    ret =  scipy.optimize.fsolve(f,numpy.eye(N).reshape((N**2,))).reshape((N,N))
    return ret

def get_eq_eps(gamma,eta,alpha,lamb, N=2):
    inva = numpy.linalg.pinv(alpha)
    f = lambda e, g=gamma,et=eta,a=inva,l=lamb,N=N :\
            d_eps_dt(e.reshape((N,N)),g,et,a,l,N=N).reshape((N**2,))
    ret =  scipy.optimize.fsolve(f,numpy.eye(N).reshape((N**2,))).reshape((N,N))
    return ret

if __name__=='__main__':
    gamma = 0.4
    omega = 0.8
    #a = numpy.array([[1.0,1.0],[-omega**2, -gamma]])
    a = numpy.array([[0.0,1.0,0.0,0.0],[-omega**2,-gamma,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,-omega**2,-gamma]])
    eta = .40*numpy.diag([0.0,1.0,0.0,1.0])
    dtheta = 0.1
    phi = 0.5
    sigma = eta**2*numpy.eye(4)
    thetas = numpy.arange(0.0001,numpy.pi/2,0.05)
#    eps = numpy.zeros_like(alphas)
    const_eps = numpy.zeros_like(thetas)
    kalman_eps = numpy.zeros_like(thetas)
    for n,i in enumerate(thetas):
        alpha = 0.1*numpy.diag([numpy.tan(i),0.0,1.0/numpy.tan(i),0.0])
        lamb = phi*numpy.sqrt((2*numpy.pi)**2)*0.1/(dtheta)**2
        print n, 'of',thetas.size
        const_eps[n] = numpy.trace( get_eq_eps( a, eta, alpha,lamb,N=4 ) )
        kalman_eps[n] = numpy.trace( get_eq_kalman( a, eta, alpha ,N=4) )
        """
    for n,i in enumerate(alphas):
        for m,j in enumerate(alphas):
            print m,n
            alpha = numpy.array([[i**2,0.0],[0.0,j**2]])
            lamb = phi*numpy.sqrt(2*numpy.pi*numpy.linalg.det(alpha))
            eps[n,m] = numpy.trace(get_eq_eps( a, eta, alpha, lamb ))
"""
    ax1 = plt.subplot(111)
    #ax2 = plt.subplot(321)
    #ax3 = plt.subplot(323, sharex=ax2)
    #ax4 = plt.subplot(325, sharex=ax2)

    ax1.plot( thetas, const_eps )
    ax1.plot( thetas, kalman_eps )
    ax1.legend(['Poisson'])
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'MMSE')
    plt.savefig('estimation_const.png')

   # plt.imshow(eps)
   # plt.colorbar()
   # plt.savefig('estimation_multi_hm.png')
