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
    return numpy.dot( a, eps) + numpy.dot( eps, a.T) + numpy.dot(eta, eta.T) -lamb*numpy.linalg.solve( alpha+eps ,numpy.dot(eps,eps))
    #return -numpy.dot(gamma,eps)-numpy.dot(eps,gamma.T)+numpy.dot(eta.T,eta)-lamb*numpy.linalg.solve(alpha+eps,numpy.dot(eps,eps))

def get_eq_eps(gamma,eta,alpha,lamb):
    f = lambda e : d_eps_dt(e.reshape((2,2)),gamma,eta,alpha,lamb).reshape((4,))
    ret =  scipy.optimize.fsolve(f,numpy.eye(2).reshape((4,))).reshape((2,2))
    return ret

if __name__=='__main__':
    a = -0.1*numpy.eye(2)
    eta = .40*numpy.eye(2)
    dtheta = 0.5
    phi = .05
    sigma = numpy.eye(2)
    thetas = numpy.arange(0.0001,numpy.pi/2-0.0001,0.0001)
#    eps = numpy.zeros_like(alphas)
    const_eps = numpy.zeros_like(thetas)
    for n,i in enumerate(thetas):
        alpha = numpy.array([[numpy.tan(i),0.0],[0.0,1.0/numpy.tan(i)]])
        lamb = phi*numpy.sqrt((2*numpy.pi)**2*numpy.linalg.det(alpha))/dtheta**2
        print lamb
        const_eps[n] = numpy.trace( get_eq_eps( a, eta, alpha,lamb ) )
        """
    for n,i in enumerate(alphas):
        for m,j in enumerate(alphas):
            print m,n
            alpha = numpy.array([[i**2,0.0],[0.0,j**2]])
            lamb = phi*numpy.sqrt(2*numpy.pi*numpy.linalg.det(alpha))
            eps[n,m] = numpy.trace(get_eq_eps( a, eta, alpha, lamb ))
"""
    plt.plot( thetas, const_eps )
    plt.savefig('estimation_const.png')

   # plt.imshow(eps)
   # plt.colorbar()
   # plt.savefig('estimation_multi_hm.png')
