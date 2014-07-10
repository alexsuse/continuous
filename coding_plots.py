#!/usr/bin/env python
'''
finds the optimal encoder for the filtering problem.
'''
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import prettyplotlib as ppl
from prettyplotlib import plt
import coding_matern as cm
from estimation import get_eq_eps

if __name__=='__main__':

    alphas = np.arange(0.01,4.0,0.01)
    eps = np.zeros_like(alphas)

    gamma = 0.1
    eta = 1.0
    phi = 0.2
    dtheta = 0.7
    for n,alpha in enumerate(alphas):
        lamb = phi*np.sqrt(2*np.pi*alpha)/dtheta
        eps[n] = get_eq_eps( gamma, eta, alpha, lamb )
 
    ax1 = plt.subplot(133)
    ax2 = plt.subplot(331)
    ax3 = plt.subplot(334, sharex=ax2)
    ax4 = plt.subplot(337, sharex=ax2)
    ax5 = plt.subplot(332)
    ax6 = plt.subplot(335, sharex=ax5)
    ax7 = plt.subplot(338, sharex=ax5)

    def rate(x,alpha,phi,theta):
        return phi*np.exp(-(x-theta)**2/(2*alpha**2))

    thetas = np.arange(-7.0,7.0,dtheta)
    xs = np.arange(-2.2,2.2,0.01)

    rates1 = [[rate(x,0.3,phi,t) for x in xs] for t in thetas]
    rates2 = [[rate(x,1.0,phi,t) for x in xs] for t in thetas]
    rates3 = [[rate(x,2.0,phi,t) for x in xs] for t in thetas]

    plt.rcParams['text.usetex']=True

    ppl.plot( alphas, eps, ax=ax1 )
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$MMSE$')
    map(lambda x : ppl.plot(xs,x, ax=ax2), rates1)
    map(lambda x : ppl.plot(xs,x, ax=ax3), rates2)
    map(lambda x : ppl.plot(xs,x, ax=ax4), rates3)

    cm.getMaternSample(gamma=gamma,eta=eta,order=1,phi=phi,dtheta=dtheta,plot=True,ax=ax5,alpha=0.3, timewindow=20000, label=r'$\alpha=0.3$')
    cm.getMaternSample(gamma=gamma,eta=eta,order=1,phi=phi,dtheta=dtheta,plot=True,ax=ax6,alpha=1.0, timewindow=20000, label=r'$\alpha=1.0$')
    cm.getMaternSample(gamma=gamma,eta=eta,order=1,phi=phi,dtheta=dtheta,plot=True,ax=ax7,alpha=2.0, timewindow=20000, label=r'$\alpha=2.0$')

    map(lambda x : x.set_ylabel(r'$Rate$'),[ax2,ax3,ax4])
    map(lambda x : x.set_ylim([-0.01,1.7*phi]),[ax2,ax3,ax4])
    ax2.set_title('Coding Strategies')
   # ax2.legend([r'$\alpha=0.3$'])
   # ax3.legend([r'$\alpha=1.0$'])
   # ax4.legend([r'$\alpha=2.0$'])
    ax4.set_xlabel(r'$x$')

    ppl.plot(xs,rates1[0],label=r'$\alpha=0.3$',ax=ax2)
    plt.savefig('estimation_uni.png',dpi=300)
    plt.savefig('estimation_uni.eps')
