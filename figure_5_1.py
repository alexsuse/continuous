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

def d_eps_dt(eps,gamma,eta,alpha,lamb):
    return -2*gamma*eps+eta**2-lamb*eps**2/(alpha**2+eps)
#return -np.dot(gamma,eps)-np.dot(eps,gamma.T)+np.dot(eta.T,eta)-lamb*np.linalg.solve(alpha+eps,np.dot(eps,eps))

def get_eq_eps(gamma,eta,alpha,lamb):
    f = lambda e : d_eps_dt(e,gamma,eta,alpha,lamb)
    return opt.fsolve(f,1.0)

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


    ax1 = ppl.subplot2grid((3,3),[0,2],rowspan=3)
    ax2 = ppl.subplot2grid((3,3),[0,0])
    ax3 = ppl.subplot2grid((3,3),[1,0], sharex=ax2)
    ax4 = ppl.subplot2grid((3,3),[2,0], sharex=ax2)
    ax5 = ppl.subplot2grid((3,3),[0,1])
    ax6 = ppl.subplot2grid((3,3),[1,1], sharex=ax5)
    ax7 = ppl.subplot2grid((3,3),[2,1], sharex=ax5)

    def rate(x,alpha,phi,theta):
        return phi*np.exp(-(x-theta)**2/(2*alpha**2))

    thetas = np.arange(-7.0,7.0,dtheta)
    xs = np.arange(-2.2,2.2,0.01)

    rates1 = [[rate(x,0.1,phi,t) for x in xs] for t in thetas]
    rates2 = [[rate(x,1.0,phi,t) for x in xs] for t in thetas]
    rates3 = [[rate(x,2.0,phi,t) for x in xs] for t in thetas]

    plt.rcParams['text.usetex']=True
    #plt.rcParams['axes.labelsize']=6

    font = {'family':'normal',
            'weight':'bold',
            'size':8}
    plt.rc('font',**font)

    ppl.plot( alphas, eps, ax=ax1)
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$MMSE$')
    ax1.xaxis.set_ticks(np.arange(0.0,4.0,1.0))
    ppl.plot(xs,rates1[0],label=r'$\alpha=0.1$',ax=ax2)
    lines1 = map(lambda x : ppl.plot(xs,x,ax=ax2), rates1[1:])
    ppl.plot(xs,rates3[0],label=r'$\alpha=1.0$',ax=ax3)
    lines2 = map(lambda x : ppl.plot(xs,x,ax=ax3), rates2[1:])
    ppl.plot(xs,rates3[0],label=r'$\alpha=2.0$',ax=ax4)
    lines3 = map(lambda x : ppl.plot(xs,x,ax=ax4), rates3[1:])

    cm.getMaternSample(gamma=gamma,eta=eta,order=1,phi=phi,
                       dtheta=dtheta,plot=True,ax=ax5,alpha=0.1,
                       timewindow=20000, label=r'$\alpha=0.1$')
    cm.getMaternSample(gamma=gamma,eta=eta,order=1,phi=phi,
                       dtheta=dtheta,plot=True,ax=ax6,alpha=1.0,
                       timewindow=20000, label=r'$\alpha=1.0$')
    cm.getMaternSample(gamma=gamma,eta=eta,order=1,phi=phi,dtheta=dtheta,
                       plot=True,ax=ax7,alpha=2.0, timewindow=20000,
                       label=r'$\alpha=2.0$')
    ax7.set_xlabel(r'Time [s]')

    map(lambda x : x.set_ylabel(r'$Rate$'),[ax2,ax3,ax4])
    map(lambda x : x.set_ylim([-0.01,1.4*phi]),[ax2,ax3,ax4])
    map(lambda x : ppl.legend(x),[ax2,ax3,ax4])
    ax2.set_title('Coding Strategies')
   # ppl.legend(ax2,[lines1[0]],[r'$\alpha=0.1$'])
   # ppl.legend(ax3,[lines2[0]],[r'$\alpha=1.0$'])
   # ppl.legend(ax4,[lines3[0]],[r'$\alpha=2.0$'])
    ax4.set_xlabel(r'$x$')

    plt.savefig('estimation_uni.png',dpi=400)
    plt.savefig('estimation_uni.pdf')
    plt.savefig('estimation_uni.eps')
