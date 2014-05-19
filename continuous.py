#!/usr/bin/env python
"""
continuous.py -- evaluate the covariance part of the optimal cost-to-go for LQG in continuous time

See git repository alexsuse/Thesis for more information.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt
import estimation as est

T = 2
dt = 0.001
q = 0.01
QT = 0.1
r = 0.1
eta = 1.0
a = -0.1
b = 0.2
alpha = 0.1
dtheta = 0.05
phi = 0.1

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


def full_stoc_f(sigma0,S,dt,a,sigma,alpha,b,q,r,la,NSamples,rands=None):
    f = sigma0*S[0]
    sigmas = full_stoc_sigma(sigma0,dt,N,a,sigma,alpha,la,NSamples,rands)
    f += dt*np.sum((b**2*S**2/r)*sigmas)
    return f

if __name__=='__main__':
    print __doc__

    # Number of time steps
    N = int(T/dt)

    # precompute solutions to the Ricatti eqtn
    S = solve_riccatti(N,dt,QT,a,b,q,r)

    # alpha values to be considered
    alphas = np.arange(0.001,20.0,0.1)

    # initial value of sigma
    s = 2.0

    # fs holds mean field costs
    fs = np.zeros_like(alphas)
    # full_fs holds stochastic costs
    full_fs = np.zeros_like(alphas)
    # estimation_eps holds etsimation errors
    estimation_eps = np.zeros_like(alphas)

    # number of samples to be used for stoch simulations  
    Nsamples = 5000
    print 'running '+str(alphas.size)+' runs'

    # precompute randoms for better visualization
    rands = np.random.uniform(size=(Nsamples,N))

    for exponent in [+0.0]:
        print 'running for exponent=%lf'%exponent
        finame = 'control_exp_'+str(exponent)+'_max_alpha_'+str(alphas[-1])+'.png'
        if finame in os.listdir('.'):
            print "already ran for this value"
            continue
        for i,alpha in enumerate(alphas):
            # main loop, compute lambda
            la = np.sqrt(2*np.pi*alpha**(2+exponent))*phi/dtheta
            fs[i] = mf_f(s,S,dt,a,eta,alpha,b,q,r,la)
            full_fs[i] = full_stoc_f(s,S,dt,a,eta,alpha,b,q,r,la,Nsamples,rands=rands)
            estimation_eps[i] = est.get_eq_eps(-a,eta,alpha,la)

        # find minimum of mf cost
        fsmin,indfs = (np.min(fs),np.argmin(fs))
        # find minimum of stoch cost
        fullmin,indfull = (np.min(full_fs),np.argmin(full_fs))
        rc('text',usetex='true')
        # redimension estimation error to be on the same range as control cost
        estimation_eps = np.max(full_fs)*estimation_eps/np.max(estimation_eps)
        # find minimum
        epsmin,indeps = (np.min(estimation_eps),np.argmin(estimation_eps))
        # plot it up
        plt.plot(alphas,fs,alphas,full_fs,alphas,estimation_eps,
                 alphas[indfs],fsmin,'o',
                 alphas[indfull],fullmin,'o',
                 alphas[indeps],epsmin,'o')
        plt.legend(['Mean Field', 'Stochastic Average','Filtering Error'])
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$f(\Sigma)$')
        plt.show()
        plt.savefig(finame)
        plt.close()

    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)

    l1, = ax1.plot(alphas, estimation_eps,'b' )
    ax1.plot(alphas[indeps],epsmin,'ko')

    l2,l3, = ax2.plot( alphas, fs,'r', alphas, full_fs,'g' )
    ax2.plot(alphas[indfs],fsmin,'ko',alphas[indfull],fullmin,'ko')

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(axis='x',which='both',bottom='off')
    ax2.tick_params(axis='x',which='both',top='off')
   
    ax1.set_ylabel(r'$MMSE$')
    ax2.set_ylabel(r'$f(\Sigma_0,t_0)$')
    ax2.set_xlabel(r'$\alpha$')
    
    plt.figlegend([l1,l2,l3],['estimation','mean field','stochastic'],'upper right')
    plt.savefig('comparison_uni.eps')
