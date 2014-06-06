#!/usr/bin/env python
"""
continuous.py -- evaluate the covariance part of the optimal cost-to-go for LQG in continuous time

See git repository alexsuse/Thesis for more information.
"""
import os
import cPickle as pic
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt
import estimation as est
from estimation import full_stoc_sigma

T = 2
dt = 0.001
q = 0.5
QT = 0.0
r = 0.1
eta = 0.6
a = -1.0
b = 0.2
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
    #f = sigma0*S[0]
    f = dt*np.sum((b**2*S**2/r)*mf_sigma(sigma0,dt,S.size,a,sigma,alpha,la))
    return f

def mutual_info(dt,a,eta,alpha,la,NSamples,rands=None):
    sigma0 = est.get_eq_kalman(-a,eta,1e10)
    sigmas = full_stoc_sigma(sigma0,dt,1,a,eta,alpha,la,NSamples,rands,discard=10000)
    return np.log(sigma0) - np.mean(np.log(sigmas))

def full_stoc_f(sigma0,S,dt,a,sigma,alpha,b,q,r,la,NSamples,rands=None):
    #f = sigma0*S[0]
    sigmas = full_stoc_sigma(sigma0,dt,N,a,sigma,alpha,la,NSamples,rands)
    f = dt*np.sum((b**2*S**2/r)*np.mean(sigmas,axis=0))
    return f

def kalman_sigma(sigma0,dt,N,a,sigma,alpha):
    s = np.zeros(N)
    s[0] = sigma0
    for i in range(N)[1:]:
        change = 2*a*s[i-1]+sigma**2
        change -= s[i-1]*s[i-1]/alpha**2
        s[i] =s[i-1]+dt*change
    return s

def lqg_f(sigma0,S,dt,a,sigma,alpha,b,q,r):
    f = dt*np.sum((b**2*S**2/r)*kalman_sigma(sigma0,dt,S.size,a,sigma,alpha))
    return f

if __name__=='__main__':
    print __doc__

    # Number of time steps
    N = int(T/dt)

    # precompute solutions to the Ricatti eqtn
    S = solve_riccatti(N,dt,QT,a,b,q,r)

    # alpha values to be considered
    alphas = np.arange(0.1,5.0,0.05)

    # initial value of sigma
    s = 2.0

    # fs holds mean field costs
    fs = np.zeros_like(alphas)
    # full_fs holds stochastic costs
    full_fs = np.zeros_like(alphas)
    # estimation_eps holds etsimation errors
    estimation_eps = np.zeros_like(alphas)
    # kalman_eps gives the equilibrium variance for the KB filter
    kalman_eps = np.zeros_like(alphas)
    # lqg_fs
    lqg_fs = np.zeros_like(alphas)
    # mutual_information
    mi = np.zeros_like(alphas)

    # number of samples to be used for stoch simulations  
    Nsamples = 300
    print 'running '+str(alphas.size)+' runs'

    # precompute randoms for better visualization
    rands = np.random.uniform(size=(Nsamples,N))

    for i,alpha in enumerate(alphas):
        # main loop, compute lambda
        print i, 'of', alphas.size
        la = np.sqrt(2*np.pi*alpha**2)*phi/dtheta
        fs[i] = mf_f(s,S,dt,a,eta,alpha,b,q,r,la)
        full_fs[i] = full_stoc_f(s,S,dt,a,eta,alpha,b,q,r,la,Nsamples,rands=rands)
        estimation_eps[i] = est.get_eq_eps(-a,eta,alpha,la)
        kalman_eps[i] = est.get_eq_kalman(-a,eta,alpha)
        lqg_fs[i] = lqg_f(s,S,dt,a,eta,alpha,b,q,r)
        mi[i] = mutual_info(dt,a,eta,alpha,la,4*Nsamples)

    # find minimum of mf cost
    fsmin,indfs = (np.min(fs),np.argmin(fs))
    # find minimum of stoch cost
    fullmin,indfull = (np.min(full_fs),np.argmin(full_fs))
    rc('text',usetex='true')
    # find minimum
    epsmin,indeps = (np.min(estimation_eps),np.argmin(estimation_eps))
    # plot it up
    mi = mi*estimation_eps[0]/np.max(mi)

    dic = {'LQG control':lqg_fs,
           'poisson filtering':estimation_eps,
           'kalman filtering':kalman_eps,
           'mutual information':mi,
           'mf poisson control':fs,
           'full poisson control':full_fs,
           'alphas':alphas}

    with open("figure-1-1-low.pik","w") as fi:
        pic.dump(dic, fi)

    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)

    l1, = ax1.plot(alphas, estimation_eps,'b')
    l2, = ax1.plot(alphas, kalman_eps, 'k-.' )
    l6, = ax1.plot(alphas, mi, 'g')
    ax1.plot(alphas[indeps],epsmin,'ko')

    ax1.text(alphas[2],0.16,'a)')

    l3,l4,=ax2.plot( alphas, fs,'r', alphas, full_fs,'g' )
    l5, = ax2.plot(  alphas, lqg_fs, 'k-.' )
    ax2.plot(alphas[indfs],fsmin,'ko',alphas[indfull],fullmin,'ko')

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(axis='x',which='both',bottom='off')
    ax2.tick_params(axis='x',which='both',top='off')
   
    ax1.set_ylabel(r'$MMSE$')
    ax2.set_ylabel(r'$f(\Sigma_0,t_0)$')
    ax2.set_xlabel(r'$p$')
    
    plt.figlegend([l1,l2,l6,l3,l4,l5],
                  ['estimation','Kalman filter','Mutual Information','mean field','stochastic','LQG control'],
                  'upper right')
    plt.savefig('comparison_uni_low.eps')
    plt.savefig('comparison_uni_low.png')
    os.system("echo \"file\" | mutt -a \"comparison_uni_low.png\" -s \"Plot\" -- alexsusemihl@gmail.com")
