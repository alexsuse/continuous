#!/usr/bin/env python
"""
continuous.py -- evaluate the covariance part of the optimal cost-to-go for LQG in continuous time

See git repository alexsuse/Thesis for more information.
"""
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import estimation as est

T = 2
dt = 0.001
q = 0.01*np.eye(2) #running state cost
QT = 0.1*np.eye(2) #final state cost
R = 0.1*np.eye(2) #running control cost
eta = 1.0*np.eye(2) #system noise
a = -0.1*np.eye(2) #system regenerative force
b = 0.2*np.eye(2) #control constant
alpha = 0.1*np.eye(2) #observation noise
dtheta = 0.05 #neuron spacing
phi = 0.01 #neuron maximal rate

def solve_riccatti(N,dt,QT,a,b,q,r):
    s = np.zeros((N,2,2))
    s[-1] = QT
    for i in range(N-1):
        Sa = np.dot( s[N-i-1], a)
        Sb = np.dot( s[N-i-1], b)
        s[N-i-2] = s[N-i-1]+dt*(- np.dot( Sb, np.linalg.solve( R, Sb.T ) ) + Sa + Sa.T + 0.5 * q ) 
    return s

def mf_f(sigma0,S,dt,a, eta ,alpha,b,q,r,la):
    f = np.trace( np.dot( sigma0, S[0] ) )

    sigmas = mf_sigma( sigma0, dt, S.size, a, eta, alpha, la )

    bs = np.dot( S, b)

    integral = np.sum( [ np.dot( bss, np.linalg.solve( R, np.dot( bss, sigmas[i] ) ) ) for i,bss in enumerate(bs) ]  , axis = 0)
    
    integral = np.trace( integral )

    f += dt * integral
    
    return f


def mf_sigma(sigma0, dt, N, a , eta, alpha, la):
    s = np.zeros((N,2,2))
    s[0] = sigma0
    eta2 = np.dot( eta, eta.T )
    for i in range(N)[1:]:
        sigma_a = np.dot( s[i-1] , a )
        s2 = np.dot( s[i-1], s[i-1] )
        jump_term = np.linalg.solve( s[i-1] + alpha, s2 )
        s[i] =s[i-1]+dt*( sigma_a + sigma_a.T + eta2 -la*jump_term )
    return s

def full_stoc_sigma(sigma0, dt, N, a, eta, alpha, la, NSamples, rands=None):
   
    sigmas = np.zeros((N, NSamples , 2, 2))

    sigmas[0] = sigma0

    if rands is None:
        rng = np.random.RandomState(12345)
        rands = ( rng.uniform( size=(N, Nsamples ) ) < la*dt ).astype('int')

    else:
        rands = (rands<la*dt).astype('int')

    eta2 = np.dot( eta, eta.T )

    for i in xrange(1,N):
        asigmas = np.dot( sigmas[i-1], a) 
        nojump = asigmas + asigmas.swapaxes( 1, 2 ) + eta2
        jump = [ np.linalg.solve( si + alpha, np.dot( si, si ) ) for si in sigmas[i-1]]
        splus1 = np.asarray( [ sigmas[i-1]+dt*nojump, jump] )
        sigmas[i] = splus1[ rands[i], range( NSamples ) ]

    return np.mean( sigmas, axis=1 )


def full_stoc_f(sigma0,S,dt,a,sigma,alpha,b,q,r,la,NSamples,rands=None):
    f = np.trace( np.dot( sigma0, S[0] ) )
    
    sigmas = full_stoc_sigma( sigma0, dt, N, a, sigma, alpha, la, NSamples, rands )
   
    bs = np.dot( S, b )
    
    integral = np.trace( np.sum( [ np.dot( bss, np.linalg.solve( R, np.dot( bss, sigmas[i] ) ) ) for i,bss in enumerate(bs) ], axis = 0  ) )

    f += dt * integral
    return f

if __name__=='__main__':
    N = int(T/dt)
    S = solve_riccatti(N,dt,QT,a,b,q,R)
    alphas = np.arange(0.001,4.0,0.01)
    s = 2.0*np.eye(2)
    fs = np.zeros_like(alphas)
    full_fs = np.zeros_like(alphas)
    estimation_eps = np.zeros_like(alphas)
    Nsamples = 100
    print 'running '+str(alphas.size)+' runs'
    rands = np.random.uniform(size=(N,Nsamples))
    for i,al in enumerate(alphas):
        alpha = al*np.eye(2)
        print '---->\n.....Alpha = %s' % str(al) 
        la = np.sqrt(2*np.pi*np.linalg.det(alpha)**2)*phi/dtheta
        fs[i] = mf_f(s,S,dt,a,eta,alpha,b,q,R,la)
        print '---->\n.....Mean field %s' % str(fs[i])
        full_fs[i] = full_stoc_f(s,S,dt,a,eta,alpha,b,q,R,la,Nsamples,rands=rands)
        print '---->\n.....Stochastic %s' % str(full_fs[i])
    #    estimation_eps[i] = est.get_eq_eps(-a,eta,alpha,la)
    #    print '---->\n.....Average filtering error %s' % str(estimation_eps[i])
    fsmin,indfs = (np.min(fs),np.argmin(fs))
    fullmin,indfull = (np.min(full_fs),np.argmin(full_fs))
    #epsmin,indeps = (np.min(estimation_eps),np.argmin(full_fs))
    rc('text',usetex='true')
    #estimation_eps = np.max(full_fs)*estimation_eps/np.max(estimation_eps)
    plt.plot(alphas,fs,alphas,full_fs,alphas[indfs],fsmin,'o',alphas[indfull],fullmin,'o')
    plt.legend(['Mean Field', 'Stochastic Average'])#,'Filtering Error'])^
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$f(\Sigma)$')
    plt.show()
    #plt.plot(alphas,fs,alphas,full_fs,alphas,estimation_eps,alphas[indfs],fsmin,'o',alphas[indfull],fullmin,'o',alphas[indeps],epsmin,'o')
    #plt.legend(['Mean Field', 'Stochastic Average','Filtering Error'])
    #plt.xlabel(r'$\alpha$')
    #plt.ylabel(r'$f(\Sigma)$')
    try:
        a = raw_input('file name?')
        print 'saving to %s'%a
        plt.savefig(a)
    except SyntaxError,NameError:
        print 'saving to params.png'
        plt.savefig('params.png',dpi=300)
    plt.show()
    #print "don't run this as a script, come on!"
