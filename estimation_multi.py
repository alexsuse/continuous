#!/usr/bin/env python
'''
finds the optimal encoder for the filtering problem.
'''

import cPickle as pic
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


def full_stoc_sigma(sigma0,dt,N,a,eta,alpha,la,NSamples,rands=None, discard=0,D=2):
    sigmas = numpy.zeros((NSamples,N,D,D))
    inva = numpy.linalg.pinv(alpha)
    sigmas[:,0] = sigma0
    eta2 = numpy.dot(eta,eta.T)

    if rands==None:
        rng = numpy.random.RandomState(12345)
        rands = (rng.uniform(size=(NSamples,N))<la*dt).astype('int')
    else:
        assert rands.shape == (NSamples, N)
        rands = (rands<la*dt).astype('int')
    
    for j in xrange(0, discard):
        rand_sample = (rng.uniform(size=(NSamples,1))<la*dt).astype('int')
        dsigma = [numpy.dot(a,sigmas[i,0,:,:])+numpy.dot(sigmas[i,0,:,:],a.T) + eta2 for i in range(NSamples)]
        dsigma = numpy.array(dsigma)
        sigmajump = [numpy.linalg.solve(numpy.eye(D)+numpy.dot(inva,sigmas[i,0,:,:]).T,sigmas[i,0,:,:])
                    for i in range(NSamples)]

        splus1 = numpy.asarray([sigmas[:,0]+dt*dsigma,
                             sigmajump]) 
        sigmas[:,0,:,:] = splus1[rand_sample[:,0],range(NSamples)]
    
    for j in xrange(0,N-1):
        dsigma = [numpy.dot(a,sigmas[i,j,:,:])+numpy.dot(sigmas[i,j,:,:],a.T) + eta2 for i in range(NSamples)]
        dsigma = numpy.array(dsigma)
        sigmajump = [numpy.linalg.solve(numpy.eye(D)+numpy.dot(inva,sigmas[i,j,:,:]).T,sigmas[i,j,:,:])
                     for i in range(NSamples)]

        splus1 = numpy.asarray([sigmas[:,j]+dt*dsigma,
                             sigmajump]) 
        
        sigmas[:,j+1,:,:] = splus1[rands[:,j],range(NSamples)]
    
    return sigmas


if __name__=='__main__':
    gamma = 0.4
    omega = 0.8
    #a = numpy.array([[1.0,1.0],[-omega**2, -gamma]])
    a = numpy.array([[0.0,1.0],[-omega**2,-gamma]])
    eta = .40*numpy.diag([0.0,1.0])
    dtheta = 0.1
    phi = 0.5
    sigma = eta[1,1]**2*numpy.eye(2)
#    eps = numpy.zeros_like(alphas)
    alphas = numpy.arange(0.0,3.0,0.4)
    phis = numpy.arange(0.0,2.0,1.0)
    eps = numpy.zeros((alphas.size,phis.size))
    stoc_eps = numpy.zeros((alphas.size,phis.size))

    try:
        dic = pic.load(open("figure_5_5.pik","r"))
        eps = dic['mean_field']
        stoc_eps = dic['stochastic']
        print "Found Pickle, skipping simulation"

    except:

        phi = 0.1
        N = 10000
        dt = 0.001
        discard = 100
        for n,alpha in enumerate(alphas):
            print n, alphas.size
            for m, phi in enumerate(phis):
                print n,m,alphas.size
                alpha_matrix = numpy.diag([alpha,0.0])
                lamb = phi*numpy.sqrt(2*numpy.pi*alpha)
                eps[n,m] = numpy.trace(get_eq_eps( a, eta, alpha_matrix, lamb ))
                stoc_eps[n,m] =  numpy.trace(full_stoc_sigma(0.01, dt, N, a,
                                               eta, alpha_matrix, lamb, 100,
                                               discard=discard).mean(axis=0).mean(axis=0))

        with open("figure_5_5.pik","w") as fi:
            print "Writing pickle to figure_5_3.pik"
            pic.dump({'alphas':alphas,'phis':phis,'mean_field':eps,'stochastic':stoc_eps},fi)

    print "Plotting..."
    import prettyplotlib as ppl
    ppl.mpl.use('Agg')
    from prettyplotlib import plt
    
    font = {'size':16}
    plt.rc('font',**font)

    fig, (ax1,ax2) = ppl.subplots(1,2,figsize = (18,8))
    
    alphas2,phis2 = numpy.meshgrid(numpy.arange(alphas.min(),alphas.max()+dalpha,dalpha)-dalpha/2,
                                numpy.arange(phis.min(),phis.max()+dphi,dphi)-dphi/2)

    yellorred = brewer2mpl.get_map('YlOrRd','Sequential',9).mpl_colormap

    p = ppl.pcolormesh(fig,ax1,alphas2,phis2,eps.T,cmap = yellorred)
    ax1.axis([alphas2.min(),alphas2.max(),phis2.min(),phis2.max()])

    #xticks = numpy.arange(alphas.min(),alphas.max(),0.5)
    #xlabels = numpy.arange(alphas.min(),alphas.max(),0.5)-alphas.min()

    #yticks = numpy.arange(phis.min(),phis.max(),0.5)
    #ylabels = numpy.arange(phis.min(),phis.max(),0.5)-phis.min()

    #plt.xticks(xticks,xlabels,axes=ax1)
    #plt.yticks(yticks,ylabels,axes=ax1)

    cb = plt.colorbar(p, ax=ax1) 
    cb.set_ticks(numpy.array([0.3,0.4,0.5]))
    cb.set_ticklabels(numpy.array([0.3,0.4,0.5]))

    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$\phi$')
    ax1.set_title(r'MMSE ($\epsilon$)')

    ppl.plot( alphas - 0.001, eps[:,1], label=r'$\phi = '+ str(phis[1]-0.001) + r'$',axes=ax2)
    ppl.plot( alphas[numpy.argmin(eps[:,1])] - 0.001, numpy.min(eps[:,1]), 'bo',axes=ax2)
    ppl.plot( alphas - 0.001, eps[:,2], label=r'$\phi = '+ str(phis[2]-0.001) + r'$',axes=ax2)
    ppl.plot( alphas[numpy.argmin(eps[:,2])] - 0.001, numpy.min(eps[:,2]), 'bo',axes=ax2)
    ppl.plot( alphas - 0.001, eps[:,3], label=r'$\phi = '+ str(phis[3]-0.001) + r'$',axes=ax2)
    ppl.plot( alphas[numpy.argmin(eps[:,3])] - 0.001, numpy.min(eps[:,3]), 'bo',axes=ax2)
    ppl.plot( alphas - 0.001, eps[:,4], label=r'$\phi = '+ str(phis[4]-0.001) + r'$',axes=ax2)
    ppl.plot( alphas[numpy.argmin(eps[:,4])] - 0.001, numpy.min(eps[:,4]), 'bo',axes=ax2)
    ppl.plot( alphas - 0.001, stoc_eps[:,1], '-.', axes=ax2)
    ppl.plot( alphas - 0.001, stoc_eps[:,2], '-.', axes=ax2)
    ppl.plot( alphas - 0.001, stoc_eps[:,3], '-.', axes=ax2)
    ppl.plot( alphas - 0.001, stoc_eps[:,4], '-.', axes=ax2)
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\epsilon$')
    ax2.set_title(r'MMSE as a function of $\alpha$')
    ax2.legend()
    plt.savefig('figure_5_5.png',dpi=300)
    plt.savefig('figure_5_5.eps')
    os.system("echo \"all done\" | mutt -a \"../figures/figure_5_3.eps\" -s \"Plot\" -- alexsusemihl@gmail.com")
