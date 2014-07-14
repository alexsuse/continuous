#!/usr/bin/python

import numpy as np
import sys
import poissonneuron as pn
import gaussianenv as ge
import prettyplotlib as ppl
from prettyplotlib import plt
#import matplotlib.pyplot as plt
#from matplotlib import cm

cm = ppl.mpl.cm

def getMaternSample( gamma = 1.0, eta = 1.0, order = 2, alpha = 0.1, phi = 1.0, dtheta = 0.2, dt = 0.001, repetitions = 100, timewindow = 10000, spacesteps = 400, plot = False, sample = None,outname = 'OU_coding', ax=None, label=''):
    zeta = 2
    L = 0.8
    N = 1
    a = np.zeros(N*N)
    sigma = 0.001

    if plot and ax is not None:
        repetitions=1


    e = ge.GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma,order)
    gam = e.getgamma()
    print gam
    
    et = e.geteta()
    abar  = np.sqrt(2.0*np.pi)*alpha*phi/dtheta
    
    code = pn.PoissonCode(np.arange(-4.0,4.0,dtheta),alpha,phi)
    space = np.arange(-4.0,4.0,8.0/spacesteps)
    weight = 1.0/repetitions
    sigmaavg = np.zeros((timewindow,order,order))
    sigmaeq = np.zeros((timewindow,order,order))
    sigmamf = np.zeros((timewindow,order,order))
    sigmaeq[-1,:,:] = 0.001*np.eye(order)
    for i in range(timewindow):
        sigmaeq[i,:,:] = sigmaeq[i-1,:,:] - dt*(np.dot(gam,sigmaeq[i-1,:,:])+np.dot(sigmaeq[i-1,:,:],gam.T)-et)
    for i in range(timewindow):
        sigmaeq[i,:,:] = sigmaeq[i-1,:,:] - dt*(np.dot(gam,sigmaeq[i-1,:,:])+np.dot(sigmaeq[i-1,:,:],gam.T)-et)
    sigmamf[-1,:,:] = sigmaeq[-1,:,:]
    for i in range(timewindow):
        sigmamf[i,:,:] = sigmamf[i-1,:,:] - dt*(np.dot(gam,sigmamf[i-1,:,:])+np.dot(sigmamf[i-1,:,:],gam.T)-et) - dt*abar*np.dot(np.array([sigmamf[i-1,:,0]]).T,np.array([sigmamf[i-1,:,0]]))/(alpha**2+sigmamf[i-1,0,0])
    for k in range(repetitions):
        mu = np.zeros((timewindow,order))
        stim = np.zeros((timewindow,order))
        sigma = np.zeros((timewindow,order,order))
        sigma[-1,:,:] = sigmaeq[-1,:,:]
        sigmanew = np.zeros((order,order))
        P = np.zeros((spacesteps,timewindow))
        spcount = 0
        Astar = np.zeros((order,order))
        Astar[0,0] = 1.0/alpha**2
        spikers = []
        times = []
        for i in range(timewindow):
            print "run %d of %d, time %d of %d"%(k,repetitions,i,timewindow)
            s = e.samplestep(dt).ravel()
            stim[i,:] = s
            spi = code.spikes(s[0],dt)
            if sum(spi)>=1:
                spcount +=1
                ids = np.where(spi==1)
                thet = np.zeros_like(mu[i,:])
                thet[0] = code.neurons[ids[0]].theta[0]
                spikers.append(thet[0])
                times.append(i*dt)
                sigma[i,:,:] = sigma[i-1,:,:] - np.dot(np.array([sigma[i-1,:,0]]).T,np.array([sigma[i-1,:,0]]))/(alpha**2+sigma[i-1,0,0])
                mu[i,:] = np.linalg.solve(np.identity(order)+np.dot(sigma[i-1,:,:],Astar),mu[i-1,:]+np.dot(sigma[i-1,:,:],np.dot(Astar,thet)))
            else:
                mu[i,:] = mu[i-1,:] - dt*np.dot(gam,mu[i-1,:])
                sigma[i,:,:] = sigma[i-1,:,:] - dt*(np.dot(gam,sigma[i-1,:,:])+np.dot(sigma[i-1,:,:],gam.T)-et)
        sigmaavg = sigmaavg + sigma*weight
        print "Run", k, "Firing rate was ", np.float(spcount)/(timewindow*dt), "abar is ", abar
    
    for i in range(timewindow):
        P[:,i] = np.exp(-(space-mu[i,0])**2/(2.0*sigma[i,0,0]))/(np.sqrt(2.0*np.pi*sigma[i,0,0]))        

    if plot == True:

        if ax==None:
            plt.rc('text',usetex=True)
            fig, (ax1,ax2) = ppl.subplots(2,1)
            ts = np.arange(0.0,dt*timewindow,dt)
            mline, = ppl.plot(ts,mu[:,0],label='Reconstruction',ax=ax1)
            rline, = ppl.plot( ts,stim[:,0],label='Stimulus',ax=ax1)
            sps, = ppl.plot( times,spikers,'o',label='Spikes',ax=ax1)
            ax1.set_title('Second Order OU Process')
            ax1.set_ylabel(r'Position [cm]')
            #ax.imshow(P,extent=[0,ts[-1],4.0,-4.0],aspect='auto',cmap=cm.gist_yarg)
            leg = ppl.legend(ax1)
            frame = leg.get_frame()
            frame.set_alpha(0.6)
            p = ppl.fill_between(ts, mu[:,0]-np.sqrt(sigma[:,0,0]), mu[:,0]+np.sqrt(sigma[:,0,0]),alpha=0.3,ax=ax1)
            samp, = ppl.plot(ts,sigma[:,0,0],label='Sample Variance',ax=ax2)
            mfline, = ppl.plot( ts,sigmamf[:,0,0], label='Mean-Field Variance',ax=ax2)
            avgline, = ppl.plot( ts,sigmaavg[:,0,0], label='Average Variance (MSE)',ax=ax2)
            eqline, = ppl.plot(ts, sigmaeq[:,0,0],label='Stationary Variance',ax=ax2)
            ax2.set_title('Dynamics of the Posterior Variance')
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel(r'Posterior Variance [cm$^2$]')
            ax2.set_ylim([-0.01,0.32])
            leg = ppl.legend(ax2)
            frame = leg.get_frame()
            frame.set_alpha(0.6)
            plt.savefig(outname+'.eps')
            plt.savefig(outname+'.pdf')
            plt.savefig(outname+'.png',dpi=300)
        else:
            plt.rc('text',usetex=True)
            ts = np.arange(0.0,dt*timewindow,dt)
            rline, = ppl.plot( ts,stim[:,0],ax=ax)
            mline, = ppl.plot(ts,mu[:,0],ax=ax)
            sps, = ppl.plot(times,spikers,'o',ax=ax)
            ppl.fill_between(ts, mu[:,0]-np.sqrt(sigma[:,0,0]), mu[:,0]+np.sqrt(sigma[:,0,0]),alpha=0.2,ax = ax)
            ax.set_ylabel(r'Position [cm]')

    return [P,sigmaavg, sigma, sigmamf,sigmaeq]


def getMaternPredError(window = 100, gamma = 1.0, eta = 1.0, order = 2, alpha = 0.1, phi = 2.0, dtheta = 0.3, dt = 0.001, repetitions = 100, timewindow = 10000, plot = False):
    zeta = 2
    L = 0.8
    N = 1
    a = np.zeros(N*N)
    sigma = 0.001
    interval = window
    normalizer = 0
    prederror = np.zeros((window,order,order))
    predictor = np.zeros((window,order))
    e = ge.GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma,order)
    gam = e.getgamma()
    et = e.geteta()
    abar  = np.sqrt(2.0*np.pi)*alpha*phi/dtheta
    code = pn.PoissonCode(np.arange(-4.0,4.0,dtheta),alpha,phi)
    weight = 1.0/repetitions
    sigmaavg = np.zeros((timewindow,order,order))
    sigmamf = np.zeros((timewindow,order,order))
    sigmamf[-1,:,:] = np.ones_like(sigmamf[-1,:,:])
    for i in range(timewindow):
        sigmamf[i,:,:] = sigmamf[i-1,:,:] - dt*(np.dot(gam,sigmamf[i-1,:,:])+np.dot(sigmamf[i-1,:,:],gam.T)-et) - dt*abar*np.dot(np.array([sigmamf[i-1,:,0]]).T,np.array([sigmamf[i-1,:,0]]))/(alpha**2+sigmamf[i-1,0,0])
    for k in range(repetitions):
        mu = np.zeros((timewindow,order))
        s = e.samplestep(dt).ravel()
        mu[-1,:] = s
        stim = np.zeros((timewindow,order))
        sigma = np.zeros((timewindow,order,order))
        sigma[-1,:,:] = sigmamf[-1,:,:]
        sigmanew = np.zeros((order,order))
        spcount = 0
        Astar = np.zeros((order,order))
        Astar[0,0] = 1.0/alpha**2
        for i in range(timewindow):
            print "run %d of %d, time %d of %d"%(k,repetitions,i,timewindow)
            s = e.samplestep(dt).ravel()
            stim[i,:] = s
            spi = code.spikes(s[0],dt)
            if sum(spi)>=1:
                spcount +=1
                ids = np.where(spi==1)
                thet = np.zeros_like(mu[i,:])
                thet[0] = code.neurons[ids[0]].theta[0]
                sigma[i,:,:] = sigma[i-1,:,:] - np.dot(np.array([sigma[i-1,:,0]]).T,np.array([sigma[i-1,:,0]]))/(alpha**2+sigma[i-1,0,0])
                mu[i,:] = np.linalg.solve(np.identity(order)+np.dot(sigma[i-1,:,:],Astar),mu[i-1,:]+np.dot(sigma[i-1,:,:],np.dot(Astar,thet)))
            else:
                mu[i,:] = mu[i-1,:] - dt*np.dot(gam,mu[i-1,:])
                sigma[i,:,:] = sigma[i-1,:,:] - dt*(np.dot(gam,sigma[i-1,:,:])+np.dot(sigma[i-1,:,:],gam.T)-et)
            if i>window:
                if i%interval==interval-1:
                    normalizer +=1
                    predictor[0,:] = mu[i-window,:]
                    for j in range(1,window):
                        predictor[j,:] = predictor[j-1,:]-dt*np.dot(gam,predictor[j-1,:])
                    for j in range(window):
                        x = predictor[j,:]-stim[i-window+j,:]
                        prederror[j,:,:] += np.dot(np.array([x]).T,np.array([x]))
        sigmaavg = sigmaavg + sigma*weight
        print "Run", k, "Firing rate was ", np.float(spcount)/(timewindow*dt), "abar is ", abar
    prederror = prederror/normalizer
    sigma_theory = np.zeros((window,order,order))
    sigma_theory[0,:,:] = prederror[0,:,:]
    for i in range(1,window):
        sigma_theory[i,:,:] = sigma_theory[i-1,:,:]  - dt*(np.dot(gam,sigma_theory[i-1,:,:])+np.dot(sigma_theory[i-1,:,:],gam.T)-et)
    
    return [prederror,sigma_theory,sigma,predictor,stim,mu]

def getMaternEqVariance( gamma = 1.0, eta = 1.0, order = 2, alpha = 0.2, phi = 1.3, dtheta = 0.3, dt = 0.001, samples = 100, timewindow = 10000, spacesteps = 400, plot = False, Trelax = 1, histmax = 0.8 ):
    zeta = 2
    L = 0.8
    N = 1
    a = np.zeros(N*N)
    sigma = 0.001
    e = ge.GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma,order)
    gam = e.getgamma()
    et = e.geteta()
    abar  = np.sqrt(2.0*np.pi)*alpha*phi/dtheta
    code = pn.PoissonCode(np.arange(-4.0,4.0,dtheta),alpha,phi)
    space = np.arange(-4.0,4.0,8.0/spacesteps)
    sigmaavg = np.zeros((samples,order,order))
    

    mu = np.zeros((order))
    sigma = np.zeros((order,order))
    sigmamf = np.zeros((order,order))
    sigma[:,:] = 0.001*np.eye(order)
    sigmamf[:,:] = 0.001*np.eye(order)
    sigmanew = np.zeros((order,order))
    for i in range(timewindow):
        s = e.samplestep(dt).ravel()
        spi = code.spikes(s[0],dt)
        if sum(spi)>=1:
            #lam = np.linalg.inv(sigma[:,:])
            #ids = np.where(spi==1)
            #thet = np.zeros_like(mu[:])
            #thet[0] = code.neurons[ids[0]].theta[0]
            sigma = sigma - np.dot(np.array([sigma[:,0]]).T,np.array([sigma[:,0]]))/(alpha**2+sigma[0,0])
            #mu = np.dot(sigma,np.dot(lam,mu)+thet/alpha**2)    
        else:
            #mu = mu - dt*np.dot(gam,mu)
            sigma = sigma - dt*(np.dot(gam,sigma)+np.dot(sigma,gam.T)-et)
    
    sample_interval = 3
    for i in range(sample_interval*samples):
        s = e.samplestep(dt).ravel()
        spi = code.spikes(s[0],dt)
        if sum(spi)>=1:
            #lam = np.linalg.inv(sigma[:,:])
            #ids = np.where(spi==1)
            #thet = np.zeros_like(mu[:])
            #thet[0] = code.neurons[ids[0]].theta[0]
            #mu = np.dot(sigma,np.dot(lam,mu)+thet/alpha**2)    
            sigma = sigma - np.dot(np.array([sigma[:,0]]).T,np.array([sigma[:,0]]))/(alpha**2+sigma[0,0])
        else:
            #mu = mu - dt*np.dot(gam,mu)
            sigma = sigma - dt*(np.dot(gam,sigma)+np.dot(sigma,gam.T)-et)
        if i%sample_interval == 0:
            sigmaavg[i/sample_interval,:,:] = sigma[:,:]

    (vers,subvers,_,_,_) = sys.version_info
    if subvers>=7:
        [freqs,xs] = np.histogram(sigmaavg[:,0,0], bins = np.arange(0.0,histmax,histmax/80),normed = True)#, new = True)
    else:
        [freqs,xs] = np.histogram(sigmaavg[:,0,0], bins = np.arange(0.0,histmax,histmax/80),normed = True, new = True)
    xs = 0.5*(xs[0:-1]+xs[1:])
    sigma = np.average(sigmaavg,axis=0)

    for i in range(timewindow):
                sigmamf = sigmamf + dt*(et - np.dot(gam,sigmamf) - np.dot(sigmamf,gam.T)) -dt*abar*np.dot(np.array([sigmamf[:,0]]).T,np.array([sigmamf[:,0]]))/(alpha**2+sigmamf[0,0])
    for i in range(timewindow):
        sigmanew = sigmanew + dt*(et - np.dot(gam,sigmanew) - np.dot(sigmanew,gam.T))
    
    return [sigma, sigmamf, sigmanew,  xs, freqs]

if __name__=='__main__':
    getMaternSample(plot=True,outname='test_coding_plot',repetitions=100,timewindow=10000)