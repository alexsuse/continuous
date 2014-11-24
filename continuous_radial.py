#!/usr/bin/env python
"""
continuous.py -- evaluate the covariance part of the optimal cost-to-go for LQG in continuous time

See git repository alexsuse/Thesis for more information.
"""
import sys
import numpy
##import matplotlib
##matplotlib.use('Agg')
##from matplotlib import rc
##import matplotlib.pyplot as plt
from estimation_multi import get_eq_eps, d_eps_dt, get_eq_kalman, d_eps_kalman
from IPython.parallel import Client
import prettyplotlib as ppl
from prettyplotlib import plt

"""
Parameters, really ugly, well...
"""
T = 2
dt = 0.001
q = numpy.array([[0.2,0.0],[0.0,0.1]]) #running state cost
QT = 0.0*q #final state cost
R = numpy.array([[0.1,0.0],[0.0,0.2]]) #running control cost
eta = .6*numpy.eye(2) #system noise
a = -0.4*numpy.eye(2) #system regenerative force
b = 1.0*numpy.eye(2) #control constant
dtheta = 0.1 #neuron spacing
phi = 0.5 #neuron maximal rate


def solve_riccatti(N,dt,QT,a,b,q,r):
    """
    As the name suggests, solve_riccatti
    solves the riccatti matrix equation
    for the LGQ control problem.
    arguments:
    N    :: number of time steps
    dt   :: time increment
    QT   :: final state cost
    a    :: system dynamics parameter
    b    :: control parameter
    q    :: instantaneous state cost
    r    :: instantaneous control cost
    """
    s = numpy.zeros((N,2,2))
    s[-1] = QT
    for i in range(N-1):
        Sa = numpy.dot( s[N-i-1], a)
        Sb = numpy.dot( s[N-i-1], b.T)
        s[N-i-2] = s[N-i-1]+dt*(- numpy.dot( Sb, numpy.linalg.solve( R, Sb.T ) ) + Sa + Sa.T + 0.5 * q ) 
    return s

def kalman_f(sigma0,S,dt,a, eta ,alpha,b,q,r):
    """
    mf_f computes the mean-field approximation
    to the variance component of the control cost
    given by the function f. The value of f is given
    by tr(s(0) sigma0 ) + int tr[s(y) b R^-1(y) b s(y) sigma(y)] dy
    arguments:
    sigma0 :: initial value of sigma at t=0
    S      :: solution of the riccatti equation
    dt     :: time increment
    a      :: parameter of system dynamics
    eta    :: sqrt of noise covariance
    alpha  :: tuning matrix of neurons
    b      :: control parameter of system
    q      :: state-cost parameters
    r      :: control-cost parameters
    la     :: population firing rate
    """
    f = numpy.trace( numpy.dot( sigma0, S[0] ) )

    sigmas = kalman_sigma( sigma0, dt, S.size, a, eta, alpha )

    bs = numpy.dot( S, b.T)

    integral = numpy.trace( numpy.sum( [ numpy.dot( bss, numpy.linalg.solve( R, numpy.dot( bss.T, sigmas[i] ) ) ) for i,bss in enumerate(bs) ]  , axis = 0) )
    
    f += dt * integral
    
    return f

def mf_f(sigma0,S,dt,a, eta ,alpha,b,q,r,la):
    """
    mf_f computes the mean-field approximation
    to the variance component of the control cost
    given by the function f. The value of f is given
    by tr(s(0) sigma0 ) + int tr[s(y) b R^-1(y) b s(y) sigma(y)] dy
    arguments:
    sigma0 :: initial value of sigma at t=0
    S      :: solution of the riccatti equation
    dt     :: time increment
    a      :: parameter of system dynamics
    eta    :: sqrt of noise covariance
    alpha  :: tuning matrix of neurons
    b      :: control parameter of system
    q      :: state-cost parameters
    r      :: control-cost parameters
    la     :: population firing rate
    """
    f = numpy.trace( numpy.dot( sigma0, S[0] ) )

    sigmas = mf_sigma( sigma0, dt, S.size, a, eta, alpha, la )

    bs = numpy.dot( S, b.T)

    integral = numpy.trace( numpy.sum( [ numpy.dot( bss, numpy.linalg.solve( R, numpy.dot( bss.T, sigmas[i] ) ) ) for i,bss in enumerate(bs) ]  , axis = 0) )
    
    f += dt * integral
    
    return f


def kalman_sigma(sigma0, dt, N, a , eta, alpha):
    """
    mf_sigma computes the expected value of sigma
    under the mean-field approximation given hte parameters.
    system dynamics is given by
    dx = a x dt + eta dW
    arguments are
    sigma0 :: initial value of sigma at t=0
    dt     :: time increment
    N      :: number of time intervals
    a      :: parameter of system dynamics
    eta    :: sqrt of covariance matrix
    alpha  :: width of tuninig functinos
    la     :: population firing rate
    """
    s = numpy.zeros((N,2,2))
    s[0] = sigma0
    eta2 = numpy.dot( eta, eta.T )
    for i in xrange(1,N):
        sigma_a = numpy.dot( s[i-1] , a )
        obs_term = numpy.dot(s[i-1],numpy.linalg.solve( alpha, s[i-1] ))
        s[i] =s[i-1]+dt*( sigma_a + sigma_a.T + eta2 -obs_term )
    return s

def mf_sigma(sigma0, dt, N, a , eta, alpha, la):
    """
    mf_sigma computes the expected value of sigma
    under the mean-field approximation given hte parameters.
    system dynamics is given by
    dx = a x dt + eta dW
    arguments are
    sigma0 :: initial value of sigma at t=0
    dt     :: time increment
    N      :: number of time intervals
    a      :: parameter of system dynamics
    eta    :: sqrt of covariance matrix
    alpha  :: width of tuninig functinos
    la     :: population firing rate
    """
    s = numpy.zeros((N,2,2))
    s[0] = sigma0
    eta2 = numpy.dot( eta, eta.T )
    for i in xrange(1,N):
        sigma_a = numpy.dot( s[i-1] , a )
        jump_term = numpy.dot(s[i-1],numpy.linalg.solve( s[i-1] + alpha, s[i-1] ))
        s[i] =s[i-1]+dt*( sigma_a + sigma_a.T + eta2 -la*jump_term )
    return s

def full_stoc_sigma(sigma0, dt, N, a, eta, alpha, la, NSamples, rands=None):
    """
    Full stochastic sigma expectation.
    Computes NSamples samples of the covariance dynamics nd return the mean
    args:
    sigma0 :: initial value of sigma
    dt     :: time increment
    N      :: number of time steps
    a      :: regenerative force
    eta    :: sqroot of the noise covariance
    alpha  :: tuning matrix
    la     :: population firing rate
    NSamples:: number of samples
    rands  :: precomputed random numbers (optional)
    """
    sigmas = numpy.zeros((N, NSamples , 2, 2))

    sigmas[0] = sigma0

    if rands is None:
        rng = numpy.random.RandomState(12345)
        rands = ( rng.uniform( size=(N, NSamples ) ) < la*dt ).astype('int')

    else:
        assert rands.shape == (N,NSamples)
        rands = (rands<la*dt).astype('int')

    eta2 = numpy.dot( eta, eta.T )

    for i in xrange(1,N):
        asigmas = numpy.dot( sigmas[i-1], a) 
        nojump = asigmas + asigmas.swapaxes( 1, 2 ) + eta2
        jump = [ numpy.dot(si,numpy.linalg.solve( si + alpha, alpha )) for si in sigmas[i-1]]
        splus1 = numpy.asarray( [ sigmas[i-1]+dt*nojump, jump] )
        sigmas[i] = splus1[ rands[i], range( NSamples ) ]

    return numpy.mean( sigmas, axis=1 )


def full_stoc_f(sigma0, S, dt, a, eta, alpha, b, q, r, la, NSamples,rands=None):
    """
    Computes the full stochastic version of covariance
    component of the control cost. It computes the expected
    value of the covariance and then uses the same formula as 
    mf_f. args
    sigma0 :: initial value of sigma
    S      :: solution of the ricatti equation
    dt     :: time increment
    a      :: regenerative force matrix
    eta    :: sqrt of noise covariance matrix
    alpha  :: tuning matrix
    b      :: control force matrix
    q      :: instantaneous state costs
    r      :: instantaneous control costs
    la     :: population firing rate
    NSamples :: number of samples
    rands  :: precomputed random nubers (optional)
    """
    f = numpy.trace( numpy.dot( sigma0, S[0] ) )
    
    sigmas = full_stoc_sigma( sigma0, dt, N, a, eta, alpha, la, NSamples, rands )
   
    bs = numpy.dot( S, b )
    
    integral = numpy.trace( numpy.sum( [ numpy.dot( bss, numpy.linalg.solve( R, numpy.dot( bss.T, sigmas[i] ) ) ) for i,bss in enumerate(bs) ], axis = 0  ) )

    f += dt * integral
    return f

if __name__=='__main__':
    print __doc__


    N = int(T/dt)
    #precompute solution to the Ricatti equation    
    S = solve_riccatti(N,dt,QT,a,b,q,R)

    #range of covariance matrices evaluated
    thetas = numpy.arange(0.05,numpy.pi/3,.03)

    #initial sigma value
    s = 2.0*numpy.eye(2)

    #preallocating numpy vectors for better performance
    est_eps = numpy.zeros_like( thetas )
    k_est_eps = numpy.zeros_like( thetas )
    fs = numpy.zeros_like( thetas )
    full_fs =  numpy.zeros_like( thetas )
    k_cont_fs = numpy.zeros_like( thetas )
    #estimation_eps = numpy.zeros_like(alphas)
    NSamples = 500

    radial = lambda t :  0.1*numpy.diag([numpy.tan(t), 1.0/numpy.tan(t)])
    la = numpy.sqrt((2*numpy.pi)**2*numpy.linalg.det(radial(thetas[0])))*phi/(dtheta**2)
    estimation = lambda (n,t) : (n, numpy.trace( get_eq_eps( a, eta, radial(t), la )))
    mean_field = lambda (n,t) : (n,mf_f(s,S,dt,a,eta,radial(t),b,q,R,la))
    full_stoc = lambda (n,t) : (n,full_stoc_f(s,S,dt,a,eta,radial(t),b,q,R,la,NSamples,rands=rands))
    k_estimation = lambda (n,t) : (n, numpy.trace( get_eq_kalman( a, eta, radial(t) ) ) )
    k_control = lambda (n,t) : (n, kalman_f(s, S, dt, a, eta, radial(t), b, q, R ) )

    print 'running '+str(thetas.shape[0])+' runs'
    rands = numpy.random.uniform(size=(N,NSamples))

    args = []
    for i,t in enumerate(thetas):
        
        args.append((i,t)) 

    try:
        c = Client()
        dview = c[:]

        with dview.sync_imports():
            import numpy
            import scipy
            import scipy.optimize
        #dview = c.load_balanced_view()

        print 'all good with parallel'

        mymap = lambda (f,args) : dview.map_async(f, args )

        dview.push({'radial':radial})
        dview.push({'get_eq_eps':get_eq_eps,'d_eps_dt':d_eps_dt,
                    'get_eq_kalman':get_eq_kalman,'d_eps_kalman':d_eps_kalman})
        dview.push({'mf_f':mf_f,'full_stoc_f':full_stoc_f,'s':s,'S':S,'a':a,'N':N,'la':la,
                    'eta':eta,'b':b,'q':q,'R':R,'NSamples':NSamples,'rands':rands,'dt':dt})
        dview.push({'mf_sigma':mf_sigma,'full_stoc_sigma':full_stoc_sigma,'estimation':estimation,
                    'kalman_sigma':kalman_sigma,'kalman_f':kalman_f})

    except:
        mymap = lambda (f,args): map(f,args)
    #est_calls = dview.map_async( estimation, args, ordered=False )
    #mf_calls = dview.map_async( mean_field, args, ordered=False )
    #full_calls = dview.map_async( full_stoc, args, ordered=False )
    
    est_calls = mymap((estimation, args))
    print "estimation done"
    k_est_calls = mymap((k_estimation, args))
    print "kalman estimation"
    mf_calls  = mymap((mean_field, args))
    print "mean field done"
    full_calls = mymap((full_stoc, args))
    print "stochastic done"
    k_control_calls = mymap((k_control, args ))

    gotten = []
    for n,res in enumerate(k_est_calls):
        n,v = res
        gotten.append(n)
        print 'Kalman %d entries in, %d'%(len(gotten),n)
        k_est_eps[n] = v
    print 'kalman estimation is in'

    gotten = []
    for n,res in enumerate(est_calls):
        n,v = res
        gotten.append(n)
        print 'EST %d entries in, %d'%(len(gotten),n)
        est_eps[n] = v
    print 'estimation is in'
    
    gotten = []
    for n,res in enumerate(mf_calls):
        n,v = res
        gotten.append(n)
        print 'MF %d entries in, %d'%(len(gotten),n)
        fs[n] = v
    print 'mean field is in'
    
    gotten = []
    for n,res in enumerate(full_calls):
        n,v = res
        gotten.append(n)
        print 'Stochastic %d entries in, %d'%(len(gotten),n)
        full_fs[n] = v
    print 'stochastic is in too'

    gotten = []
    for n,res in enumerate(k_control_calls):
        n,v = res
        gotten.append(n)
        print 'LQG %d entries in, %d'%(len(gotten),n)
        k_cont_fs[n] = v
    print 'kalman control is in'

    
    fullmin,indfull = (numpy.min(full_fs),numpy.argmin(full_fs))
    mfmin,mfind = (numpy.min(fs),numpy.argmin(fs))
    epsmin,epsind = (numpy.min(est_eps),numpy.argmin(est_eps))
    kmin, kind = (numpy.min(k_est_eps),numpy.argmin(k_est_eps))
    lqgmin, lqgind = (numpy.min(k_cont_fs),numpy.argmin(k_cont_fs))

    #rc('text',usetex=True)

    fig, (ax1,ax2) = ppl.subplots(2,1,sharex=True)

    l1, = ppl.plot(thetas, est_eps, label='Point Process Filtering', ax=ax1)
    l2, = ppl.plot(thetas, k_est_eps,label = 'Kalman Filtering',ax=ax1)
    ppl.plot(thetas[epsind],epsmin,'o',color = l1.get_color(), ax=ax1)
    ppl.plot(thetas[kind],kmin,'o',color = l2.get_color(), ax=ax1)

    #ax1.text(thetas[2],0.16,'a)')

    l3, = ppl.plot(thetas, fs, label = 'Point Process Control (MF)', ax=ax2)
    l4, = ppl.plot(thetas, full_fs, label='Point Process Control (simulated)', ax=ax2)
    l5, = ppl.plot(thetas, k_cont_fs, label='LQG Control', ax=ax2 )
    ppl.plot(thetas[mfind],mfmin,'o',color=l3.get_color(), ax=ax2)
    ppl.plot(thetas[indfull], fullmin, 'o', color=l4.get_color(), ax=ax2)
    ppl.plot(thetas[lqgind], lqgmin, 'o', color=l5.get_color(), ax=ax2)

    print "lqg",lqgind,thetas[lqgind],lqgmin

    #ax1.spines['bottom'].set_visible(False)
    #ax2.spines['top'].set_visible(False)
    #ax1.tick_params(axis='x',which='both',bottom='off')
    #ax2.tick_params(axis='x',which='both',top='off')

    #ax1.set_ylabel(r'$MMSE$')
    #ax2.set_ylabel(r'$f(\Sigma_0,t_0)$')
    #ax2.set_xlabel(r'$\zeta$')
    ppl.legend(ax1,loc=4).get_frame().set_alpha(0.7)
    ppl.legend(ax2,loc=4).get_frame().set_alpha(0.7)


    #plt.figlegend([l1,l2,l3,l4,l5],['Poisson MMSE','Kalman MMSE',r'Mean Field $f$',r'Stochastic $f$',r'LQG $f$'],'upper right')
    print "Saving figure to "+sys.argv[1]+".png"
    plt.savefig(sys.argv[1]+'.pdf')
    plt.savefig(sys.argv[1]+'.png',dpi=200)
    plt.savefig(sys.argv[1]+'.eps')

    print 'eps-optimal', radial(thetas[epsind])
    print 'cont-optimal', radial(thetas[indfull])
