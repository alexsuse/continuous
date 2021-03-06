#!/usr/bin/env python
"""
continuous.py -- evaluate the covariance part of the optimal cost-to-go for LQG in continuous time

See git repository alexsuse/Thesis for more information.
"""
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt
import estimation as est
from IPython.parallel import Client

"""
Parameters, really ugly, well...
"""
T = 2
dt = 0.001
q = numpy.array([[0.04,0.0],[0.0,0.01]]) #running state cost
QT = 0.1*numpy.eye(2) #final state cost
R = numpy.array([[0.01,0.0],[0.0,0.04]]) #running control cost
eta = .4*numpy.eye(2) #system noise
a = -0.1*numpy.eye(2) #system regenerative force
b = 0.2*numpy.eye(2) #control constant
alpha = 0.1*numpy.eye(2) #observation noise
dtheta = 0.05 #neuron spacing
phi = 2.54 #neuron maximal rate


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
        Sb = numpy.dot( s[N-i-1], b)
        s[N-i-2] = s[N-i-1]+dt*(- numpy.dot( Sb, numpy.linalg.solve( R, Sb.T ) ) + Sa + Sa.T + 0.5 * q ) 
    return s

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

    bs = numpy.dot( S, b)

    integral = numpy.trace( numpy.sum( [ numpy.dot( bss, numpy.linalg.solve( R, numpy.dot( bss, sigmas[i] ) ) ) for i,bss in enumerate(bs) ]  , axis = 0) )
    
    f += dt * integral
    
    return f


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
        s2 = numpy.dot( s[i-1], s[i-1] )
        jump_term = numpy.linalg.solve( s[i-1] + alpha, s2 )
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
        rands = ( rng.uniform( size=(N, Nsamples ) ) < la*dt ).astype('int')

    else:
        assert rands.shape == (N,NSamples)
        rands = (rands<la*dt).astype('int')

    eta2 = numpy.dot( eta, eta.T )

    for i in xrange(1,N):
        asigmas = numpy.dot( sigmas[i-1], a) 
        nojump = asigmas + asigmas.swapaxes( 1, 2 ) + eta2
        jump = [ numpy.dot(numpy.linalg.solve( si + alpha, si ), alpha ) for si in sigmas[i-1]]
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
    
    integral = numpy.trace( numpy.sum( [ numpy.dot( bss, numpy.linalg.solve( R, numpy.dot( bss, sigmas[i] ) ) ) for i,bss in enumerate(bs) ], axis = 0  ) )

    f += dt * integral
    return f

if __name__=='__main__':
    print __doc__


    try:
        c = Client(profile='bryonia')
        dview = c[:]

        with dview.sync_imports():
            import numpy
        lview = c.load_balanced_view()

    except:
        print "NOPE"
        exit()

    N = int(T/dt)
    #precompute solution to the Ricatti equation    
    S = solve_riccatti(N,dt,QT,a,b,q,R)

    #range of covariance matrices evaluated
    alphas = numpy.arange(0.001,4.0,.1)

    #initial sigma value
    s = 2.0*numpy.eye(2)

    #preallocating numpy vectors for better performance
    fs = numpy.zeros((alphas.shape[0],alphas.shape[0]))
    full_fs =  numpy.zeros((alphas.shape[0],alphas.shape[0]))
    #estimation_eps = numpy.zeros_like(alphas)
    NSamples = 1

    mean_field = lambda (i,j,ax,ay,la) : (i,j,mf_f(s,S,dt,a,eta,numpy.diag([ax**2,ay**2]),b,q,R,la))
    full_stoc = lambda (i,j,ax,ay,la) : (i,j,full_stoc_f(s,S,dt,a,eta,numpy.diag([ax**2,ay**2]),b,q,R,la,NSamples,rands=rands))

    print 'running '+str(alphas.size**2)+' runs'
    rands = numpy.random.uniform(size=(N,NSamples))

    args = []
    for i,alx in enumerate(alphas):
        for j,aly in enumerate(alphas):
            
            la = numpy.sqrt((2*numpy.pi)**2*numpy.linalg.det(numpy.diag([alx**2,aly**2])))*phi/(dtheta**2)
            args.append((i,j,alx,aly,la)) 
    
    dview.push({'mf_f':mf_f,'full_stoc_f':full_stoc_f,'s':s,'S':S,'a':a,'N':N,
                'eta':eta,'b':b,'q':q,'R':R,'NSamples':NSamples,'rands':rands,'dt':dt})

    dview.push({'mf_sigma':mf_sigma,'full_stoc_sigma':full_stoc_sigma})

    mf_calls = lview.map_async( mean_field, args, ordered=False )
    full_calls = lview.map_async( full_stoc, args, ordered=False )

    gotten = []
    for n,res in enumerate(mf_calls):
        i,j,v = res
        gotten.append((i,j))
        print 'MF %d entries in, %d, %d'%(len(gotten),i,j)
        fs[i,j] = v
    print 'mean field is in'
    
    gotten = []
    for n,res in enumerate(full_calls):
        i,j,v = res
        gotten.append((i,j))
        print 'Stochastic %d entries in, %d %d'%(len(gotten),i,j)
        full_fs[i,j] = v
    print 'stochastic is in too'

    plt.subplot(2,1,1)
    plt.imshow(fs,interpolation='nearest')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(full_fs, interpolation='nearest')
    plt.colorbar()

    plt.savefig('full_multi_heatmap.png',dpi=300)
