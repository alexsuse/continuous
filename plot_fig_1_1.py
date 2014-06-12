
import cPickle as pic
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib import rc

with open("figure-1-1.pik","r") as f:
    dic = pic.load(f)
    lqg_fs = dic['LQG control']
    estimation_eps = dic['poisson filtering']
    kalman_eps = dic['kalman filtering']
    mi = dic['mutual information']
    fs = dic['mf poisson control']
    full_fs = dic['full poisson control']
    alphas = dic['alphas']

# find minimum of mf cost
fsmin,indfs = (np.min(fs),np.argmin(fs))
# find minimum of stoch cost
fullmin,indfull = (np.min(full_fs),np.argmin(full_fs))
rc('text',usetex='true')
# find minimum
epsmin,indeps = (np.min(estimation_eps),np.argmin(estimation_eps))
# plot it up
mi = mi*estimation_eps[0]/np.max(mi)

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
