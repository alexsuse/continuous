
import cPickle as pic
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import sys

#from matplotlib import rc

import prettyplotlib as ppl
from prettyplotlib import plt

try:
    filename = sys.argv[1]
except:
    filename = 'figure-1-1.pik'

with open(filename,"r") as f:
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
#rc('text',usetex='true')
# find minimum
epsmin,indeps = (np.min(estimation_eps),np.argmin(estimation_eps))
# plot it up
print mi
mi = mi*estimation_eps[0]/np.max(mi)
print mi

fig, (ax1,ax2) = ppl.subplots(2,1,sharex=True)

l1, = ppl.plot(alphas, estimation_eps,label='Point Process Filtering',ax=ax1)
l2, = ppl.plot(alphas, kalman_eps, '-.',label='Kalman Filtering',ax=ax1 )
l6, = ppl.plot(alphas, mi,label='Mutual Information',ax=ax1)
ppl.plot(alphas[indeps],epsmin,'o',color=l1.get_color(),ax=ax1)

#ax1.text(alphas[2],0.16,'a)')

l3,=ppl.plot( alphas, fs, label='Point Process Control (MF)',ax=ax2)
l4, = ppl.plot(alphas, full_fs, label='Point Process Control (Numeric)',ax=ax2 )
l5, = ppl.plot(  alphas, lqg_fs, '-.', label='LQG Control', ax=ax2 )
ppl.plot(alphas[indfs],fsmin,'o',color=l3.get_color(),ax=ax2)
ppl.plot(alphas[indfull],fullmin,'o',color=l4.get_color(),ax=ax2)

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(axis='x',which='both',bottom='off')
ax2.tick_params(axis='x',which='both',top='off')

ax1.set_ylabel(r'$MMSE$')
ax2.set_ylabel(r'$f(\Sigma_0,0)$')
ax2.set_xlabel(r'$p$')

ppl.legend(ax1,loc=4).get_frame().set_alpha(0.7)
ppl.legend(ax2,loc=4).get_frame().set_alpha(0.7)

#plt.figlegend([l1,l2,l6,l3,l4,l5],
#              ['estimation','Kalman filter','Mutual Information','mean field','stochastic','LQG control'],
#              'upper right')
plt.savefig('comparison_uni_low.pdf')
plt.savefig('comparison_uni_low.png')
