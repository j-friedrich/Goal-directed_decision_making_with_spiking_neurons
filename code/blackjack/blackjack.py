import matplotlib      # these lines to
matplotlib.use('Agg')  # work remotely
import numpy as np
import pylab as pl
from sys import path
path.append('../')
from scipy.sparse import csr_matrix
from NetPop import NetPop
import cfunctions as cfn
from functions import simpleaxis, errorfill, init_fig

init_fig()


net = NetPop(1)
W = np.copy(net.W)
dt = 1

R0 = net.R4flatPi(net.pstart_state)
Rmax = np.dot(net.calc_Qvalue().max(axis=1), net.pstart_state)

ref = 20
step = .2
rate = 400

## offline ##
try:
    perf = np.load('results/performance.npy')
except IOError:
    try:
        S = np.load('results/spikes.npz')['S']
    except IOError:
        S = [csr_matrix(
            cfn.runpop_js(net.W, step, 1, rate, 1000, 20, 2, ref, run))
            for run in range(10)]
        np.savez_compressed('results/spikes.npz', S=S)
    Tls = range(501)
    rew = np.array([[net.R4Pi2(net.get_policy2(step, i, T), net.pstart_state)
                     for i in S] for T in Tls])
    perf = (rew - R0) / (Rmax - R0)
    perf[0] = 0
    np.save('results/performance', perf)
pl.figure()
errorfill(range(len(perf)), np.mean(perf, axis=1),
          yerr=np.std(perf, axis=1) / np.sqrt(len(perf[0])))
pl.xticks([0, 200, 400], [0, 200, 400])
pl.yticks([0, .5, 1.0], [0, .5, 1.0])
pl.xlim([0, 400])
pl.ylim([0, 1])
pl.xlabel('Time [ms]')
pl.ylabel('Performance')
simpleaxis(pl.gca())
pl.tight_layout(0)
pl.savefig('performance.pdf', dpi=600)


# learning via parallel sampling
eta, T, r0 = .01, 1000, 0
try:
    perf = np.load('results/learn_performance.npy')
except IOError:
    try:
        rew = np.load('results/learn.npz')['R']
    except IOError:
        res = np.array([net.parallel_sampling_keepU(step, eta, run, rate, T,
                                                    r0=r0, reset=ref, steps=100,
                                                    start_state=net.pstart_state)
                        for run in range(10)])
        np.savez_compressed('results/learn.npz', Pi=res[:, 0], R=res[:, 1],
                            DKL=res[:, 2], RMSE=res[:, 3], W=res[:, 4])

        rew = np.load('results/learn.npz')['R']
    perf = (rew - R0) / (Rmax - R0)
    np.save('results/learn_performance', perf)
pl.figure()
errorfill(range(len(perf[0])), np.mean(perf, axis=0),
          yerr=np.std(perf, axis=0) / np.sqrt(len(perf)))
pl.xticks([0, 50, 100], [0, 50, 100])
pl.yticks([0, .5, 1.0], [0, .5, 1.0])
pl.ylim([0, 1])
pl.xlabel('Trials')
pl.ylabel('Performance')
simpleaxis(pl.gca())
pl.tight_layout(0)
pl.savefig('learn_performance.pdf', dpi=600)
