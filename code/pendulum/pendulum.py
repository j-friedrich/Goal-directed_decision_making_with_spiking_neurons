import matplotlib      # these lines to
matplotlib.use('Agg')  # work remotely
import numpy as np
import pylab as pl
from sys import path
path.append('../')
from scipy.sparse import csr_matrix
from matplotlib import colors
import random
from NetPop import NetPop
import integrate2d as cf
import cfunctions as cfn
from functions import simpleaxis, errorfill


pl.rc('figure', facecolor='white', dpi=90, frameon=False)
pl.rc('font', size=44, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
pl.rc('lines', lw=4)
pl.rc('text', usetex=True)
pl.rc('legend', **{'fontsize': 36})
pl.rc('axes', linewidth=2)
pl.rc('xtick.major', size=10, width=1.5)
pl.rc('ytick.major', size=10, width=1.5)

# colors for colorblind from  http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ['#56B4E9', '#E69F00', '#CC79A7', '#009E73']


####################
#### parameters ####
####################
gamma = .98
sigma = .5

net = NetPop(1, 3)
W = np.copy(net.W)
W[:-1, :-1] *= gamma**3
W -= (1 - gamma**3) * net.competition

valley = [8, 16]


##############################
#### function definitions ####
##############################

def get_path(x, initpv=valley, maxtrials=300):
    pv = [initpv]
    count = 0
    while count < maxtrials:
        pv += [cf.get_next_pv(pv[-1][0], pv[-1][1], net.get_a(x, *pv[-1]))]
        count += 1
    return np.array(pv)


def get_R4pv(pv):
    p, v = pv
    return np.exp(-(min(p, 16 - p) * np.pi / 8)**2 - .2 * ((v - 16) * np.pi / 8)**2)


def plot_path(x, initpv=valley, maxtrials=300, color='black', **kwargs):
    path = get_path(x, initpv, maxtrials)
    pl.scatter((path[:, 0] + 8) % 16, path[:, 1], color=color, **kwargs)
    pl.xticks([0, 8, 16], [r"-$\pi$", 0, r"$\pi$"])
    pl.yticks([8, 16, 24], [r"-$\pi$", 0, r"$\pi$"])
    pl.xlabel('Angle')
    pl.ylabel('Angular velocity', labelpad=15)
    pl.tight_layout(0)


def get_a_rnd(x, p, v):
    Q = net.get_Q(x, p, v)
    return random.choice(np.where(Q == Q.max())[0])


def get_path_rnd(x, initpv=valley, maxtrials=300):
    pv = [initpv]
    count = 0
    while count < maxtrials:
        pv += [cf.get_next_pv(pv[-1][0], pv[-1][1], get_a_rnd(x, *pv[-1]))]
        count += 1
    return np.array(pv)


def smooth_spikes(spikes, sigma, step, init=0):
    tmp = init * step * np.ones_like(spikes)
    for i in range(len(spikes) - 1):
        tmp[i + 1] = np.exp(-step / sigma) * tmp[i]\
            + (1 - np.exp(-step / sigma)) * spikes[i + 1]
    return 1000 / step * tmp


###############################
#### simulations and plots ####
###############################

# optimal DP
h = .1
try:
    np.load('results/opt_value.npy').T
except IOError:
    V = np.ones((int(16 / h), int(32 / h))) / (1 - gamma) / 2
    for i in xrange(1000):
        cf.iterate(V, gamma, h)
    np.save('results/opt_value', V)

pl.figure(figsize=(8, 5))
X, Y = np.array(np.meshgrid(np.arange(0, 16 + h, h), np.arange(0, 32 + h, h)))
Z = np.load('results/opt_value.npy').T
pl.pcolormesh(X, Y, np.concatenate((Z[:, 80:], Z[:, :80]), axis=1),
              vmin=0, vmax=1 / (1 - gamma), rasterized=True)
pl.xticks([0, 8, 16], [r"-$\pi$", 0, r"$\pi$"])
pl.yticks([8, 16, 24], [r"-$\pi$", 0, r"$\pi$"])
pl.xlim(0, 16)
pl.ylim(0, 32)
pl.xlabel('Angle')
pl.ylabel('Angular velocity', labelpad=10, y=.45)
cb = pl.colorbar()
cb.set_label('obtained Value')
cb.set_ticks([0, 5 * int(round(.1 / (1 - gamma))), 10 * int(round(.1 / (1 - gamma)))])
cb.set_ticklabels([0, 5 * int(round(.1 / (1 - gamma))), 10 * int(round(.1 / (1 - gamma)))])
pl.subplots_adjust(.17, .25, .94, .94)
pl.savefig('fig/value_obt.pdf', dpi=600)


# R0 = np.mean([np.dot(map(get_R4pv, get_path_rnd(np.zeros(net.K), initpv=valley,
#                                                 maxtrials=1000)[:-1]),
#                      gamma**np.arange(1000)) for i in range(100)])
R0 = 0.0048097750112830288  # from above
Rmax = 6.4251441272230441  # from optimalDP

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
            cfn.runpop_js(W, step, 1, rate, 1000, 20, 2, ref, run))
            for run in range(10)]
        np.savez_compressed('results/spikes.npz', S=S)
    Tls = range(501)
    rewls = np.array([[np.dot(map(get_R4pv, get_path(
        np.array((i[:T / step]).sum(axis=0))[0],
        valley, 10**3)[:-1]), gamma**np.arange(1000))
        for i in S] for T in Tls])
    rewls[0] = R0
    perf = (rewls - R0) / (Rmax - R0)
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
pl.savefig('fig/performance.pdf', dpi=600)


# plot value:
h = .1
try:
    V = np.load('results/value_obt.npy')
except IOError:
    S = np.load('results/spikes.npz')['S']
    y = np.array(map(lambda a: a.sum(axis=0), S))[:, 0]
    V = []
    for run, x in enumerate(y):
        V += [np.array([[np.dot(map(get_R4pv, get_path(x, [p, v])), gamma**np.arange(301))
                         for p in np.arange(h / 2., 16, h)]
                        for v in np.arange(h / 2., 32, h)])]
    np.save('results/value_obt', np.array(V))

Z = np.mean(V, axis=0)
X, Y = np.array(np.meshgrid(np.arange(0, 16 + h, h), np.arange(0, 32 + h, h)))
fig = pl.figure(figsize=(8, 5))
pl.pcolormesh(X, Y, np.concatenate((Z[:, 80:], Z[:, :80]), axis=1),
              vmin=0, vmax=1. / (1 - gamma), rasterized=True)
pl.xticks([0, 8, 16], [r"-$\pi$", 0, r"$\pi$"])
pl.yticks([8, 16, 24], [r"-$\pi$", 0, r"$\pi$"])
pl.xlim(0, 16)
pl.ylim(0, 32)
pl.xlabel('Angle')
pl.ylabel('Angular velocity', labelpad=10, y=.45)
cb = pl.colorbar()
cb.set_label('obtained Value')
cb.set_ticks([0, 5 * int(round(.1 / (1 - gamma))), 10 * int(round(.1 / (1 - gamma)))])
cb.set_ticklabels([0, 5 * int(round(.1 / (1 - gamma))), 10 * int(round(.1 / (1 - gamma)))])
pl.subplots_adjust(.17, .25, .94, .94)
pl.savefig('fig/value_obt.pdf', dpi=600)


# plot policy:
S = np.load('results/spikes.npz')['S']
y = np.array(map(lambda a: a.sum(axis=0), S))[:, 0]
try:
    pm = np.load('results/policy.npy')
except IOError:
    pm = np.array([[net.get_a(np.mean(y, axis=0), p, v)
                    for p in np.arange(0 + h / 2., 16, h)]
                   for v in np.arange(h / 2., 32, h)])
    np.save('results/policy', np.array(pm))
# make a color map of fixed colors
cmap = colors.ListedColormap(['blue', 'green', 'red'])
bounds = [0, 2. / 3, 4. / 3, 2]
norm = colors.BoundaryNorm(bounds, cmap.N)
X, Y = np.array(np.meshgrid(np.arange(0, 16 + h, h), np.arange(0, 32 + h, h)))
# plot
pl.figure(figsize=(8, 5))
pl.pcolormesh(X, Y, np.concatenate((pm[:, 80:], pm[:, :80]), axis=1),
              cmap=cmap, norm=norm, rasterized=True)
pl.xticks([0, 8, 16], [r"-$\pi$", 0, r"$\pi$"])
pl.yticks([8, 16, 24], [r"-$\pi$", 0, r"$\pi$"])
pl.xlim(0, 16)
pl.ylim(0, 32)
pl.xlabel('Angle')
pl.ylabel('Angular velocity')
cb = pl.colorbar(cmap=cmap, norm=norm, boundaries=bounds)
cb.set_label('Action')
cb.set_ticks([1. / 3, 1, 5. / 3])
cb.set_ticklabels([r'$-$', 0, '$+$'])
# x and valley from pendulum.py
plot_path(y.mean(0), valley, 150,
          color=[pl.cm.Greys(i) for i in np.linspace(0, 0.99, 151)], linewidths=4)
pl.subplots_adjust(.18, .25, .95, .94)
pl.savefig('fig/policy.pdf', dpi=600)


## learning via parallel sampling ##

eta, T, r0, s = .05, 1000, 0, 500
try:
    perf = np.load('results/learn_performance.npy')
except IOError:
    try:
        Seq = np.load('results/res.npz')['Seq']
    except IOError:
        res = np.array([np.array(net.parallel_sampling_keepU(
            step, eta, run, rate, T, r0=r0, reset=ref,
            gamma=gamma, trials=100, samples=s)) for run in range(10)])
        np.savez_compressed('results/res', U=res[:, 0], Seq=res[:, 1], W=res[:, 2])
        Seq = res[:, 1]
    rls = np.array(map(lambda x: x[:, 2], Seq))
    rewls = np.array(map(lambda a: np.dot(a, gamma**np.arange(300)), rls), dtype=float)
    rewls[:, 0] = R0
    perf = (rewls - R0) / (Rmax - R0)
    perf = perf * (perf > 0)
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
pl.savefig('fig/learn_performance.pdf', dpi=600)


# initialize our network at two different points in the state space of its neural
# activities that correspond to representing the same (approximate) value function

uu0 = .2
res0 = [cfn.runpopU_js(W, uu0 * np.ones(net.K), step, 1, rate, 1000, 20, 2, ref, run)[0]
        for run in range(30)]
res1 = [cfn.runpopU_js(W, np.hstack([np.ravel(
    np.outer(np.ones(net.K / 3), [-12, uu0 * .75, 2.25 * uu0])), np.array([1])]),
    step, 1, rate, 1000, 20, 2, ref, run)[0]
    for run in range(30)]

pl.figure(figsize=(6, 6))
for i in range(4):
    p, v = [(0, 17), (8, 17), (5, 22), (2, 18), (10, 27)][i]
    pl.plot(smooth_spikes(np.sum([
        net.get_Q(s, p, v) for s in np.mean(res0, 0)], 1),
        40., step, 3 * uu0) / rate, c=col[i], zorder=10)
    pl.plot(smooth_spikes(np.sum([
        net.get_Q(s, p, v) for s in np.mean(res1, 0)], 1),
        40., step, 3 * uu0) / rate, '--', c=col[i], zorder=10)
pl.xticks([0, 200 / step, 400 / step], [0, 200, 400])
pl.yticks([0, 5, 10], [0, 5, 10])
pl.xlim([0, 450 / step])
pl.xlabel('Time [ms]')
pl.ylabel('Value', labelpad=-5)
simpleaxis(pl.gca())
pl.subplots_adjust(.19, .22, .99, .99)
pl.savefig('fig/value.pdf', dpi=600)
pl.xlim([0, 25 / step])
pl.xticks([0, 10 / step, 20 / step], [0, 10, 20])
pl.yticks([1, 2], [1, 2])
pl.ylim(.95, 2.9)
pl.xlabel('')
pl.ylabel('')
pl.gcf().set_size_inches(4, 3)
pl.subplots_adjust(.09, .17, .99, .99)
pl.savefig('fig/value2.pdf', dpi=600, transparent=True)


from scipy.signal import medfilt
pl.figure(figsize=(6, 6))
for k in range(4):
    p, v, a = [(0, 17, 1), (8, 17, 2), (5, 22, 2), (2, 18, 0), (10, 27, 2)][k]
    i = p * 99 + v * 3 + a
    pl.plot(medfilt(smooth_spikes(
            np.mean(res0, 0)[:, i], 40., step, uu0) / 1000., 9), c=col[k], zorder=10)
    line, = pl.plot(medfilt(smooth_spikes(
        np.mean(res1, 0)[:, i], 40., step,
        [0, uu0 * .75, 2.25 * uu0][a]) / 1000., 9), '--', c=col[k], zorder=10)
pl.xticks([0, 400 / step, 800 / step], [0, 400, 800])
pl.yticks([0, 1], [0, 1])
pl.xlim([0, 950 / step])
pl.ylim(0, 2)
pl.xlabel('Time [ms]')
pl.ylabel('Firing rate [kHz]')
simpleaxis(pl.gca())
pl.subplots_adjust(.17, .22, .99, .99)
pl.savefig('fig/rate.pdf', dpi=600)
pl.xlim([0, 25 / step])
pl.xticks([0, 10 / step, 20 / step], [0, 10, 20])
pl.ylim(-.005, .6)
pl.yticks([0, .5], [0, .5])
pl.xlabel('')
pl.ylabel('')
pl.gcf().set_size_inches(4, 3)
pl.subplots_adjust(.2, .17, .99, .99)
pl.savefig('fig/rate2.pdf', dpi=600, transparent=True)
