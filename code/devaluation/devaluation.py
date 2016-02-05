import numpy as np
import pylab as pl
from sys import path, argv
path.append('../')
from NetPop import NetPop
from functions import simpleaxis, smooth_spikes

savefig = False if len(argv) == 1 or int(argv[1]) == 0 else True
deval = False if len(argv) < 3 else True

pl.rc('figure', facecolor='white', dpi=90, frameon=False)
pl.rc('font', size=44, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
pl.rc('lines', lw=5)
pl.rc('text', usetex=True)
pl.rc('legend', **{'fontsize': 30})
pl.rc('axes', linewidth=2)
pl.rc('xtick.major', size=10, width=1.5)
pl.rc('ytick.major', size=10, width=1.5)


# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ['#56B4E9', '#0072B2',  '#F0E442', '#E69F00', '#CC79A7', '#D55E00', '#009E73', '#999999']


def runU(W, uinit, step, rate=100, T=700, tm=50, ts=2, reset=3, delay=0):
    K = len(W)
    spikes = np.zeros((T / step, K))
    spikes[0, K - 1] = .3
    fs = np.exp(-step / ts)  # decay factor synapse
    fm = np.exp(-step / tm)  # decay factor membrane
    # combine factor to save computing time
    f = 1. / ts * np.sqrt(fs) * step / tm * np.sqrt(fm)
    c = np.zeros(K)  # current
    u = uinit
    f *= (1 + reset)
    f2 = 1. * reset / tm
    for t in range(1, int(T / step)):
        c *= fs
        if 1 in spikes[int(t - 1 - delay / step)]:
            c += np.dot(W, spikes[int(t - 1 - delay / step)]) * f
        u = u * fm + c - f2 * spikes[t - 1]
        u[K - 1] = rate / 1000. * \
            (.1 + .65 * np.exp(-(t * step - 250) ** 2 / tau_r ** 2))
        spikes[t] = np.random.rand(K) < step * u
    return spikes


def accumulate(values, start=0):
    for v in values:
        start += v
        yield start


def get_t(x, c):
    g0 = accumulate(x)
    th = thresh * gamma ** c
    for t in range(3500):
        g = g0.next()
        if abs(g) >= th:
            break
    return t, g

popsize = 1
gamma = .7
step = .2
tau_r = 80.
rate = 100

###  simulation ###
net = NetPop()
W = np.copy(net.W)
W[:-1, :-1] *= gamma
W -= (1 - gamma) * net.competition
if deval:
    W[2, -1] /= 2.
s = []
for seed in range(30):
    u0 = np.zeros(len(W))
    np.random.seed(seed)
    s += [runU(W, u0, step, rate)]
s = np.array(s)
pre = np.zeros((int(250 / step), len(W)))
pres = np.array([np.vstack((pre, ss)) for ss in s])
Ratels = [np.array([np.mean(np.array([smooth_spikes(p[:, j * popsize + i], 40, .2)
                                      for i in range(popsize)]), axis=0)
                    for j in range(len(W))])[:, 250 / step:750 / step] for p in pres]


# plot neural activities:
pl.figure()
for c in range(6):
    pl.plot(np.array(Ratels).mean(axis=0)[c], color=col[c], zorder=5)
pl.xlabel('Time [ms]', labelpad=0)
pl.ylabel('Firing rate [Hz]', labelpad=10)
pl.xticks([0, 250 / step, 500 / step], [0, 250, 500])
pl.yticks(*[[0, 20, 40, 60, 80]] * 2)
pl.xlim(0, 500 / step)
pl.ylim(0, 64)
pl.plot(smooth_spikes([step * 100 / 1000. *
                       (.1 + .65 * np.exp(-(t * step - 500) ** 2 / tau_r ** 2))
                       for t in range(int(1000 / step))], 40, .2)[250 / step:750 / step],
        color='black', zorder=4, lw=2)
pl.subplots_adjust(.18, .21, .945, .99)
simpleaxis(pl.gca())
if savefig:
    pl.savefig('deval' + ('d' if deval else '') + '.pdf', dpi=600)
else:
    pl.show()


# plot RT from threshold:
sdiff = s[:, :, range(0, 6, 2)] - s[:, :, range(1, 6, 2)]
t = np.zeros((3, 30, 2))
thresh = 5
for i, c in enumerate([1, 0, 0]):
    t[i] = map(lambda x: get_t(x, c), sdiff[:, :, i])
dec = t[:, :, 1]
t = t[:, :, 0] * .2
pl.figure(figsize=(4.5, 6))
pl.errorbar(range(3), t.mean(axis=1), yerr=t.std(axis=1) / np.sqrt(30),
            fmt='s', ms=22, color=col[1], mec=col[1], capthick=2, zorder=10)
pl.xlabel('State')
pl.ylabel('Reaction time [ms]', labelpad=10)
pl.xticks(*[range(3)] * 2)
pl.yticks(*[[250, 300, 350, 400]] * 2)
pl.ylim(210, 460)
pl.xlim(-.2, 2.2)
pl.subplots_adjust(.38, .21, .99, .99)
simpleaxis(pl.gca())
if savefig:
    pl.savefig('deval_RT' + ('d' if deval else '') + '.pdf', dpi=600)
else:
    pl.show()


# plot decision at threshold:
pl.figure(figsize=(4.5, 6))
pl.bar(np.arange(-.4, 2), .5 + .5 * np.sign(dec).mean(1), color=col[1])
pl.xlabel('State')
pl.ylabel(r'Left choice [\%]', labelpad=-8)
pl.xticks(range(3), range(3))
pl.yticks([0, .5, 1], [0, 50, 100])
pl.ylim(0, 1)
pl.xlim(-.5, 2.5)
pl.subplots_adjust(.33, .21, .99, .95)
simpleaxis(pl.gca())
if savefig:
    pl.savefig('deval_Dec' + ('d' if deval else '') + '.pdf', dpi=600)
else:
    pl.show()
