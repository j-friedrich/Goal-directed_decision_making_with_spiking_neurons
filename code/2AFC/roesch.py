# produces Fig similar to Fig.5 Roesch, Olson (2003)
# Impact of expected reward on neuronal activity in prefrontal cortex,
# frontal and supplementary eye fields and premotor cortex.
import numpy as np
import pylab as pl
from sys import path, argv
path.append('../')
from NetPop import NetPop
from scipy.signal import convolve
from run import run
from functions import simpleaxis

savefig = False if len(argv) == 1 else True

pl.rc('figure', facecolor='white', dpi=90, frameon=False)
pl.rc('font', size=44, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
pl.rc('lines', lw=4)
pl.rc('text', usetex=True)
pl.rc('legend', **{'fontsize': 32})
pl.rc('axes', linewidth=2)
pl.rc('xtick.major', size=10, width=1.5)
pl.rc('ytick.major', size=10, width=1.5)

# colors for colorblind from  http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ["#009E73", "#0072B2", "#D55E00", "#E69F00",
       "#56B4E9", "#F0E442", "#CC79A7", "#999999"]


def smooth_spikes(spikes, sigma, step):
    return convolve(spikes, [1. / np.sqrt(2 * np.pi) / sigma * 1000
                             * np.exp(-i**2 / (2 * (sigma / step)**2))
                             for i in range(int(-5 * sigma / step),
                                            int(5 * sigma / step + 1))], 'same')


def sim(u0, rate, reset, tm, th, rb, r0, r1, r2, rp, tmean, tsigma, ir, runs):
    step = .2
    net = NetPop(1)
    np.random.seed(0)
    tmp = u0 * np.ones(len(net.W))
    tmp[-1] = .5
    s = np.array([run(net.W, u0 * np.ones(len(net.W)), step, 1,
                      rate=rate, T=1700, tm=tm, reset=reset, seed=i,
                      th=th, rbase=rb, r0=r0, r1=r1, rp=rp, tmean=tmean, tsigma=tsigma)
                  for i in range(runs)])
    s3 = np.array([run(net.W, u0 * np.ones(len(net.W)), step, 1,
                       rate=rate, T=1700, tm=tm, reset=reset, seed=i,
                       th=th, rbase=rb, r0=r0, r1=r2, rp=rp, tmean=tmean, tsigma=tsigma)
                   for i in range(runs)])
    pre = np.ones((int(250 / step), len(net.W))) * step * ir * 100 / 1000.
    pre[:, -1] = 0
    return np.array(
        [smooth_spikes(np.vstack((pre, np.mean(s3, axis=0)))[:, 1], 40, step),
            smooth_spikes(np.vstack((pre, np.mean(s3, axis=0)))[:, 0], 40, step),
            smooth_spikes(np.vstack((pre, np.mean(s, axis=0)))[:, 1], 40, step),
            smooth_spikes(np.vstack((pre, np.mean(s, axis=0)))[:, 0], 40, step),
            smooth_spikes(np.vstack((pre, np.mean(s, axis=0)))[:, 2], 40, step)])


# sim:
res = sim(1.8, 450, 0, 50, 1.65, 0, 2 * 0.28, 2 * 0.33, 2 * 0.43, 5.2, 95, 30, 22, 30)


fig = pl.figure()
ax = fig.add_subplot(111)
# plot simulation results
ax.plot(np.arange(0, 1500, .2), res[0, 250 * 5:1750 * 5] / 100., c=col[1], lw=5)
ax.plot(np.arange(0, 1500, .2), res[1, 250 * 5:1750 * 5] / 100., c=col[6], lw=5)
ax.plot(np.arange(0, 1500, .2), res[2, 250 * 5:1750 * 5] / 100., c=col[1], lw=3)
ax.plot(np.arange(0, 1500, .2), res[3, 250 * 5:1750 * 5] / 100., c=col[6], lw=3)
# labels & layout
pl.xlabel('Time from direc.\ cue [s]')
pl.ylabel('Firing rate [Hz]', y=.45)
pl.xticks([0, 500, 1000], [0, 0.5, 1.0])
pl.xlim(0, 1500)
pl.yticks([15, 20, 25], [15, 20, 25])
pl.ylim(15, 25.8)
simpleaxis(pl.gca())
# reward neuron
ax2 = ax.twinx()
# fake outside plotrange for legend
p3, = ax2.plot([0, 100], [-10, -10], c='grey', lw=6, zorder=-5)
# fake outside plotrange for legend
p4, = ax2.plot([0, 100], [-10, -10], c='grey', lw=2, zorder=-5)
# fake outside plotrange for legend
p5, = ax2.plot([0, 100], [-10, -10], c=col[1], lw=4, zorder=-5)
# fake outside plotrange for legend
p6, = ax2.plot([0, 100], [-10, -10], c=col[6], lw=4, zorder=-5)
rp, tmean, tsigma = 5.2, 95, 30
ax2.plot(np.arange(0, 1500, .2), smooth_spikes(
    [.2 * 450 / 1000. / 100 * (1 + rp * np.exp(-(t * .2 - tmean)**2 / tsigma**2))
     for t in range(int(1750 / .2))], 40, .2)[:1500 * 5], c='black', lw=2, zorder=-5)
ax2.set_ylim(2, 60)
ax2.set_yticks([0, 10])
ax2.set_yticklabels([0, 10])
ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
ax.patch.set_visible(False)  # hide the 'canvas'
# right axis  & layout
pl.plot([1500, 1500], [0, 15], c='black', lw=2, clip_on=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig.subplots_adjust(.17, .22, .92, .95)
lg2 = pl.legend([p3, p4, p5, p6], [r'$r_A = 3$', r'$r_A = 1$', 'A', 'B'],
                bbox_to_anchor=(1.07, 1.14), labelspacing=.23, handletextpad=.15, handlelength=1.6)
lg2.draw_frame(False)

if savefig:
    pl.savefig('fig/Roesch.pdf', dpi=600)
else:
    pl.show()
