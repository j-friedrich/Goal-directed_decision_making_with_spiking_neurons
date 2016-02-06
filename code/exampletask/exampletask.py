import numpy as np
import pylab as pl
from sys import argv
from Net import Net
import cfunctions as cfn
from functions import simpleaxis, init_fig

savefig = False if len(argv) == 1 else True

net = Net()
step = .1

init_fig()
# colors for colorblind from  http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ['#56B4E9', '#0072B2',  '#F0E442', '#E69F00',
       '#CC79A7', '#D55E00', '#40e0d0', '#009E73']


# Voltage
spikes, u = cfn.run(net.W, step, 400, 50, 20, 2, 20, 2)
fig1, ax = pl.subplots(nrows=2, ncols=1)
for i, j in enumerate([7, 1]):
    ax[i].plot(u[:, j] + 2 * spikes[:, j], color=col[j], zorder=5)
    ax[i].set_ylim(-1, 3.5)
    ax[i].axis('off')
ax[-1].plot([0, 10 / step], [-1, -1], c='k', lw=7, clip_on=False, zorder=5)
ax[-1].plot([0, 0], [1, 2], c='k', lw=7, clip_on=False, zorder=5)
fig1.subplots_adjust(hspace=-.1)
fig1.subplots_adjust(.04, .1, 1, 1)
if savefig:
    pl.savefig('u.pdf', dpi=600, transparent=True)
else:
    pl.show()

# Spikes
spikes, u = cfn.run(net.W, step, 400, 100, 20, 2, 20, 2)
pl.figure()
for i in xrange(8):
    map(lambda a: pl.plot([a, a], [.95 + i, .05 + i],
                          c=col[i]), np.where(spikes[:, i] == 1)[0])
pl.yticks(np.arange(.4, 8), ['0L', '0R', '1L', '1R', '2L', '2R', '3L', '3R'])
pl.gca().yaxis.set_tick_params(width=0)
pl.xticks(np.arange(0, len(u) + 1, 40 / step), [0, 40, 80])
simpleaxis(pl.gca())
pl.tight_layout(pad=.03)
if savefig:
    pl.savefig('spikes.pdf', dpi=600, transparent=True)
else:
    pl.show()


def f(W, x, ref):
    y = (1 - 1. / 2000) * x + 1. / 2000\
        * ((1 + ref) * np.dot(W, x * (x > 0)) - ref * x * (x > 0))
    y[net.r] = 1
    return y

x = np.zeros((1201, net.K))
np.random.seed(2)
x[0] = [0, 0, 0, .2, .55, .45, -.3, -1, 1]
for i in xrange(len(x) - 1):
    x[i + 1] = f(net.W, x[i], 20)

# Rate
lstyle = ['--', '-', '-', '-', '-', '--', '--', '-']
fig1, ax = pl.subplots(nrows=8, ncols=1)
for i in xrange(8):
    ax[7 - i].plot(x[:, i] * (x[:, i] > 0), lstyle[i],
                   color=col[i], zorder=5, clip_on=False)
    ax[7 - i].set_ylim(0, 1)
    ax[7 - i].set_xticks(np.arange(0, len(x) + 1, 500))
    ax[7 - i].set_xticklabels([])
    ax[7 - i].set_yticks([0, 1])
    ax[7 - i].set_yticklabels([])
    simpleaxis(ax[-i])
ax[-1].set_xticklabels([0, 5, 10])
ax[-1].set_yticklabels([0, 400])
pl.tight_layout(pad=.03, h_pad=-.25)
if savefig:
    pl.savefig('rate.pdf', dpi=600, transparent=True)
else:
    pl.show()


# Value
fig1, ax = pl.subplots(nrows=4, ncols=1)
for i in xrange(4):
    if i == 1:
        ax[3 - i].plot([0, len(x) - 1], [.75] * 2, '--', color='black', clip_on=False)
    else:
        ax[3 - i].plot([0, len(x) - 1], [1] * 2, '--', color='black', clip_on=False)
    ax[3 - i].plot((x[:, 2 * i] * (x[:, 2 * i] > 0) + x[:, 2 * i + 1]
                    * (x[:, 2 * i + 1] > 0)), color=col[2 * i], clip_on=False)
    ax[3 - i].plot((x[:, 2 * i] * (x[:, 2 * i] > 0) + x[:, 2 * i + 1]
                    * (x[:, 2 * i + 1] > 0)), '--', color=col[2 * i + 1], clip_on=False)
    ax[3 - i].set_xticks(np.arange(0, len(x) + 1, 500))
    ax[3 - i].set_xticklabels([])
    ax[3 - i].set_yticks([0, 1])
    ax[3 - i].set_yticklabels([])
    simpleaxis(ax[i])
ax[-1].set_xticklabels([0, 5, 10])
ax[-1].set_yticklabels([0, 1])
pl.tight_layout(pad=.03, h_pad=-.2)
if savefig:
    pl.savefig('value.pdf', dpi=600, transparent=True)
else:
    pl.show()
