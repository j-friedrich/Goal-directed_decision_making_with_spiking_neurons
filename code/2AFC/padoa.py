# produces Figs similar to Camillo Padoa-Schioppa & John A. Assad (2006)
# Neurons in the orbitofrontal cortex encode economic value

# Import matplotlib before matplotlib.pyplot to set the rcParams
# in order to use LaTeX
import matplotlib as mpl
# Use true LaTeX and bigger font
mpl.rc('text', usetex=True)
# Include packages `amssymb` and `amsmath` in LaTeX preamble
# as they include extended math support (symbols, envisonments etc.)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb}", r"\usepackage{amsmath}"]
mpl.rcParams['figure.max_open_warning'] = 50

import matplotlib.pyplot as pl
import numpy as np
from sys import path, argv
path.append('../')
from NetPop import NetPop
from scipy.special import erfinv
from scipy.optimize import curve_fit
from runPadoa import run
from functions import simpleaxis, smooth_spikes, accumulate, init_fig


savefig = False if len(argv) == 1 else True

init_fig()
# colors for colorblind from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ["#009E73", "#0072B2", "#D55E00", "#E69F00",
       "#56B4E9", "#F0E442", "#CC79A7", "#999999"]
col2 = map(tuple, [np.array([.902, .624, 0]) +
                   (np.array([0, .62, .451]) -
                    np.array([.902, .624, 0])) * i / 4 for i in range(5)]) +\
    map(tuple, [np.array([0, .62, .451]) +
                (np.array([.337, .706, .914]) -
                    np.array([0, .62, .451])) * i / 4 for i in range(1, 5)])


u0 = .2
thresh = 7
rate = 200
reset = 0.0
tm = 25.0
tdec = 300
trise = 110
tdly = 60
runs = 100
step = .2
net = NetPop(1)


##### function definitions #####

def sim(u0, rate, reset, tm, offers, ratio, tdec, trise, tdly,
        runs, rb=0, V0=0, gamma=1, seedoff=0):
    np.random.seed(0)
    s = np.zeros((len(offers), runs, int(1200 / step), net.K))
    for j, offer in enumerate(offers):
        s[j] = np.array([run(net.W, u0 * np.ones(len(net.W)),
                             step, 1, rate=rate, T=1200, tm=tm, reset=reset,
                             seed=i if seedoff == 0 else seedoff + i + j * runs,
                             r0=offers[j, 0], r1=offers[j, 1] * ratio,
                             tdec=tdec, trise=trise, tdly=tdly, rb=rb, V0=V0, gamma=gamma)
                         for i in range(runs)])
    return s


def get_t(x):
    g0 = accumulate(x[int(tdly / step):, 0])
    g1 = accumulate(x[int(tdly / step):, 1])
    for t in range(5000):
        a0 = g0.next()
        a1 = g1.next()
        if abs(a0 - a1) >= thresh:
            break
    return t + int(tdly / step), a0, a1


def getRT(w):  # arg which 3b or 3e
    x = np.transpose(s[w], (1, 0, 2, 3))
    rt = np.array([[get_t(a)[0] for a in x[r]] for r in range(len(x))])
    return rt


def n_score(rt, offers, ratio):
    rt = rt.reshape(-1)
    v = np.array([offers[:, 0], ratio * offers[:, 1]]).T
    vratio = 1. * np.min(v, axis=1) / np.max(v, axis=1)
    data = np.array([np.outer(np.ones(runs), vratio).reshape(-1),
                     np.sqrt(2) * erfinv(2 * ((np.argsort(np.argsort(rt)) + .5) /
                                              len(rt)) - 1)])  # n-score
    A = np.vstack([data[0], np.ones(len(data[0]))]).T
    c1 = np.linalg.lstsq(A, data[1])[0]
    return data, c1


#### choice probs at reaction time ####
def plotChoice(w):  # arg which 3b or 3e
    offers = eval('offers' + w)
    ratio = eval('ratio' + w)
    pB = eval('pB' + w)
    v = np.array([offers[1:-1, 0], ratio * offers[1:-1, 1]]).T
    logratio = np.log(v[:, 0] / v[:, 1])
    x = np.transpose(s[w], (1, 0, 2, 3))
    rts = np.array([[get_t(a) for a in x[r]]
                    for r in range(len(x))], dtype=int)
    choice = rts[:, :, 1] > rts[:, :, 2]
    pBsim = np.mean(choice, 0)
    # plot
    pl.figure()
    pl.plot(logratio, pB[1:-1], '--', marker='s', ms=30,  mfc='w',
            mec=col[0], mew=3, color=col[0], zorder=10, clip_on=False)
    pl.plot(logratio, pBsim[1:-1], color=col[1], marker='s', ms=22,
            mec=col[1], clip_on=False, zorder=10)
    pl.plot([logratio[0] - .5, logratio[-1] + .5], [pB[0], pB[-1]], 's', ms=30,
            mfc='w', mec=col[0], mew=3, color=col[0], zorder=10, clip_on=False)
    pl.plot([logratio[0] - .5, logratio[-1] + .5], [pBsim[0], pBsim[-1]], 's',
            color=col[1], ms=22, mec=col[1], clip_on=False, zorder=10)
    pl.xticks([logratio[0] - .5, -1, 0, 1, logratio[-1] + .5, ],
              [r"$-\infty$", -1, 0, 1, r"$\infty$"])
    tt = pl.gca().get_xaxis().majorTicks
    for i in range(len(tt)):
        tt[i].set_pad(17)
    pl.yticks([0, .5, 1], [0, 50, 100])
    pl.xlim(logratio[0] - .65, logratio[-1] + .5)
    pl.ylim(0, 1)
    pl.xlabel('log(V(B)/V(A))', labelpad=-2)
    pl.ylabel('B choice [\%]', labelpad=-15)
    pl.text(.05, .83, 'A=' + str(ratio) + 'B', transform=pl.gca().transAxes)
    pl.subplots_adjust(.17, .23, .96, .95)
    simpleaxis(pl.gca())
    # broken axis
    pl.plot([logratio[0] - .27, logratio[0] - .23],
            [0, 0], c='w', lw=2, clip_on=False, zorder=11)
    pl.plot([logratio[0] - .29, logratio[0] - .25],
            [-.02, .02], c='k', lw=2, clip_on=False, zorder=11)
    pl.plot([logratio[0] - .25, logratio[0] - .21],
            [-.02, .02], c='k', lw=2, clip_on=False, zorder=11)
    pl.plot([logratio[-1] + .28, logratio[-1] + .24],
            [0, 0], c='w', lw=2, clip_on=False, zorder=11)
    pl.plot([logratio[-1] + .3, logratio[-1] + .26],
            [.02, -.02], c='k', lw=2, clip_on=False, zorder=11)
    pl.plot([logratio[-1] + .26, logratio[-1] + .22],
            [.02, -.02], c='k', lw=2, clip_on=False, zorder=11)
    if savefig:
        pl.savefig('Padoa' + w + '.pdf', dpi=600)
    else:
        pl.show()
    return choice

### some exp data ###

offers3e = np.array(
    [[0, 1], [1, 3], [1, 2], [1, 1], [2, 1], [3, 1], [4, 1], [6, 1], [2, 0]])
ratio3e = 2.2
pB3e = [0, 0, 0, 0, .3570, .8810, .9494, 1, 1]
f3e = [0, .14, 0, .43, .65, 1.88, 3.99, 11.1, 1.17]
fSEM3e = [0, 0, 0, 0, 0, 0, .48, 1.05, 0]

offers3b = np.array(
    [[0, 1], [1, 3], [1, 2], [1, 1], [2, 1], [3, 1], [4, 1], [6, 1], [10, 1], [2, 0]])
ratio3b = 2.5
pB3b = [0, 0, 0, 0, .1392, .7848, .9443, .9595, 1, 1]
f3b = [12.02, 36.47, 29.26, 8.66, 5.53, 8.42, 15.39, 24.61, 36.71, 5.21]
fSEM3b = [2.36, 5.29, 5.13, 1.96, 1.36, 1.68, 2.89, 4.09, 5.17, 1.20]


##### simulate #####

s = {}
s['3e'] = sim(u0, rate, reset, tm, offers3e, ratio3e, tdec, trise, tdly, runs)
s['3b'] = sim(u0, rate, reset, tm, offers3b, ratio3b, tdec, trise, tdly, runs)
vr = np.arange(.1, 1, .1)
s['RT'] = sim(u0, rate, reset, tm,  5 * np.array([1. / (1 + vr), 1. / (1 + 1. / vr)]).T,
              1, tdec, trise, tdly, runs, seedoff=runs * 500)

# try:
#     x = np.load('sim3e.npy')
# except IOError:
#     x = sim(u0, rate, reset, tm, offers3e, ratio3e, tdec, trise, tdly, runs)
#     np.save('sim3e.npy', x)

# try:
#     x = np.load('sim3b.npy')
# except IOError:
#     x = sim(u0, rate, reset, tm, offers3b, ratio3b,
#             tdec, trise, tdly, runs)
#     np.save('sim3b.npy', x)

# try:
#     x = np.load('simRT.npy')
# except IOError:
#     vr = np.arange(.1, 1, .1)
#     x = sim(u0, rate, reset, tm,  5 * np.array([1. / (1 + vr), 1. / (1 + 1. / vr)]).T,
#             1, tdec, trise, tdly, runs, seedoff=runs * 500)
#     np.save('simRT.npy', x)


#
#
#
#
#
#


##### evaluate #####

#### plot firing rate in [0,500ms] ####
rate3e = 4.4
pop3e = rate / rate3e
rate3b = 15
pop3b = rate / rate3b

x = s['3e']
y = 2 * np.transpose(x[:, :, :int(500 / step), :2].sum(axis=2), (1, 0, 2))
pl.figure()
pl.plot(range(1, len(offers3e) - 1), f3e[1:-1], '--', color=col[0],
        marker='o', ms=30, mfc='w', mec=col[0], mew=3, zorder=10)
pl.plot(range(1, len(offers3e) - 1), y.mean(axis=0)[1:len(offers3e) - 1, 0] / pop3e,
        color=col[1], marker='o', ms=22, mec=col[1], zorder=11)
pl.errorbar(range(len(offers3e)), f3e, yerr=fSEM3e, fmt='o', ms=30, color=col[0],
            mfc='w', mec=col[0], mew=3, capthick=2, zorder=10, clip_on=False)
pl.errorbar(range(len(offers3e)), y.mean(axis=0)[:, 0] / pop3e,
            yerr=y.std(axis=0)[:, 0] / np.sqrt(len(y)) / pop3e, fmt='o', ms=22,
            color=col[1], mec=col[1], capthick=2, clip_on=False, zorder=11)
pl.xticks(range(len(offers3e)), [
          str(a[0]) + ':' + str(a[1]) for i, a in enumerate(offers3e)])
tt = pl.gca().get_xaxis().majorTicks
for i in range(0, len(tt), 2):
    tt[i].set_pad(40)
for i in range(1, len(tt), 2):
    tt[i].set_pad(13)
pl.yticks([0, 10], [0, 10])
pl.xlim(-.3, len(offers3e) - .7)
pl.ylim(0, 14)
pl.xlabel('offers (\#B:\#A)', labelpad=-2)
pl.ylabel('Firing rate [Hz]', y=.45)
pl.text(.05, .83, 'A=' + str(ratio3e) + 'B', transform=pl.gca().transAxes)
pl.subplots_adjust(.17, .28, .98, .99)
simpleaxis(pl.gca())
lg = pl.legend(
    ['Experiment', 'Model'], bbox_to_anchor=(.63, .85), handlelength=2, frameon=False)
if savefig:
    pl.savefig('Padoa3e2.pdf', dpi=600)
else:
    pl.show()

x = s['3b']
y = 2 * np.transpose(x[:, :, :int(500 / step), :2].sum(axis=2), (1, 0, 2))
pl.figure()
pl.plot(range(1, len(offers3b) - 1),
        f3b[1:-1], '--', color=col[0], zorder=10, clip_on=False)
pl.errorbar(range(len(offers3b)), f3b, yerr=fSEM3b, fmt='o', ms=30, color=col[0],
            mfc='w', mec=col[0], mew=3, capthick=2, zorder=10, clip_on=False)
pl.errorbar(range(len(offers3b)), y.mean(axis=0).sum(axis=1) / pop3b,
            yerr=y.std(axis=0).sum(axis=1) / np.sqrt(len(y)) / pop3b, fmt='o',
            ms=22, color=col[1], mec=col[1], capthick=2, clip_on=False, zorder=11)
pl.plot(range(1, len(offers3b) - 1), y.mean(axis=0).sum(axis=1)
        [1:len(offers3b) - 1] / pop3b, color=col[1], clip_on=False, zorder=11)
pl.xticks(range(len(offers3b)), [
          str(a[0]) + ':' + str(a[1]) for i, a in enumerate(offers3b)])
tt = pl.gca().get_xaxis().majorTicks
for i in range(0, len(tt), 2):
    tt[i].set_pad(40)
pl.yticks([0, 20, 40], [0, 20, 40])
pl.xlim(-.3, len(offers3b) - .7)
pl.ylim(0, 43)
pl.xlabel('offers (\#B:\#A)', labelpad=-2)
pl.ylabel('Firing rate [Hz]', y=.45)
pl.text(.35, .83, 'A=' + str(ratio3b) + 'B', transform=pl.gca().transAxes)
pl.subplots_adjust(.17, .28, .98, .99)
simpleaxis(pl.gca())
if savefig:
    pl.savefig('Padoa3b2.pdf', dpi=600)
else:
    pl.show()


rt3eRaw = getRT('3e')
choice3eRT = plotChoice('3e')

rt3bRaw = getRT('3b')
choice3bRT = plotChoice('3b')


# eval extra simulations for RT
x = np.transpose(s['RT'], (1, 0, 2, 3))
rtRaw = np.array([[get_t(a)[0] for a in x[r]]
                  for r in range(len(x))])
offers = np.array([np.arange(.1, 1, .1), np.ones(9)]).T
rt, reg = n_score(rtRaw, offers, 1)


def fit(y, a, b, c):
    return np.exp(y / a) / b + c
popt, pcov = curve_fit(fit, np.sort(rt[1]), step *
                       np.sort(rtRaw.reshape(-1)), p0=[2, .025, 62])

fig = pl.figure(figsize=(8, 6.5))
# boxplot
ax = fig.add_axes([.5, .24, .49, .67])
pl.rc('lines', lw=3)
bp = ax.boxplot(step * rtRaw)
for i in range(9):
    pl.setp(bp['boxes'][i], color=col2[i])
    pl.setp(bp['whiskers'][2 * i], color=col2[i])
    pl.setp(bp['whiskers'][2 * i + 1], color=col2[i])
    pl.setp(bp['fliers'][i], color=col2[i])
pl.rc('lines', lw=4)
ax.set_xlim(0, 10)
pl.xticks([1, 5, 9], [.1, .5, .9])
pl.yticks([160, 260, 360], [100, 200, 300])
pl.xlabel('Value ratio')
pl.ylabel('Decision time [ms]', y=.99, rotation=0, labelpad=-70)
simpleaxis(pl.gca())
# histogram of raw decision times
ax3 = ax.twiny()
ax3.hist(step * rtRaw.reshape(-1), 20,
         color=col[7], orientation='horizontal')
ax3.set_xlim(0, 200 * runs / 30.)
ax3.set_ylim(60, step * np.max(rtRaw) - 30)
pl.xticks([])
ax3.spines['left'].set_visible(False)
simpleaxis(ax3)
# histogram of nscores
ax2 = fig.add_axes([.01, .24, .49, .67])
ax2.hist(-rt[1], 20, color=col[7])
ax2.set_xlim(-3, 3)
pl.xticks([-2, 0, 2], [2, 0, -2])
pl.yticks([])
pl.ylim(0, 200 * runs / 30.)
pl.xlabel('Normalized\nreaction time', labelpad=0, linespacing=.9)
ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
ax.patch.set_visible(False)  # hide the 'canvas'
ax2.spines['left'].set_visible(False)
simpleaxis(ax2)
# function nscore(RT) and logarithmic fit
ax4 = ax2.twinx()
ax4.plot(-np.linspace(-3, 3, 100), map(lambda a: fit(a, *popt), np.linspace(-3, 3, 100)),
         '--', color='black')
ax4.plot(-np.sort(rt[1]), step * np.sort(rtRaw.reshape(-1)), color=col[1])
ax4.set_xlim(-3, 3)
ax4.set_ylim(60, step * np.max(rtRaw) - 30)
pl.yticks([])
ax4.spines['left'].set_visible(False)
simpleaxis(ax4)
if savefig:
    pl.savefig('RT(rho).pdf', dpi=600)
else:
    pl.show()


n = rt[1].reshape((runs, -1))
pl.figure()
pl.errorbar(offers[:, 0], n.mean(axis=0), yerr=n.std(axis=0) / np.sqrt(runs), fmt='s',
            ms=22, color=col[1], mec=col[1], capthick=2, clip_on=False, zorder=11)
pl.plot([0, 1], [reg[1], sum(reg)], color=col[1], lw=5)
pl.xlim(0, 1)
pl.xticks([0, 0.5, 1], [0, 0.5, 1.0])
pl.xlabel('Value ratio')
pl.ylabel('Norm. reaction time', y=.44)
pl.subplots_adjust(.17, .22, .955, .99)
simpleaxis(pl.gca())
pl.ylim(-1.2, 1.2)
pl.yticks([-1, 0, 1], [-1, 0, 1])
if savefig:
    pl.savefig('RT.pdf', dpi=600)
else:
    pl.show()


pl.figure(figsize=(5, 3.75))
pl.hist((n - n.mean(0)[np.newaxis, :]).ravel(),
        np.arange(-3.25, 3.5, .5), normed=True, color=col[1])
pl.xticks([-2, 0, 2], [-2, 0, 2])
pl.xlabel('Residual')
simpleaxis(pl.gca())
pl.gca().axes.get_yaxis().set_visible(False)
pl.subplots_adjust(.17, .3, .99, .99)
if savefig:
    pl.savefig('residuals.pdf', dpi=600)
else:
    pl.show()


res2 = np.array(s['RT'].mean(1))
res2[:, 1, 0] = 1. / 3
fig = pl.figure()
for r in range(9):
    g = accumulate(
        res2[r, int(tdly / step):, 0] - res2[r, int(tdly / step):, 1])
    pc = np.zeros(5000)
    for i in range(int(tdly / step), 5000):
        pc[i] = g.next()
    pl.plot(np.arange(0, 1000, step), pc, color=col2[r], clip_on=False)
pl.plot([0, 300], [thresh, thresh], c='r')
pl.plot([60, 60], [-.5, thresh + .5], '--', color='black')
# labels & layout
pl.xlabel('Time from offer [s]')
pl.ylabel('Spike count difference', y=.44)
pl.xticks([0, 100, 200], [0, .1, .2])
pl.xlim(0, 300)
pl.yticks([0, 5, 10], [0, 5, 10])
pl.ylim(-.5, thresh + .5)
simpleaxis(pl.gca())
pl.subplots_adjust(.17, .22, .99, .99)
if savefig:
    pl.savefig('Delta(t).pdf', dpi=600)
else:
    pl.show()


### activity profiles ###

errx = -5.2
fac = .022

res = sim(.3, 1000, 0., 25., np.array([[.97, 1], [1.07, 1], [1.3, 1]]),
          1, tdec, trise, tdly, 30).mean(1)
pre = np.zeros((int(250 / step), len(net.W)))
res = np.array([map(lambda x: smooth_spikes(x, 40, step),
                    np.vstack((pre, res[j])).T) for j in range(3)])
res2 = res[:, 0, 250 * 5:1250 * 5]

fig = pl.figure()
ax = fig.add_subplot(111)
p3, = ax.plot(np.arange(0, 1000, step), fac * res2[0], color=col[6])
p2, = ax.plot(np.arange(0, 1000, step), fac * res2[1], color=col[1])
p1, = ax.plot(np.arange(0, 1000, step), fac * res2[2], color=col[2])
# labels & layout
pl.xlabel('Time from offer [s]')
pl.ylabel('Relative firing rate [Hz]', labelpad=15, y=.38)
pl.xticks([0, 500], [0, 0.5])
pl.xlim(0, 1000)
pl.yticks([0, 5], [0, 5])
pl.ylim(-1.5, 9.5)
simpleaxis(pl.gca())
# reward neuron
ax2 = ax.twinx()
ax2.plot(np.arange(0, 1250, step), smooth_spikes(
    [step * fac / 3. * (0 if t * step < tdly else
                        np.exp(-(t * step - tdly) / tdec) -
                        np.exp(-(t * step - tdly) / trise))
     for t in range(int(1250 / step))], 10, .2), color='black', zorder=-1, lw=2)
ax2.set_xlim(0, 1000)
ax2.set_ylim(0, 10)
ax2.set_yticks([0, 2])
ax2.set_yticklabels([0, 2])
ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
ax.patch.set_visible(False)  # hide the 'canvas'
# right axis  & layout
pl.plot([1000, 1000], [0, 2], c='black', lw=2, clip_on=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
lg = pl.legend([p1, p2, p3], ['High', 'Medium', 'Low'], bbox_to_anchor=(1.12, 1),
               labelspacing=.2, handletextpad=.15, handlelength=1.6, frameon=False)
fig.subplots_adjust(.17, .22, .95, .95)
if savefig:
    pl.savefig('Padoa13_2.pdf', dpi=600)
else:
    pl.show()
