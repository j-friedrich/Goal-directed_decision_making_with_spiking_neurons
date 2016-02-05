import numpy as np
import pylab as pl
from sys import argv, path
path.append('../')
from NetPop import NetPop, NetPopPredict
from functions import simpleaxis, smooth_spikes

savefig = False if len(argv) == 1 else True

pl.rc('figure', facecolor='white', dpi=90, frameon=False)
pl.rc('font', size=44, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
pl.rc('lines', lw=4)
pl.rc('text', usetex=True)
pl.rc('legend', **{'fontsize': 30})
pl.rc('axes', linewidth=2)
pl.rc('xtick.major', size=10, width=1.5)
pl.rc('ytick.major', size=10, width=1.5)


# colors for colorblind from
# http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
colRT = ["#009E73", "#0072B2", "#D55E00", "#E69F00",
         "#56B4E9", "#F0E442", "#CC79A7", "#999999"]

# orange to green to cyan:
col = map(tuple, [np.array([.902, .624, 0]) +
                  (np.array([0, .62, .451]) -
                   np.array([.902, .624, 0])) * i / 4 for i in range(5)]) +\
    map(tuple, [np.array([0, .62, .451]) +
                (np.array([.337, .706, .914]) -
                 np.array([0, .62, .451])) * i / 3 for i in range(1, 4)])


def runpopU(W, uinit, step, pop_size, rate=100,
            T=700, tm=50, ts=2, reset=3, delay=0):
    K = len(W)
    spikes = np.zeros((T / step, K))
    spikes[0, K - pop_size:] = 1
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
        u[K - pop_size:] = rate / 1000. * \
            (.1 + .65 * np.exp(-(t * step - 250) ** 2 / tau_r ** 2))
        spikes[t] = np.random.rand(K) < step * u
    return spikes, u


def simulate(seed, na=2, rate=100, predict=False):
    net = NetPopPredict() if predict else NetPop(na)
    W = np.copy(net.W)
    W[:-1, :-1] *= gamma
    W -= (1 - gamma) * net.competition
    u0 = np.linspace(-2, .04, len(W))
    np.random.seed(seed)
    s, u = runpopU(W * 1. / popsize, u0, step, popsize, rate)
    pre = np.zeros((int(250 / step), len(W)))
    for t in [-40]:
        pre[int(t / step), range(-2, -2 - na, -1)] = 1
    for t in [-240, -200, -160, -120, -80, -10, -2, -1]:
        pre[int(t / step), range(-2 - na, -2 - 2 * na, -1)] = .1
    pres = np.vstack((pre, s))
    tmp = np.array([np.mean(np.array([smooth_spikes(pres[:, j * popsize + i], 40, .2)
                                      for i in range(popsize)]), axis=0)
                    for j in range(len(W) - na, len(W) - 8 * na - 1, -na)])[:, 250 / step:750 / step]
    tmp2 = np.array([np.mean(np.array([smooth_spikes(pres[:, j * popsize + i], 40, .2)
                                       for i in range(popsize)]), axis=0)
                     for j in range(len(W) - na - 1, len(W) - 8 * na - 2, -na)])[:, 250 / step:750 / step]
    return [np.array(map(np.argmax, tmp)) * step, tmp, tmp2, s]


def accumulate(values, start=0):
    for v in values:
        start += v
        yield start


def get_t(x, c):
    g0 = accumulate(x)
    th = thresh * gamma ** c
    for t in range(3500):
        if g0.next() >= th:
            break
    return t


gamma = .7
popsize = 1
step = .2
tau_r = 80.


###  simulation ###
res = map(simulate, range(30))
Ratels = np.array(map(lambda a: a[1], res))

res3 = map(lambda a: simulate(a, 3), range(30))
Ratels3 = np.array(map(lambda a: a[1], res3))

res4 = map(lambda a: simulate(a, 4), range(30))
Ratels4 = np.array(map(lambda a: a[1], res4))


### plot neural activities from simulation ###

pl.figure()
for c in range(8):
    pl.plot(Ratels.mean(axis=0)[c], color=col[c])
pl.xlabel('Time from target [ms]', labelpad=0)
pl.ylabel('Firing rate [Hz]', labelpad=10)
pl.xticks([0, 250 / step, 500 / step], [0, 250, 500])
pl.yticks([0, 20, 40, 60, 80], [0, 20, 40, 60, 80])
pl.xlim(0, 500 / step)
pl.ylim(0, 64)
pl.plot(smooth_spikes([step * 100 / 1000. *
                       (.1 + .65 * np.exp(-(t * step - 500) ** 2 / tau_r ** 2))
                       for t in range(int(1000 / step))], 40, .2)[250 / step:750 / step],
        color='black', zorder=-1, lw=2)
pl.subplots_adjust(.18, .21, .945, .99)
simpleaxis(pl.gca())
if savefig:
    pl.savefig('Sohn.pdf', dpi=600)
else:
    pl.show()


pl.figure()
pl.plot(Ratels.mean(axis=0)[3], color=col[3])
l, = pl.plot(Ratels3.mean(axis=0)[3], '--', color=col[3])
l.set_dashes([10, 10])
l, = pl.plot(Ratels4.mean(axis=0)[3], ':', color=col[3])
l.set_dashes([3, 3])
pl.xlabel('Time from target [ms]', labelpad=0)
pl.ylabel('Firing rate [Hz]', labelpad=10)
pl.xticks([0, 250 / step, 500 / step], [0, 250, 500])
pl.yticks([0, 10, 20], [0, 10, 20])
pl.xlim(0, 500 / step)
pl.ylim(0, 25)
lg = pl.legend(range(2, 5), title='  Number\nof actions',
               bbox_to_anchor=(.37, 1.04), handlelength=2.25, handletextpad=.2)
lg.get_title().set_fontsize(30)
lg.draw_frame(False)
pl.subplots_adjust(.18, .21, .945, .99)
simpleaxis(pl.gca())
if savefig:
    pl.savefig('Sohn2.pdf', dpi=600)
else:
    pl.show()


### reaction times ###

# exp data:
RTmonkeyD8 = [236.66666666666663, 251.90476190476187, 271.90476190476187,
              309.5238095238095, 340, 348.0952380952381, 381.4285714285714, 362.38095238095235]
RTmonkeyD2 = [277.1596115580618, 304.90751838300906]
RTmonkeyD4 = [257.5674075509715, 286.3367690340928,
              300.64521281848715, 356.4476349451328]
RTmonkeyD6 = [248.14841026479337, 259.5785122900019, 278.4211809027685,
              329.72126283240135, 359.63278200480795, 358.28459846717465]
RTmonkeyJ8 = [144.44444444444443, 188.88888888888889, 235.55555555555557, 246.1111111111111,
              279.44444444444446, 287.22222222222223, 306.1111111111111, 326.1111111111111]
RTmonkeyD = np.array(RTmonkeyD8) + np.array(RTmonkeyD6 +
                                            [0] * 2) + np.array(RTmonkeyD4 + [0] * 4) + np.array(RTmonkeyD2 + [0] * 6)
RTmonkeyD = RTmonkeyD / np.array([4, 4, 3, 3, 2, 2, 1, 1])


# plot RT from threshold:
s = np.array(map(lambda a: simulate(a, 2, 1000)[3], range(30)))
sdiff = s[:, :, range(19, 4, -2)] - s[:, :, range(18, 3, -2)]
t = np.zeros((8, 30))
thresh = 70
for c in range(8):
    t[c] = map(lambda x: get_t(x, c), sdiff[:, :, c])
t *= step
pl.figure()
pl.plot(RTmonkeyD, '--s',
        color=colRT[0], marker='s', ms=30, mfc='w', mec=colRT[0], mew=3)
pl.plot(t.mean(axis=1), color=colRT[1], marker='s', ms=22, mec=colRT[1])
pl.errorbar(range(8), t.mean(axis=1), yerr=t.std(axis=1) / np.sqrt(30),
            fmt='s', ms=22, color=colRT[1], mec=colRT[1], capthick=2, zorder=10)
pl.xlabel('NRM')
pl.ylabel('Reaction time [ms]', labelpad=5)  # , position=(0,.45))
pl.xticks(range(8), range(8))
pl.yticks(*[[250, 300, 350, 400]] * 2)
pl.ylim(220, 420)
pl.xlim(-.3, 7.3)
pl.subplots_adjust(.2, .21, .99, .99)
simpleaxis(pl.gca())
lg = pl.legend(['Experiment', 'Model'],
               bbox_to_anchor=(.6, 1.04), handlelength=2.25)
lg.draw_frame(False)
if savefig:
    pl.savefig('Sohn_RT.pdf', dpi=600)
else:
    pl.show()


### neural activities PAI ###

botv = np.genfromtxt('PAI/SohnBP2.txt')
botv3 = np.genfromtxt('PAI/SohnBP3.txt')
botv4 = np.genfromtxt('PAI/SohnBP4.txt')
pl.figure()
for i in range(8):
    pl.plot(np.r_[[.5], botv[i, :-1]] * [10 + 55 * np.exp(-t ** 2 / 4. ** 2)
                                         for t in range(-10, 190)], color=col[i])
pl.xlabel('Iteration')
pl.ylabel('Firing rate [Hz]', labelpad=10)
pl.xticks([0, 10, 20], [0, 10, 20])
pl.yticks([0, 20, 40, 60], [0, 20, 40, 60])
pl.ylim(0, 65)
pl.xlim(0, 20)
pl.subplots_adjust(.18, .21, .96, .99)
simpleaxis(pl.gca())
if savefig:
    pl.savefig('SohnPAI.pdf', dpi=600)
else:
    pl.show()

pl.figure()
pl.plot(np.r_[[1. / 2], botv[3, :-1]] * [10 + 55 * np.exp(-t ** 2 / 4. ** 2)
                                         for t in range(-10, 190)], color=col[3])
l, = pl.plot(np.r_[[1. / 3], botv3[3, :-1]] * [10 + 55 * np.exp(-t ** 2 / 4. ** 2)
                                               for t in range(-10, 190)], '--', color=col[3])
l.set_dashes([10, 10])
l, = pl.plot(np.r_[[1. / 4], botv4[3, :-1]] * [10 + 55 * np.exp(-t ** 2 / 4. ** 2)
                                               for t in range(-10, 190)], '--', color=col[3])
l.set_dashes([3, 3])
pl.xlabel('Iteration')
pl.ylabel('Firing rate [Hz]', labelpad=10)
pl.xticks([0, 10, 20], [0, 10, 20])
pl.yticks([0, 10, 20, 30], [0, 10, 20, 30])
pl.ylim(0, 37)
pl.xlim(0, 20)
pl.subplots_adjust(.18, .21, .96, .99)
simpleaxis(pl.gca())
if savefig:
    pl.savefig('SohnPAI2.pdf', dpi=600)
else:
    pl.show()
#
#
#
#
#
#
#
#
#
#
###  simulation for prediction ###

res = map(lambda run: simulate(run, predict=True), range(30))
RTls = np.array(map(lambda a: a[0], res))
Ratels = np.array(map(lambda a: a[1], res))


### plot neural activities from simulation ###

pl.figure(figsize=(8, 6.3))
for c in range(8):
    pl.plot(Ratels.mean(axis=0)[c], color=col[c])
pl.xlabel('Time from target [ms]')
pl.ylabel('Firing rate [Hz]', labelpad=10)
pl.xticks([0, 250 / step, 500 / step], [0, 250, 500])
pl.yticks([0, 20, 40, 60], [0, 20, 40, 60])
pl.xlim(0, 500 / step)
pl.ylim(0, 83.5)
lg = pl.legend(range(8), title='NRM', bbox_to_anchor=(.82, 1.14), ncol=1, loc='upper left',
               labelspacing=.15, columnspacing=0, handletextpad=.2, handlelength=1.5)
lg.get_title().set_fontsize(30)
lg.draw_frame(False)
pl.plot(smooth_spikes([step * 100 / 1000. * (.1 + .65 * np.exp(-(t * step - 500) ** 2 / tau_r ** 2))
                       for t in range(int(1000 / step))], 40, .2)[250 / step:750 / step],
        color='black', zorder=-1, lw=2)
pl.subplots_adjust(.18, .21, .945, .95)
simpleaxis(pl.gca())
if savefig:
    pl.savefig('SohnPredict.pdf', dpi=600)
else:
    pl.show()


# plot RT from threshold:
s = np.array(map(lambda a: simulate(a, 2, 1000, predict=True)[3], range(30)))
sdiff = s[:, :, range(19, 4, -2)] - s[:, :, range(18, 3, -2)]
t = np.zeros((8, 30))
thresh = 70
for c in range(8):
    t[c] = map(lambda x: get_t(x, c), sdiff[:, :, c])
t *= step
fig = pl.figure(figsize=(8.8, 6.3))
ax = fig.add_subplot(111)
pl.plot(t.mean(axis=1), color=colRT[1], marker='s', ms=22, mec=colRT[1])
pl.errorbar(range(8), t.mean(axis=1), yerr=t.std(axis=1) / np.sqrt(30),
            fmt='s', ms=22, color=colRT[1], mec=colRT[1], capthick=2, zorder=10)
pl.xlabel('NRM')
pl.ylabel('Reaction time [ms]', labelpad=-5)
pl.xticks(range(8), range(8))
pl.yticks([200, 250, 300], [200, 250, 300])
pl.ylim(160, 325)
pl.xlim(-.3, 7.3)
simpleaxis(pl.gca())
MAXls = np.array([[Ratels[run, nrm, int(RTls[run, nrm] / step)]
                   for nrm in range(len(RTls[0]))] for run in range(len(RTls))])
pl.ylabel('Reaction time [ms]', labelpad=0, color=colRT[1])
ax2 = ax.twinx()
ax2.plot(MAXls.mean(axis=0), color=colRT[6], marker='o', ms=22, mec=colRT[6])
ax2.errorbar(range(8), MAXls.mean(axis=0), yerr=MAXls.std(axis=0) / np.sqrt(len(MAXls)),
             fmt='o', ms=22, color=colRT[6], mec=colRT[6], capthick=2)
ax2.set_xlim(-.3, 7.3)
ax2.set_ylim(0, 83.5)
ax2.set_yticks([0, 20, 40, 60])
ax2.set_yticklabels([0, 20, 40, 60])
ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
ax.patch.set_visible(False)  # hide the 'canvas'
# right axis  & layout
ax2.spines['top'].set_visible(False)
ax2.set_ylabel('Peak firing rate [Hz]', color=colRT[6], labelpad=5, y=.49)
fig.subplots_adjust(.175, .21, .85, .95)
if savefig:
    pl.savefig('SohnPredict_RT.pdf', dpi=600)
else:
    pl.show()
