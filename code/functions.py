import numpy as np
import pylab as pl
from scipy.signal import convolve


def init_fig():
    pl.rc('figure', facecolor='white', dpi=90, frameon=False)
    pl.rc('font', size=44, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
    pl.rc('lines', lw=4)
    pl.rc('text', usetex=True)
    pl.rc('legend', **{'fontsize': 32})
    pl.rc('axes', linewidth=2)
    pl.rc('xtick.major', size=10, width=1.5)
    pl.rc('ytick.major', size=10, width=1.5)


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else pl.gca()
    if color is None:
        # color = ax._get_lines.color_cycle.next()
        color = ax._get_lines.prop_cycler.next()['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def smooth_spikes(spikes, sigma, step):
    return convolve(spikes, [1. / np.sqrt(2 * np.pi) / sigma *
                             1000 * np.exp(-i ** 2 / (2 * (sigma / step) ** 2))
                             for i in range(int(-5 * sigma / step),
                                            int(5 * sigma / step + 1))], 'same')


def accumulate(values, start=0):
    for v in values:
        start += v
        yield start
