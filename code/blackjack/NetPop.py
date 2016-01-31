import numpy as np
import pylab as pl
from sys import path
path.append('../')
from DPNetPop import DPNetPop


class NetPop(DPNetPop):
    """
    Network for blackjack a la Sutton&Barto book 5.1
    18+1(hand value+busted) x 10(face card dealer) x 2(ace) states

    """

    def __init__(self, pop_size=10, set_W=True):
        self.pdealer = np.zeros((27, 12, 2))
        for i in range(2, 10):
            self.pdealer[i, i, 0] = 1. / 13
        self.pdealer[10, 10, 0] = 4. / 13
        self.pdealer[11, 11, 1] = 1. / 13

        def draw(x):
            for card in range(2, 12):
                if card == 10:
                    if x[0] + card > 21 and x[2] > 0:
                        self.pdealer[x[0] + card - 10, x[1], x[2] -
                                     1] += self.pdealer[tuple(x)] * 4. / 13
                    else:
                        self.pdealer[x[0] + card, x[1], x[2]] += self.pdealer[tuple(x)] * 4. / 13
                elif card == 11:
                    if x[0] + card > 21:
                        self.pdealer[x[0] + 1, x[1], x[2]] += self.pdealer[tuple(x)] * 1. / 13
                    else:
                        self.pdealer[x[0] + 11, x[1], x[2] + 1] += self.pdealer[tuple(x)] * 1. / 13
                else:
                    if x[0] + card > 21 and x[2] > 0:
                        self.pdealer[x[0] + card - 10, x[1], x[2] -
                                     1] += self.pdealer[tuple(x)] * 1. / 13
                    else:
                        self.pdealer[x[0] + card, x[1], x[2]] += self.pdealer[tuple(x)] * 1. / 13
            self.pdealer[tuple(x)] = 0
        while np.sum(self.pdealer[:17]) > 0:
            for hand in range(2, 17):
                for face in range(2, 12):
                    for ace in [0, 1]:
                        if self.pdealer[hand, face, ace] > 0:
                            draw([hand, face, ace])

        def next_x(x, card):
            xx = np.array(x) + [card, 0, card == 11]
            if xx[0] > 21 and xx[-1] > 0:
                xx[0] -= 10
                xx[-1] -= 1
            if xx[0] > 21:
                xx[0] = 22
            return xx

        pxinit = np.zeros((22, 12, 2))
        for hand in range(2, 12):
            ace = 1 if hand == 11 else 0
            for face in range(2, 12):
                if hand == 10 and face == 10:
                    pxinit[hand, face, ace] = 4. / 13 * 4. / 13
                elif hand == 10 or face == 10:
                    pxinit[hand, face, ace] = 4. / 13 / 13
                else:
                    pxinit[hand, face, ace] = 1. / 13 / 13
        self.pxinit = np.zeros((22, 12, 2))
        self.pstart_state = np.zeros(380)
        for hand in range(2, 12):
            for face in range(2, 12):
                ace = 1 if hand == 11 else 0
                for card in range(2, 12):
                    nx = tuple(next_x([hand, face, ace], card))
                    self.pxinit[nx] += pxinit[hand, face, ace] * \
                        (4. / 13 if card == 10 else 1. / 13)
                    self.pstart_state[self.get_state_nr(nx)] = self.pxinit[nx]

        # transition_cpt(s,a,s') = p(s'|a,s)
        # 0 = Hit, 1 = Stick
        transition_cpt = np.zeros((380, 2, 380), dtype=float)
        for hand in range(4, 22):
            for face in range(2, 12):
                for ace in [0, 1]:
                    if ace == 1 and hand < 11:
                        continue
                    for card in range(2, 12):
                        transition_cpt[self.get_state_nr([hand, face, ace]), 0,
                                       self.get_state_nr(tuple(next_x([hand, face, ace], card)))]\
                            = 4. / 13 if card == 10 else 1. / 13

        # reward_cpt(s,a,r) = p(r|s,a)
        reward_sizes = np.zeros((380, 2, 3))  # reward can have 3 values (0,.5,1)
        reward_sizes[:, :, 1] = .5
        reward_sizes[:, :, 2] = 1
        reward_cpt = np.zeros((380, 2, 3))
        reward_cpt[:, :, 0] = 1  # default no reward
        for hand in range(4, 22):
            for face in range(2, 12):
                for ace in [0, 1]:
                    reward_cpt[self.get_state_nr([hand, face, ace]), 1] = self.get_prew(hand, face)

        DPNetPop.__init__(self, transition_cpt, reward_cpt, pop_size, reward_sizes, set_W)

    # x is list [hand value, dealer's face up card, useable ace]
    def get_state_nr(self, x):
        return (x[0] - 4) + (x[1] - 2) * 19 + x[2] * 19 * 10

    # return [p(r=0),p(r=.5),p(r=1)]
    def get_prew(self, hand, face):
        if hand > 21:
            return np.array([1, 0, 0])
        else:
            pdraw = np.sum(self.pdealer[hand, face])
            pwin = np.sum(self.pdealer[:hand, face]) + np.sum(self.pdealer[22:, face])
            ploose = np.sum(self.pdealer[hand + 1:22, face])
            return np.array([ploose, pdraw, pwin]) / np.sum(self.pdealer[:, face])

    def plot_policy(self, policy, figsize=(12, 6), fontsize=16):
        X, Y = np.array(np.meshgrid(range(2, 13), range(11, 23))) - .5
        fig = pl.figure(figsize=figsize)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        def foo(ax, ace):
            ax.pcolormesh(X, Y, np.array([[policy[self.get_state_nr([hand, face, ace])]
                                           for hand in range(11, 22)] for face in range(2, 12)]).T)
            ax.set_xlim(1.5, 11.5)
            ax.set_ylim(10.5, 21.5)
            ax.set_xticks(range(2, 12))
            ax.set_yticks(range(11, 22))
            ax.set_xticklabels(range(2, 11) + ['A'])
            ax.set_title(['No usable ace', 'Usable ace'][ace])
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)
        foo(ax1, 1)
        foo(ax2, 0)
        return fig
