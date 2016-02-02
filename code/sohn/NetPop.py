import numpy as np
from sys import path
path.append('../')
from DPNetPop import DPNetPop


class NetPop(DPNetPop):
    """
    Network with hard coded transition and reward probabilities
    Network has loops => arbitrary long episodes possible
    Just a chain to get plot a la Sohn&Lee (2007) Fig7.
    """

    def __init__(self, n_actions=2, pop_size=1, set_W=True):
        # transition_cpt(s,a,s') = p(s'|a,s)
        transition_cpt = np.zeros((10, n_actions, 10), dtype=float)
        transition_cpt[:, :, 0] = 1
        transition_cpt[:, 1, 0] = 0
        for s in range(9):
            transition_cpt[s, 1, s + 1] = 1
        # reward_cpt(s,a,r) = p(r|s,a)
        # 0 = False, 1 = True
        reward_cpt = np.zeros((10, n_actions))
        reward_cpt[9, 1] = 1
        DPNetPop.__init__(
            self, transition_cpt, reward_cpt, pop_size, set_W=set_W)


class NetPopPredict(DPNetPop):
    """
    Network with hard coded transition and reward probabilities
    Network has loops => arbitrary long episodes possible
    A chain with intermediate reward.
    """

    def __init__(self, pop_size=1, set_W=True):
        # transition_cpt(s,a,s') = p(s'|a,s)
        transition_cpt = np.zeros((10, 2, 10), dtype=float)
        for s in range(9):
            transition_cpt[s, 1, s + 1] = 1
        for s in range(6):
            transition_cpt[s, 0, 0] = 1
        for s in range(6, 10):
            transition_cpt[s, 0, 6] = 1

        # reward_cpt(s,a,r) = p(r|s,a)
        # 0 = False, 1 = True
        reward_cpt = np.zeros((10, 2))
        reward_cpt[9, 1] = 1
        reward_cpt[5, 1] = 1
        DPNetPop.__init__(
            self, transition_cpt, reward_cpt, pop_size, set_W=set_W)
