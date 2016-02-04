import numpy as np
from sys import path
path.append('../')
from DPNetPop import DPNetPop


class NetPop(DPNetPop):
    """
    Network with hard coded transition and reward probabilities
    ONE-STEP-MAZE
    """

    def __init__(self, pop_size=10):
        # transition_cpt(s,a,s') = p(s'|a,s)
        # 0 = Left, 1 = Right
        transition_cpt = np.zeros((1, 2, 1), dtype=float)
        # reward_cpt(s,a,r) = p(r|s,a)
        # 0 = False, 1 = True
        reward_cpt = np.array([[0, .1]])
        DPNetPop.__init__(self, transition_cpt, reward_cpt, pop_size)
