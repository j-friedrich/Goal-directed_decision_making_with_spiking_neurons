import numpy as np
from sys import path
path.append('../')
from DPNetPop import DPNetPop


class NetPop(DPNetPop):

    """
    Network with hard coded transition and reward probabilities
    TWO-STEP-MAZE
                        ----| R=4
                      l/
            -------S1--|
          l/          r\
           |            ----| R=0
           |
      S0 --|
           |
           |            ----| R=2
          r\          l/
            -------S2--|
                      r\
                        ----| R=3
    """

    def __init__(self, pop_size=1, set_W=True):
        # transition_cpt(s,a,s') = p(s'|a,s)
        # 0 = Left, 1 = Right
        transition_cpt = np.zeros((3, 2, 3), dtype=float)
        transition_cpt[0, 0, 1] = 1  # Going left in state 0 leads to state 1.
        transition_cpt[0, 1, 2] = 1  # Going right in state 0 leads to state 2.
        # reward_cpt(s,a,r) = p(r|s,a)
        # 0 = False, 1 = True
        reward_cpt = np.array([[0, 0], [1, 0], [.5, .75]])
        DPNetPop.__init__(self, transition_cpt, reward_cpt, pop_size, set_W=set_W)
