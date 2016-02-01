import numpy as np
from sys import path
path.append('../')
from DPNet import DPNet


class Net(DPNet):
    """
    Network with hard coded transition and reward probabilities
    TWO-STEP-MAZE
                              R=1
                         .75|
                        ----|
                      l/ .25|
            -------S1--|      R=0
          l/          r\ .25|
           |            ----|
           |             .75|
      S0 --|                  R=1
           |              l/
           |    ---S2------|
          r\ .5/          r\
            ---|              R=0
             .5|          l/
               \---S3------|
                          r\
                              R=1
    """

    def __init__(self):
        # transition_cpt(s,a,s') = p(s'|a,s)
        # 0 = Left, 1 = Right
        transition_cpt = np.zeros((4, 2, 4), dtype=float)
        transition_cpt[0, 0, 1] = 1  # Going left in state 0 leads to state 1.
        transition_cpt[0, 1, 2] = .5  # Going right in state 0 leads to state 2.
        transition_cpt[0, 1, 3] = .5  # or state 3.
        # reward_cpt(s,a,r) = p(r|s,a)
        # 0 = False, 1 = True
        reward_cpt = np.array([[0, 0], [.75, .75], [1, 0], [0, 1]])
        DPNet.__init__(self, transition_cpt, reward_cpt)
