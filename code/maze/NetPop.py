import numpy as np
from sys import path
path.append('../')
from DPNetPop import DPNetPop


class NetPop(DPNetPop):
    """
    Network with hard coded transition and reward probabilities
    Network has loops => arbitrary long episodes possible
    Maze from Strens(2000) from Dearden, Friedman and Russell (1998).
    """

    def __init__(self, pop_size=10, set_W=True):
        # transition_cpt(s,a,s') = p(s'|a,s)
        # 0 = North, 1 = East, 2=South, 3=West
        transition_cpt = np.zeros((264, 4, 264), dtype=float)
        p = [.05, .9, .05]
        for y in range(6):
            for x in range(7):
                for a in range(4):
                    if self.get_state_nr([y, x], [0, 0, 0]) < 0:
                        continue  # wall

                    for aa in range(3):

                        b = np.mod(a + aa - 1, 4)
                        npos = self.get_next_pos([y, x], b)

                        if np.min(npos) < 0 or np.max(npos - [5, 6]) > 0\
                           or self.get_state_nr(npos, [0, 0, 0]) < 0:  # walk against wall
                            for s in self.get_state_nr([y, x]):
                                transition_cpt[s, a, s] += p[aa]
                            continue
                        npos = list(npos)
                        for f0 in [0, 1]:
                            for f1 in [0, 1]:
                                for f2 in [0, 1]:
                                    if not (npos in [[0, 2], [5, 0], [4, 6]]):
                                        transition_cpt[self.get_state_nr([y, x], [f0, f1, f2]),
                                                       a, self.get_state_nr(npos, [f0, f1, f2]), ] += p[aa]
                                    elif npos == [0, 2]:
                                        transition_cpt[self.get_state_nr([y, x], [f0, f1, f2]),
                                                       a, self.get_state_nr(npos, [1, f1, f2]), ] += p[aa]
                                    elif npos == [5, 0]:
                                        transition_cpt[self.get_state_nr([y, x], [f0, f1, f2]),
                                                       a, self.get_state_nr(npos, [f0, 1, f2]), ] += p[aa]
                                    elif npos == [4, 6]:
                                        transition_cpt[self.get_state_nr([y, x], [f0, f1, f2]),
                                                       a, self.get_state_nr(npos, [f0, f1, 1]), ] += p[aa]

        # reward_cpt(s,a,r) = p(r|s,a)
        # 0 = False, 1 = True
        reward_cpt = np.ones((264, 4))
        reward_sizes = np.zeros((264, 4))
        for f0 in [0, 1]:
            for f1 in [0, 1]:
                for f2 in [0, 1]:
                    reward_sizes[self.get_state_nr([0, 6], [f0, f1, f2])] = (f0 + f1 + f2)
                    transition_cpt[self.get_state_nr([0, 6], [f0, f1, f2])] *= 0
        DPNetPop.__init__(self, transition_cpt, reward_cpt, pop_size, reward_sizes, set_W)

    def get_state_nr(self, coords, flags=None):
        xy = np.reshape(np.arange(6 * 7), (6, 7))
        xy[0, 1] = xy[1, 1] = xy[0, 4] = xy[1, 4] = xy[3, 0] = xy[
            3, 1] = xy[3, 5] = xy[3, 6] = xy[5, 6] = -1000
        s = 0
        for y in range(6):
            for x in range(7):
                if xy[y, x] != -1000:
                    xy[y, x] = s
                    s += 1
        if flags is None:
            return [i * 33 + xy[tuple(coords)] for i in range(8)]
        else:
            return np.dot(flags, 2**np.arange(3)) * 33 + xy[tuple(coords)]

    def get_next_pos(self, coords, action):
        return np.array(coords) + np.array([[-1, 0], [0, 1], [1, 0], [0, -1]][action])

    def get_pos(self, state_nr):
        for y in range(6):
            for x in range(7):
                if state_nr in self.get_state_nr([y, x]):
                    return [y, x]
