import numpy as np
import cfunctions as cfn


class DPNet(object):
    """
    General network class to perfrom dynamic programming
        transition_cpt: conditional probability table of transition function
        reward_cp: conditional probability table of reward function
    """

    def __init__(self, transition_cpt, reward_cpt):
        # save the environmental parameters
        self.num_states, self.num_actions = np.shape(reward_cpt)
        self.transition_cpt = transition_cpt
        self.reward_cpt = reward_cpt
        # neuron numbering
        self.sa = np.reshape(np.arange(self.num_states * self.num_actions),
                             [self.num_states, self.num_actions])   # state-action pairs
        self.r = self.num_states * self.num_actions  # reward
        self.K = self.r + 1  # number of neurons
        # neuron label
        self.label = ["" for x in range(self.K)]
        for i in xrange(self.num_states):
            for j in xrange(self.num_actions):
                self.label[self.sa[i, j]] = "s" + str(i) + "_a" + str(j)
        self.label[self.r] = "r"
        # weights
        self.W = np.zeros((self.K, self.K), dtype=float)
        for i in xrange(self.num_states):
            for j in xrange(self.num_actions):
                for k in xrange(self.num_states):
                    for l in xrange(self.num_actions):
                        self.W[self.sa[i, j], self.sa[k, l]] = transition_cpt[i, j, k]  # sa to sa'
                        if i == k and l != j:
                            self.W[self.sa[i, j], self.sa[k, l]] = -1
                self.W[self.sa[i, j], self.r] = reward_cpt[i, j]  # sa to r
        # boolean matrix indicating plastic weights
        self.plastic = self.W >= 0

    def unlearn(self, rew=.5):
        """
        Unlearn weights to flat state transitions
        (but always change state due to mutual competition!)
        and to reward rew
        """
        for i in xrange(self.num_states):
            for j in xrange(self.num_actions):
                for k in xrange(self.num_states):
                    for l in xrange(self.num_actions):
                        self.W[self.sa[i, j], self.sa[k, l]] = 1. / (self.num_states - 1)\
                            if i != k else 0  # sa to sa'
                        if i == k and l != j:
                            self.W[self.sa[i, j], self.sa[k, l]] = -1
                self.W[self.sa[i, j], self.r] = rew  # sa to r

    def learn(self, step, eta, run, rate, T=500, r0=.5,
              tm=20, ts=2, reset=0, trials=100, start_state=0):
        np.random.seed(run)
        # maybe put this outside function to continue learning instead of fresh start
        self.unlearn(r0)
        pi = np.zeros((trials, self.num_states), dtype=int)
        seq = [[None]] * trials  # sequences might have differnt length -> list instead array
        W = np.zeros((trials, self.K, self.K), dtype=float)
        for t in xrange(trials):
            res = cfn.run(self.W, step, rate, T, tm, ts, reset, run)
            pi[t] = self.get_policy(step, res[0])
            seq[t] = self.sample4Pi(pi[t], start_state)
            self.update_weights(seq[t], eta)
            W[t] = self.W
        return np.array([pi, seq, W])
