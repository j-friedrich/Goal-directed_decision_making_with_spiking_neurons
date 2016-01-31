import numpy as np
import random
from sys import stdout
import cfunctions as cfn


class DPNetPop(object):
    """
    General network class to perfrom dynamic programming
        transition_cpt: conditional probability table of transition function
        reward_cp: conditional probability table of reward function
        pop_size: number of neurons per state-action
        set_W: set weights and competition. Might be 'False' for evaluation to save time
    """

    def __init__(self, transition_cpt, reward_cpt, pop_size, reward_sizes=None, set_W=True):
        # save the environmental parameters
        self.num_states, self.num_actions = np.shape(reward_cpt)[:2]
        self.transition_cpt = transition_cpt
        self.reward_cpt = reward_cpt
        self.reward_sizes = np.ones(np.shape(reward_cpt)) if reward_sizes is None else reward_sizes
        self.pop_size = pop_size
        # neuron numbering
        self.sa = np.reshape(np.arange(self.num_states * self.num_actions * pop_size),
                             [self.num_states, self.num_actions, pop_size])   # state-action pairs
        self.r = self.num_states * self.num_actions * pop_size + np.arange(pop_size)  # reward
        self.K = self.r[-1] + 1  # number of neurons
        # weights
        self.W = np.zeros((self.K, self.K), dtype=float)
        self.competition = np.zeros((self.K, self.K), dtype=int)
        if set_W:
            for i in xrange(self.num_states):
                for j in xrange(self.num_actions):
                    for m in xrange(pop_size):
                        for k in xrange(self.num_states):
                            for l in xrange(self.num_actions):
                                for n in xrange(pop_size):
                                    self.W[self.sa[i, j, m], self.sa[k, l, n]
                                           ] = transition_cpt[i, j, k]  # sa to sa'
                                    if i == k and l != j:
                                        self.W[self.sa[i, j, m], self.sa[k, l, n]] -= 1
                                        self.competition[self.sa[i, j, m], self.sa[k, l, n]] = 1
                        self.W[self.sa[i, j, m], self.r] = np.dot(
                            reward_cpt[i, j], self.reward_sizes[i, j])  # sa to r

    def unlearn(self, rew=.5, mode='fix'):
        """
        Unlearn weights to flat state and transitions. An additional terminal state is assumed,
            thus unlearn to initial average transition probabilities p=1/(num_states+1)
        Unlearn weight to reward rew
        mode determines the weight distribution between neuron pools (rand, unipre, unipost, default=fix)

        """
        for i in xrange(self.num_states):
            for j in xrange(self.num_actions):
                for m in xrange(self.pop_size):
                    for k in xrange(self.num_states):
                        for l in xrange(self.num_actions):
                            for n in xrange(self.pop_size):
                                if mode == 'rand':
                                    self.W[self.sa[i, j, m], self.sa[k, l, n]] = 2. / \
                                        (self.num_states + 1) * np.random.rand()  # sa to sa'
                                elif mode == 'unipre':
                                    self.W[self.sa[i, j, m], self.sa[k, l, n]] = 2. / \
                                        (self.num_states + 1) * n / \
                                        (self.pop_size - 1)  # sa to sa'
                                elif mode == 'unipost':
                                    self.W[self.sa[i, j, m], self.sa[k, l, n]] = 2. / \
                                        (self.num_states + 1) * m / \
                                        (self.pop_size - 1)  # sa to sa'
                                else:
                                    self.W[self.sa[i, j, m], self.sa[k, l, n]] = 1. / \
                                        (self.num_states + 1)  # sa to sa'
                                if i == k and l != j:
                                    self.W[self.sa[i, j, m], self.sa[k, l, n]] -= 1
                    if mode == 'rand':
                        self.W[self.sa[i, j, m], self.r] = 2 * rew * \
                            np.random.rand(self.pop_size)  # sa to r
                    elif mode == 'unipre':
                        self.W[self.sa[i, j, m], self.r] = 2 * rew * \
                            np.linspace(0, 1, self.pop_size)  # sa to r
                    elif mode == 'unipost':
                        self.W[self.sa[i, j, m], self.r] = 2. * \
                            rew * m / (self.pop_size - 1)  # sa to r
                    else:
                        self.W[self.sa[i, j, m], self.r] = rew  # sa to r


    def calc_Qvalue(self, gamma=1, accuracy=1e-20, max_iter=None):
        Q = np.zeros(np.shape(self.reward_cpt)[:2])
        # terminal states
        for s in np.array(np.where(np.sum(self.transition_cpt, axis=2) == 0)).T:
            Q[tuple(s)] = np.dot(self.reward_cpt[tuple(s)], self.reward_sizes[tuple(s)])
        # other states
        counter = 0  # avoid endless loop, return nans if infinite horizon
        while True:
            counter += 1
            Qold = np.copy(Q)
            for s in range(len(Q)):
                for a in range(len(Q[0])):
                    if np.sum(self.transition_cpt[s, a]) != 0:
                        Q[s, a] = np.dot(self.reward_cpt[s, a], self.reward_sizes[s, a])\
                            + gamma * np.dot(self.transition_cpt[s, a], np.max(Q, axis=1))
            if np.max(np.abs(Qold - Q)) < accuracy or \
               (max_iter is not None and counter > max_iter):
                break
        return Q

    def estimate_value(self, step, S, T=None):
        if T is None:
            return np.mean(np.sum(np.sum(S, axis=0)[self.sa], axis=1), axis=1) / np.sum(S[:, -1])
        else:
            return np.mean(np.sum(np.sum(S[:T / step], axis=0)[self.sa], axis=1), axis=1) / np.sum(S[:T / step, -1])

    def get_policy(self, step, S, T=None):
        if T is None:
            return map(lambda a: random.choice(np.where(a == a.max())[0]),
                       np.sum(np.reshape(np.array(S.sum(axis=0)), -1)[self.sa], axis=2))
        elif isinstance(T, (int, long, float, complex)):
            return map(lambda a: random.choice(np.where(a == a.max())[0]),
                       np.sum(np.reshape(np.array((S[:T / step]).sum(axis=0)), -1)[self.sa], axis=2))
        else:
            return map(lambda a: random.choice(np.where(a == a.max())[0]),
                       np.sum(np.reshape(np.array((S[T[0] / step:T[1] / step]).sum(axis=0)), -1)[self.sa], axis=2))

    def get_policy2(self, step, S, T=None):
        if T is None:
            return map(lambda a: list(np.where(a == a.max())[0]),
                       np.sum(np.reshape(np.array(S.sum(axis=0)), -1)[self.sa], axis=2))
        elif isinstance(T, (int, long, float, complex)):
            return map(lambda a: list(np.where(a == a.max())[0]),
                       np.sum(np.reshape(np.array((S[:T / step]).sum(axis=0)), -1)[self.sa], axis=2))
        else:
            return map(lambda a: list(np.where(a == a.max())[0]),
                       np.sum(np.reshape(np.array((S[T[0] / step:T[1] / step]).sum(axis=0)), -1)[self.sa], axis=2))

    def R4Pi(self, policy, start_state, gamma=1, epsilon=0, max_iter=np.inf):
        """
        Calculate expected discounted reward for given policy and start state
        or distribution over start states
            policy: list of actions for each state
            returns  <R|s0>
        """
        Psa = np.zeros((self.num_states, self.num_actions), dtype=float)
        if isinstance(start_state, (int, long)):
            Psa[start_state, policy[start_state]] = 1
            r = np.dot(self.reward_cpt[start_state, policy[start_state]],
                       self.reward_sizes[start_state, policy[start_state]])
        else:  # start_state is distribution: array/list p(s_0)
            for i in xrange(self.num_states):
                Psa[i, policy[i]] = start_state[i]
            if len(np.shape(self.reward_cpt)) == 2:
                r = np.sum(Psa * self.reward_cpt * self.reward_sizes)
            else:
                r = np.dot(np.reshape(np.sum(self.reward_cpt * self.reward_sizes, axis=-1), -1),
                           np.reshape(Psa, -1))
        g = gamma
        counter = 0
        while np.sum(Psa) * g > epsilon and counter < max_iter:
            counter += 1
            oldPsa = np.copy(Psa)
            Psa = 0 * Psa
            tmp = np.dot(np.reshape(self.transition_cpt, (self.num_states * self.num_actions,
                                                          self.num_states)).T, np.reshape(oldPsa, -1))
            for i in xrange(self.num_states):
                Psa[i, policy[i]] = tmp[i]
            if len(np.shape(self.reward_cpt)) == 2:
                r += np.sum(Psa * self.reward_cpt * self.reward_sizes) * g
            else:
                r += np.dot(np.reshape(np.sum(self.reward_cpt * self.reward_sizes, axis=-1), -1),
                            np.reshape(Psa, -1)) * g
            g *= gamma
        return r

    def R4Pi2(self, policy, start_state, gamma=1, epsilon=0, max_iter=np.inf):
        """
        Calculate expected discounted reward for given policy and start state
        or distribution over start states
            policy: list of actions for each state
            returns  <R|s0>
        """
        Psa = np.zeros((self.num_states, self.num_actions), dtype=float)
        if isinstance(start_state, (int, long)):
            Psa[start_state, policy[start_state]] = 1. / len(policy[start_state])
            r = np.mean([np.dot(self.reward_cpt[start_state, p],
                                self.reward_sizes[start_state, p]) for p in policy[start_state]])
        else:  # start_state is distribution: array/list p(s_0)
            for i in xrange(self.num_states):
                Psa[i, policy[i]] = start_state[i] / len(policy[i])
            if len(np.shape(self.reward_cpt)) == 2:
                r = np.sum(Psa * self.reward_cpt * self.reward_sizes)
            else:
                r = np.dot(np.reshape(np.sum(self.reward_cpt * self.reward_sizes, axis=-1), -1),
                           np.reshape(Psa, -1))
        g = gamma
        counter = 0
        while np.sum(Psa) * g > epsilon and counter < max_iter:
            counter += 1
            oldPsa = np.copy(Psa)
            Psa = 0 * Psa
            tmp = np.dot(np.reshape(self.transition_cpt, (self.num_states * self.num_actions,
                                                          self.num_states)).T, np.reshape(oldPsa, -1))
            for i in xrange(self.num_states):
                Psa[i, policy[i]] = tmp[i] / len(policy[i])
            if len(np.shape(self.reward_cpt)) == 2:
                r += np.sum(Psa * self.reward_cpt * self.reward_sizes) * g
            else:
                r += np.dot(np.reshape(np.sum(self.reward_cpt * self.reward_sizes, axis=-1), -1),
                            np.reshape(Psa, -1)) * g
            g *= gamma
        return r

    def R4flatPi(self, start_state, gamma=1, epsilon=0, max_iter=np.inf):
        """
        Calculate expected discounted reward for flat policy and given start state
        or distribution over start states
            returns  expected discounted reward
        """
        Psa = np.zeros((self.num_states, self.num_actions), dtype=float)
        if isinstance(start_state, (int, long)):
            Psa[start_state] = 1. / self.num_actions
            r = np.sum(self.reward_sizes[start_state] *
                       self.reward_cpt[start_state]) / self.num_actions
        else:  # start_state is distribution: array/list p(s_0)
            Psa = np.tile(start_state, (self.num_actions, 1)).T / self.num_actions
            if len(np.shape(self.reward_cpt)) == 2:
                r = np.sum(Psa * self.reward_cpt * self.reward_sizes)
            else:
                r = np.dot(np.reshape(np.sum(self.reward_cpt * self.reward_sizes, axis=-1), -1),
                           np.reshape(Psa, -1))
        g = gamma
        counter = 0
        while np.sum(Psa) * g > epsilon and counter < max_iter:
            counter += 1
            tmp = np.dot(np.reshape(self.transition_cpt, (self.num_states * self.num_actions,
                                                          self.num_states)).T, np.reshape(Psa, -1)) / self.num_actions
            Psa = np.array([tmp] * self.num_actions).T
            if len(np.shape(self.reward_cpt)) == 2:
                r += np.sum(Psa * self.reward_cpt * self.reward_sizes) * g
            else:
                r += np.dot(np.reshape(np.sum(self.reward_cpt * self.reward_sizes, axis=-1), -1),
                            np.reshape(Psa, -1)) * g
            g *= gamma
        return r

    def next_SR(self, s, a):
        """
        Return next state and reward for given state s and action a
        In the case of termination the number of states is returned as 'next state'
        """
        ns = list(np.random.multinomial(1, np.append(self.transition_cpt[
                  s, a], 1 - np.sum(self.transition_cpt[s, a]))) == 1).index(1)
        if len(np.shape(self.reward_cpt)) == 2:
            r = self.reward_sizes[s, a] * (np.random.rand() < self.reward_cpt[s, a])
        else:
            r = np.dot(self.reward_sizes[s, a], np.random.multinomial(1, self.reward_cpt[s, a]))
        return ns, r

    def update_weights_nonepisodic4DKL(self, pTerminal, s, a, r, ns, eta, gamma=1):
        """
        Update weights given the sequence using delta rule update
            pTerminal: transition probabilty to terminal state
            s, a, r, ns: state, action, reward, next state
            eta: learning rate
            gamma: discount factor
        """
        for k in xrange(self.pop_size):
            self.W[self.sa[s, a, k]] *= (1 - eta)
            self.W[self.sa[s, a, k]] -= eta * self.competition[self.sa[s, a, k]]
            pTerminal[self.sa[s, a, k]] *= (1 - eta)
            if ns < self.num_states:
                self.W[self.sa[s, a, k], self.sa[ns]] += eta * gamma
            else:
                pTerminal[self.sa[s, a, k]] += eta * gamma
            self.W[self.sa[s, a, k], self.r] += eta * r
        return pTerminal

    def parallel_sampling_keepU(self, step, eta, run, rate, T=500, r0=.5, tm=20, ts=2,
                                reset=0, gamma=1, steps=100, start_state=0, mode='fix',
                                epsilon=0, max_iter=np.inf):
        np.random.seed(run)
        # maybe put this outside function to continue learning instead of fresh start
        self.unlearn(r0, mode)
        pTerminal = 1 - np.sum(self.W[:-self.pop_size, :-self.pop_size], axis=0)
        pi = np.zeros((steps, self.num_states), dtype=int)
        r = np.zeros(steps, dtype=float)
        DKL = np.zeros((steps, 2), dtype=float)
        RMSE = np.zeros((steps, 2), dtype=float)
        if isinstance(T, (int, long, float, complex)):
            Tmax = T
        else:
            Tmax = T[1]
        uinit = np.zeros(self.K)
        uinit[self.r] = rate / 1000.
        for t in xrange(steps):
            print 'run', run, '  trial', t, '/', steps
            stdout.flush()
            res = cfn.runpopU_js(self.W / self.pop_size, uinit, step, self.pop_size, rate,
                                 Tmax, tm, ts, 1. * reset / self.pop_size, run)
            uinit = res[1]
            pi[t] = self.get_policy(step, res[0], T)
            for s in xrange(self.num_states):
                for a in xrange(self.num_actions):
                    ns, rr = self.next_SR(s, a)
                    pTerminal = self.update_weights_nonepisodic4DKL(
                        pTerminal, s, a, rr, ns, eta, gamma)
            r[t] = self.R4Pi(pi[t], start_state, gamma, epsilon, max_iter)
            dd = self.DKL4weights(self.W, pTerminal, gamma)
            DKL[t] = np.array([np.mean(dd[0]), np.mean(dd[1])])
            RMSE[t] = np.sqrt(self.MSE4weights(self.W, pTerminal, gamma))
        return np.array([pi, r, DKL, RMSE, np.copy(self.W)])

    def probs4weights(self, W, gamma=1):
        if self.pop_size > 1:
            W2 = np.zeros((len(W) / self.pop_size, len(W) / self.pop_size))
            for i in xrange(len(W2)):
                for j in xrange(len(W2)):
                    W2[i, j] = np.mean(W[i * self.pop_size:(i + 1) * self.pop_size,
                                         j * self.pop_size:(j + 1) * self.pop_size])
        else:
            W2 = W
        t = np.copy(self.transition_cpt)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for ss in range(self.num_states):
                    t[s, a, ss] = W2[self.num_actions * s + a, self.num_actions * ss + a] / gamma
        tmp = W2[:-1, -1]
        r = np.reshape(tmp, np.shape(self.reward_cpt)[:2])
        return t, r

    def DKL4weights(self, W, pTerminal, gamma=1):
        (t, r) = self.probs4weights(W, gamma)
        term = np.mean(np.reshape(pTerminal, (self.pop_size, -1)), axis=0)
        transT = np.array([[np.append(self.transition_cpt[s, a], 1 - np.sum(self.transition_cpt[s, a]))
                            for a in range(self.num_actions)] for s in range(self.num_states)])
        # max in next line to avoid negative probs if sum of infered probs is (numerically)>1
        tT = np.array([[np.append(t[s, a], term[self.sa[s, a]])
                        for a in range(self.num_actions)] for s in range(self.num_states)])
        rf = np.reshape(r, -1)
        tf = np.reshape(tT, -1)
        if len(np.shape(self.reward_cpt)) == 2:
            rewf = np.reshape(self.reward_cpt * self.reward_sizes, -1)
        else:
            rewf = np.reshape(np.sum(self.reward_cpt * self.reward_sizes, axis=2), -1)
        transf = np.reshape(transT, -1)
        DKLr = np.copy(rewf)
        for i in xrange(len(DKLr)):
            if 0 < rewf[i] < 1 and (rf[i] == 0 or rf[i] == 1):
                DKLr[i] = 'inf'
            elif rewf[i] == 0:
                DKLr[i] = -np.log(1 - rf[i]) if rf[i] < 1 else 'inf'
            elif rewf[i] == 1:
                DKLr[i] = -np.log(rf[i]) if rf[i] > 0 else 'inf'
            else:
                DKLr[i] = rewf[i] * np.log(rewf[i] / rf[i]) + (1 - rewf[i]) * \
                    np.log((1 - rewf[i]) / (1 - rf[i]))
        DKLr = np.reshape(DKLr, (self.num_states, self.num_actions))
        DKLt = np.copy(transf)
        for i in xrange(len(DKLt)):
            if 0 < transf[i] and tf[i] == 0:
                DKLt[i] = 'inf'
            elif transf[i] == 0:
                DKLt[i] = 0
            else:
                DKLt[i] = transf[i] * np.log(transf[i] / tf[i])
        DKLt = np.sum(np.reshape(DKLt, (self.num_states, self.num_actions, -1)), axis=2)
        return np.array([DKLt, DKLr])

    def MSE4weights(self, W, pTerminal, gamma=1):
        (t, r) = self.probs4weights(W, gamma)
        term = np.mean(np.reshape(pTerminal, (self.pop_size, -1)), axis=0)
        transT = np.array([[np.append(self.transition_cpt[s, a], 1 - np.sum(self.transition_cpt[s, a]))
                            for a in range(self.num_actions)] for s in range(self.num_states)])
        # max in next line to avoid negative probs if sum of infered probs is (numerically)>1
        tT = np.array([[np.append(t[s, a], term[self.sa[s, a]])
                        for a in range(self.num_actions)] for s in range(self.num_states)])
        rf = np.reshape(r, -1)
        tf = np.reshape(tT, -1)
        if len(np.shape(self.reward_cpt)) == 2:
            rewf = np.reshape(self.reward_cpt * self.reward_sizes, -1)
        else:
            rewf = np.reshape(np.sum(self.reward_cpt * self.reward_sizes, axis=2), -1)
        transf = np.reshape(transT, -1)
        return np.array([np.mean((transf - tf)**2), np.mean((rewf - rf)**2)])
