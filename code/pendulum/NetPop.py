import numpy as np
from sys import stdout, path
path.append('../')
from DPNetPop import DPNetPop
import cfunctions as cfn
from scipy.integrate import dblquad, quad
import integrate2d as cf


class NetPop(DPNetPop):
    """
    Network with hard coded transition and reward probabilities
    Network has loops => arbitrary long episodes possible
    perfromed rescaling and shifting:
    p <- 8/pi*p        org p in [0,2pi]
    v <- 8/pi*v + 16   org v in [-2*pi,2*pi]
    position neurons: 0,...,15
    velocity neurons: 0,...,32
    16 (position) x 33 (velocity) x 3 (action) neurons

    """

    def __init__(self, pop_size=1, steps=3, set_W=True):
        # reward_cpt(s,a,r) = p(r|s,a)
        self.num_p = num_p = 16
        self.num_v = num_v = 33
        num_s = num_p * num_v
        reward_cpt = np.ones((num_s, 3))
        self.steps = steps
        try:
            tmp = np.load('cpt.npz')
            transition_cpt = tmp['transition_cpt']
            reward_sizes = tmp['reward_sizes']
            print 'CPTs loaded'
            stdout.flush()
        except:
            print 'Creating CPTs'
            stdout.flush()
            # transition_cpt(s,a,s') = p(s'|a,s)
            # 0 = reverse (-1), 1 = no throttle (0), 2 = forward (1)
            transition_cpt = np.ones((num_s, 3, num_s), dtype=float) / 3.
            reward_sizes = np.zeros((num_s, 3))

            def pos(s):
                return self.pos(s)

            def vel(s):
                return self.vel(s)
            overlap = np.array([quad(lambda p: self.phi(p) * self.phi(j - p), -
                                     np.inf, np.inf, epsabs=1e-20) for j in range(max(num_p, num_v))])
            overlap[overlap[:, 0] < overlap[:, 1], 0] = 0
            overlap = overlap[:, 0]
            Kp = np.array([[overlap[min(abs(i - j), 16 - abs(i - j))]
                            for i in range(num_p)] for j in range(num_p)])
            Kv = np.array([[overlap[abs(i - j)]
                            for i in range(num_v)] for j in range(num_v)])
            self.invKp = invKp = np.linalg.inv(Kp)
            self.invKv = invKv = np.linalg.inv(Kv)

            def phi(i):
                sigma = .5
                return 0 if abs(i) > 5 else np.exp(-i**2 / (2 * sigma**2)) / (2 * np.pi)**(1. / 2) / sigma
            for i in xrange(num_s):
                for a in xrange(3):
                    integral = [cf.trapz2d(i, n, a, steps=steps) for n in xrange(num_s)]
                    for k in xrange(num_s):
                        transition_cpt[i, a, k] = np.sum([invKp[pos(k), pos(n)] * invKv[vel(k), vel(n)] *
                                                          integral[n] for n in xrange(num_s)])

                reward_sizes[i] = dblquad(lambda v, p: phi(pos(i) - p) * phi(vel(i) - v) * cf.get_R(p, v),
                                          pos(i) - 5, pos(i) + 5, lambda tmp: vel(i) - 5,
                                          lambda tmp: vel(i) + 5, epsabs=1e-6)[0]
            np.savez_compressed('cpt.npz', reward_sizes=reward_sizes,
                                transition_cpt=transition_cpt)
        DPNetPop.__init__(self, transition_cpt, reward_cpt, pop_size, reward_sizes, set_W)

    def pos(self, state_nr):
        return int(state_nr / self.num_v)

    def vel(self, state_nr):
        return state_nr % self.num_v

    def get_Q(self, x, p, v):
        return np.dot(np.mean((x * (x > 0))[self.sa], axis=2).T, np.array([cf.phi((p - i / self.num_v + 8) % 16 - 8) *
                                                                           cf.phi(v - i % self.num_v) for i in xrange(self.num_states)]))

    def get_a(self, x, p, v):
        Q = self.get_Q(x, p, v)
        return np.where(Q == Q.max())[0][-1]

    def update_weights_continuous(self, s, a, r, ns, eta, gamma=1):
        """
        Update weights given state, action, reward and next state  using delta rule update
            s, a, r, ns: state, action, reward, next state
            eta: learning rate
            gamma: discount factor
        """
        # (neuronindex/popsize)/num_actions = stateindex
        # (neuronindex/popsize)%num_actions = actionindex
        # stateindex/17 = positionindex
        # stateindex%17 = velocityindex
        p, v = s
        pp, vv = ns
        Swphi = np.dot((self.W + self.competition).T[:, :-1],
                       np.array([0 if (a != i / self.pop_size % 3) else
                                 cf.phi((p - i / self.pop_size / (3 * self.num_v) + 8) % 16 - 8) *
                                 cf.phi(v - (i / self.pop_size / 3) % self.num_v)
                                 for i in xrange(self.K - 1)]))
        Intphia = [cf.phi((pp - i / self.pop_size / (3 * self.num_v) + 8) % 16 - 8) *
                   cf.phi(vv - (i / self.pop_size / 3) % self.num_v) * gamma
                   for i in xrange(self.K - 1)]
        Intphia += [r]
        factor = eta * (np.array(Intphia) - Swphi)
        for i in xrange(self.num_states):
            # skip update for neurons 'far away'
            if abs(i / self.num_v - s[0]) > 2 or abs(i % self.num_v - s[1]) > 2:
                continue
            for k in xrange(self.pop_size):
                self.W[self.sa[i, a, k]] += factor *\
                    cf.phi((p - i / self.num_v + 8) % 16 - 8) *\
                    cf.phi(v - i % self.num_v)

    def parallel_sampling_keepU(self, step, eta, run, rate, T=500, r0=.5, tm=20, ts=2,
                                reset=0, gamma=1, trials=1000, mode='fix',  maxsteps=300,
                                initpv=None, initW=None, samples=100):
        np.random.seed(run)
        if initW is None:
            self.unlearn(r0, mode)
        else:
            self.W = initW
        if isinstance(T, (int, long, float, complex)):
            Tmax = T
        else:
            Tmax = T[1]
        seq = [[None]] * trials  # sequences might have differnt length -> list instead array
        scount = np.zeros((trials, self.K))
        uinit = np.zeros(self.K)
        uinit[self.r] = rate / 1000.
        for t in xrange(trials):
            print 'run', run, '  trial', t, '/', trials
            stdout.flush()
            pvls = [[8, 16]] if initpv is None else [initpv]
            a = []
            r = []
            res = cfn.runpopU_js(self.W / self.pop_size, uinit, step, self.pop_size, rate,
                                 Tmax, tm, ts, 1. * reset / self.pop_size, run)
            uinit = res[1]
            scount[t] = np.sum(res[0], axis=0)
            for counter in xrange(maxsteps):
                a += [self.get_a(scount[t], *pvls[-1])]
                pvls += [cf.get_next_pv(pvls[-1][0], pvls[-1][1], a[-1])]
                r += [cf.get_R(*pvls[-1])]
            seq[t] = [pvls[:-1], a, r]
            for i in range(samples):
                p = 16 * np.random.rand()
                v = 32 * np.random.rand()
                for a in range(3):
                    pvs = cf.get_next_pvs(p, v, a, self.steps)
                    rr = cf.get_R(*pvs)
                    self.update_weights_continuous([p, v], a, rr, pvs,
                                                   eta, gamma**self.steps)

        return np.array([scount, np.array(seq), np.copy(self.W)])
