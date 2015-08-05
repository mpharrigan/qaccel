import numpy as np
from .dag import Deref


class Random:
    def __init__(self, n_states):
        self.n_states = n_states

    def adapt(self, model, params):
        return np.random.randint(self.n_states, size=params['tpr'])


class TruePopWeightedSample:
    def __init__(self, ref_msm, zeta=1.0):
        self.ref_msm = ref_msm
        self.zeta = zeta

        self.probs = 1.0 / (ref_msm.populations_ ** zeta)
        self.cumprobs = np.cumsum(self.probs)

    def adapt(self, param, counts):
        new_states = []
        for _ in range(param.tpr):
            rr = np.random.uniform(self.cumprobs[-1])
            new_states += [np.argmax(rr < self.cumprobs)]
        return np.asarray(new_states)
