import numpy as np

class Random:

    def __init__(self, ref_msm):
        self.ref_msm = ref_msm
        self.true_n = ref_msm.n_states_

    def adapt(self, param, counts):
        return np.random.randint(self.true_n, size=param.tpr)


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
