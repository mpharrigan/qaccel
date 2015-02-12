import numpy as np

class Random:

    def __init__(self, ref_msm):
        self.ref_msm = ref_msm
        self.true_n = ref_msm.n_states_

    def adapt(self, param, counts):
        return np.random.randint(self.true_n, size=param.tpr)
