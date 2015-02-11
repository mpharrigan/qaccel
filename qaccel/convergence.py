class Frobenius:
    def __init__(self, ref_msm):
        self.ref_msm = ref_msm

    @property
    def true_n(self):
        return self.ref_msm.n_states_
