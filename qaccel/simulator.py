_ERRMSG = "Please provide n_tpr starting states. You gave {}"


class TMatSimulator:
    def __init__(self, msm):
        self.msm = msm

    def simulate(self, last, params):
        return self.msm.sample_discrete(state=last, n_steps=params['res'])

    def init(self, params):
        return [
            [[0]] for _ in range(params['tpr'])
            ]
