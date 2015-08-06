_ERRMSG = "Please provide n_tpr starting states. You gave {}"

from .deref import Deref


class TMatSimulator:
    def __init__(self, msm, parallel=True):
        self.msm = msm
        self.dref = Deref(parallel)

    def simulate_from_simulate(self, traj, params):
        traj = self.dref(traj)
        ret = self.msm.sample_discrete(state=traj[-1],
                                       n_steps=params['res'] + 1)
        return ret[1:]

    def simulate_from_adapt(self, states, i, params):
        states = self.dref(states)
        return self.msm.sample_discrete(state=states[i], n_steps=params['res'])

    def init(self, i):
        return [i]
