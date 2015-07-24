_ERRMSG = "Please provide n_tpr starting states. You gave {}"

from .dag import Deref


class Simulator:
    def sample_states(self, param, sstate):
        """Run some simulation and return new state labels

        :param param: Adaptive parameters
        :param sstate: The starting states. This should probably
                       be list-like with size = param.tpr
        """
        raise NotImplementedError


class TMatSimulator(Simulator):
    def __init__(self, msm, parallel=True):
        self.msm = msm
        self.dref = Deref(parallel)

    def sample_from_sample(self, traj, params):
        traj = self.dref.get(traj)
        ret =  self.msm.sample_discrete(state=traj[-1], n_steps=params['res']+1)
        return ret[1:]

    def sample_from_adapt(self, states, i, params):
        states = self.dref.get(states)
        return self.msm.sample_discrete(state=states[i], n_steps=params['res'])


class OpenMMSimulator(Simulator):
    def __init__(self, system, integrator):
        pass
