_ERRMSG = "Please provide n_tpr starting states. You gave {}"


class Simulator:
    def sample_states(self, param, sstate):
        """Run some simulation and return new state labels

        :param param: Adaptive parameters
        :param sstate: The starting states. This should probably
                       be list-like with size = param.tpr
        """
        raise NotImplementedError


class TMatSimulator(Simulator):
    def __init__(self, msm):
        self.msm = msm

    def sample_states(self, param, sstate):
        """Sample from MSM

        The actual length of trajectories will be spt + 1 to give
        `spt` steps. The starting state is included.
        """
        assert len(sstate) == param.tpr, _ERRMSG.format(len(sstate))
        trajs = [self.msm.sample_discrete(state=ss, n_steps=param.spt + 1)
                 for ss in sstate]
        return trajs


class OpenMMSimulator(Simulator):
    def __init__(self, system, integrator):
        pass