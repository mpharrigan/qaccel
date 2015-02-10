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
        trajs = []
        assert len(sstate) == param.tpr, _ERRMSG.format(len(sstate))
        for ss in sstate:
            trajs += [
                self.msm.sample(state=ss, n_steps=param.spt)
            ]
        return trajs


class OpenMMSimulator(Simulator):
    def __init__(self, system, integrator):
        pass