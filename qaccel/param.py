class Param:
    """Values which (may) change per run

    :param spt: Steps per traj
    :param tpr: Trajectories per round
    :param res: Resolution at which we build / check for convergence
    :param post_converge: Number of steps to do after convergence
        Actually, each time we check for convergence (with frequency res)
        and convergence is True, we decrement the number of steps left
        by res. This balances between
            - Randomly fluctuating into convergence too early and stopping
              before we get to and stay in convergence
            - Randomly fluctuating out of convergence on what would have
              been the last round and continuing simulation in the converged
              regime.
    """
    def __init__(self, *, spt, tpr, res, post_converge):
        self.spt = spt
        self.tpr = tpr
        self.res = res
        self.post_converge = post_converge