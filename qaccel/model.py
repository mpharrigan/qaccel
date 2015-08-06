from .deref import Deref
from msmbuilder.msm import MarkovStateModel
import numpy as np
from .count import make_counts


class MSMFromLabtraj:
    def __init__(self, parallel=True):
        self.dref = Deref(parallel)

    def model(self, trajs, params):
        counts = make_counts(trajs, params['n_states'], self.dref)

        msm = MarkovStateModel(lag_time=params['lag_time'],
                               prior_counts=params['prior_counts'],
                               n_timescales=params['n_timescales'],
                               )
        msm.countsmat_ = counts
        try:
            msm.transmat_, msm.populations_ = msm._fit_mle(counts)
        except ValueError:
            msm.transmat_, msm.populations_ = msm._fit_mle(np.ones_like(counts))
        msm.mapping_ = dict((i, i) for i in range(len(counts)))
        msm._is_dirty = True
        return msm
