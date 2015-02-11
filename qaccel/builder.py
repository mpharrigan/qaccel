import numpy as np

from msmbuilder.msm import MarkovStateModel


class MSMBuilder:
    def build(self, counts):
        """Add a pseudo-count and don't do trimming."""

        counts = np.copy(counts)
        counts += 1
        msm = MarkovStateModel()
        msm.countsmat_ = counts
        msm.transmat_, msm.populations_ = msm._fit_mle(counts)
        msm._is_dirty = True

        return msm

