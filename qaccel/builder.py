import numpy as np

from msmbuilder.msm import MarkovStateModel


class MSMBuilder:
    def __init__(self, msm_kwargs=None, pseudo_count=1):
        if msm_kwargs is None:
            msm_kwargs = {}
        self.msm_kwargs = msm_kwargs
        self.pseudo_count = pseudo_count

    def build(self, counts):
        """Add a pseudo-count and don't do trimming."""

        # Add pseudo count
        counts = np.copy(counts)
        counts += self.pseudo_count

        # Do msm.fit() by hand.
        msm = MarkovStateModel(**self.msm_kwargs)
        msm.countsmat_ = counts
        msm.transmat_, msm.populations_ = msm._fit_mle(counts)
        msm._is_dirty = True

        return msm

