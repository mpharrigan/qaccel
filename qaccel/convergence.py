"""Classes which determine the convergence of an MSM."""

import scipy.linalg
import scipy.stats
import numpy as np


class Multi:
    """Run multiple convergence criteria.

    behavior:
     - all : each needs to be converged
    """

    def __init__(self, *convs, behavior='all'):
        self.convs = convs
        self.behavior = behavior

    @property
    def is_done(self):
        if self.behavior == 'all':
            def _is_done(converged):
                return all(converged)
        else:
            raise ValueError("Invalid behavior")

        return _is_done

    def convergence(self, pipe, params):
        msm = pipe.recv()
        result = params.copy()
        converged = list()
        for c in self.convs:
            res = c.convergence(msm, params)
            converged += [res['converged']]
            result.update(res)

        result['converged'] = self.is_done(converged)
        return result


class Gmrq:
    def __init__(self, ref_msm, *, cutoff):
        self.ref_msm = ref_msm
        self.cutoff = cutoff

        self.S = np.diag(ref_msm.populations_)
        self.C = self.S.dot(ref_msm.transmat_)

    def convergence(self, msm, params):
        assert msm.n_timescales == self.ref_msm.n_timescales

        S = self.S
        C = self.C

        V = msm.right_eigenvectors_

        try:
            trace = np.trace(
                V.T.dot(C.dot(V)).dot(np.linalg.inv(V.T.dot(S.dot(V)))))
        except np.linalg.LinAlgError:
            trace = -1

        # Max is the sum of the true eigenvalues
        # NOTE: ref_msm must have an accurate n_timescales property
        err = self.ref_msm.score_ - trace
        converged = err < self.cutoff
        return {
            'converged': converged,
            'gmrq': err,
        }


class KL:
    def __init__(self, ref_msm, *, cutoff):
        self.ref_msm = ref_msm
        self.cutoff = cutoff

    def convergence(self, msm, params):
        kl = np.sum([scipy.stats.entropy(prow, qrow, base=2)
                     for prow, qrow in
                     zip(msm.transmat_, self.ref_msm.transmat_)
                     ])
        return {
            'converged': kl < self.cutoff,
            'kl': kl,
        }
