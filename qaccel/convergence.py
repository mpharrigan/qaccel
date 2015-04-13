"""Classes which determine the convergence of an MSM."""

import scipy.linalg
import numpy as np


class Frobenius:
    """Do the frobenius norm between transition matrix and reference

    :param ref_msm: Reference msm to compare transition matrix.
    :param cutoff: The error cutoff, below which we are 'converged'
    """

    def __init__(self, ref_msm, *, cutoff):
        self.ref_msm = ref_msm
        self.cutoff = cutoff

    def check_conv(self, msm):
        """Check for convergence

        :param msm: The test MSM
        :returns err: Some numerical error value
        :returns converged: Whether error is below a threshold
        """
        diff_mat = self.ref_msm.transmat_ - msm.transmat_
        err = scipy.linalg.norm(diff_mat, ord='fro')
        converged = err < self.cutoff

        return err, converged


    @property
    def true_n(self):
        return self.ref_msm.n_states_

class Gmrq:
    def __init__(self, ref_msm, *, cutoff):
        self.ref_msm = ref_msm
        self.cutoff = cutoff # minimum gmrq

        self.S = np.diag(ref_msm.populations_)
        self.C = self.S.dot(ref_msm.transmat_)

    def check_conv(self, msm):
        S = self.S
        C = self.C

        V = msm.right_eigenvectors_
        if msm.mapping_ != self.ref_msm.mapping_:
            V = msm._map_eigenvectors(V, self.ref_msm.mapping_)

        try:
            trace = np.trace(V.T.dot(C.dot(V)).dot(np.linalg.inv(V.T.dot(S.dot(V)))))
        except np.linalg.LinAlgError:
            trace = -1

        converged = trace > self.cutoff
        return trace, converged

    @property
    def true_n(self):
        return self.ref_msm.n_states_

