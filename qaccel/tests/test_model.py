from unittest import TestCase
from qaccel.model import MSMFromLabtraj
import numpy as np


class TestMSMFromLabtraj(TestCase):
    def setUp(self):
        self.modeler = MSMFromLabtraj(parallel=False)
        self.params = {
            'n_states': 2,
            'lag_time': 1,
            'prior_counts': 1e-5,
        }

    def test_model(self):
        chunked_trajs = [
            [[0, 0, 1, 1, 0], [0, 1, 1, 0]]
        ]
        msm = self.modeler.model(chunked_trajs, self.params)
        tmat_should_be = np.asarray([
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        counts_should_be = np.asarray([
            [2, 2],
            [2, 2],
        ])

        np.testing.assert_array_equal(msm.transmat_, tmat_should_be)
        np.testing.assert_array_equal(msm.countsmat_, counts_should_be)
