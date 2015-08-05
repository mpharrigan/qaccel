from qaccel.convergence import Gmrq
from qaccel.simulator import TMatSimulator
from qaccel.model import MSMFromLabtraj
from unittest import TestCase


class TestGmrq(TestCase):
    def setUp(self):
        from qaccel.reference.alanine import get_ref_msm
        msm = get_ref_msm()

        self.convergence = Gmrq(msm, cutoff=0.1, parallel=False)
        self.simulator = TMatSimulator(msm, parallel=False)
        self.modeler = MSMFromLabtraj(parallel=False)
        self.params = {
            'res': 10,
            'n_states': msm.n_states_,
            'lag_time': 1,
            'prior_counts': 1e-5,
        }

    def test_gmrq(self):
        traj1 = self.simulator.simulate_from_adapt([0], 0, self.params)
        model = self.modeler.model([[traj1]], self.params)
        converged1, err1 = self.convergence.convergence(model, self.params)

        self.assertGreater(err1, 0.1)
        self.params.update(res=1000)

        traj2 = self.simulator.simulate_from_simulate(traj1, self.params)
        model = self.modeler.model([[traj1, traj2]], self.params)

        converged2, err2 = self.convergence.convergence(model, self.params)
        self.assertLess(err2, err1)
        self.assertTrue(converged2)
        self.assertLess(err2, 0.1)
