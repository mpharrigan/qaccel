from qaccel.convergence import Gmrq, KL
from qaccel.simulator import TMatSimulator
from qaccel.model import MSMFromLabtraj
from unittest import TestCase


class TestGmrq(TestCase):
    def setUp(self):
        from qaccel.reference.alanine import get_ref_msm
        self.msm = get_ref_msm()

        self.simulator = TMatSimulator(self.msm, parallel=False)
        self.modeler = MSMFromLabtraj(parallel=False)
        self.params = {
            'res': 10,
            'n_states': self.msm.n_states_,
            'lag_time': 1,
            'prior_counts': 1e-5,
            'n_timescales': self.msm.n_timescales,
        }

    def test_gmrq(self):
        convergence = Gmrq(self.msm, cutoff=0.1, parallel=False)
        traj1 = self.simulator.simulate_from_adapt([0], 0, self.params)
        model = self.modeler.model([[traj1]], self.params)
        conv1 = convergence.convergence(model, self.params)

        self.assertGreater(conv1['gmrq'], 0.1)
        self.params.update(res=1000)

        traj2 = self.simulator.simulate_from_simulate(traj1, self.params)
        model = self.modeler.model([[traj1, traj2]], self.params)

        conv2 = convergence.convergence(model, self.params)
        self.assertLess(conv2['gmrq'], conv1['gmrq'])
        self.assertTrue(conv2['converged'])
        self.assertLess(conv2['gmrq'], 0.1)

    def test_kl(self):
        convergence = KL(self.msm, cutoff=5, parallel=False)
        traj1 = self.simulator.simulate_from_adapt([0], 0, self.params)
        model = self.modeler.model([[traj1]], self.params)
        conv1 = convergence.convergence(model, self.params)

        self.assertGreater(conv1['kl'], 0.1)
        self.params.update(res=1000)

        traj2 = self.simulator.simulate_from_simulate(traj1, self.params)
        model = self.modeler.model([[traj1, traj2]], self.params)

        conv2 = convergence.convergence(model, self.params)
        self.assertLess(conv2['kl'], conv1['kl'])
        self.assertLess(conv2['kl'], 5)
        self.assertTrue(conv2['converged'])

