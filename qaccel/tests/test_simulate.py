from unittest import TestCase

from qaccel.simulator import TMatSimulator
import numpy as np


class TestTMatSimulator(TestCase):
    def setUp(self):
        from msmbuilder.msm import MarkovStateModel
        msm = MarkovStateModel(verbose=False)
        msm.fit([[0, 0, 1, 1, 0]])

        self.simulator = TMatSimulator(msm, parallel=False)
        self.params = {
            'res': 30,
        }

    def test_init(self):
        traj = self.simulator.init(5)
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj[0], 5)

    def test_simulate_from_simulate(self):
        traj = self.simulator.simulate_from_simulate([0, 1], self.params)
        self.assertEqual(len(traj), self.params['res'])
        self.assertLess(np.abs(np.mean(traj) - 0.5), 0.2)

    def test_simulate_from_adapt(self):
        traj = self.simulator.simulate_from_adapt([0, 1, 1, 0], 1, self.params)
        self.assertEqual(traj[0], 1)
        self.assertEqual(len(traj), self.params['res'])
