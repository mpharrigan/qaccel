from unittest import TestCase
from multiprocessing import Pool, Pipe

from qaccel.convergence import Gmrq, KL, Multi
from qaccel.simulator import TMatSimulator
from qaccel.model import MSMFromLabtraj
from qaccel.multip import _call_convergence, _call_model


class TestConvergence(TestCase):
    def setUp(self):
        from qaccel.reference.alanine import get_ref_msm
        self.msm = get_ref_msm()

        self.simulator = TMatSimulator(self.msm)
        self.modeler = MSMFromLabtraj()
        self.params = {
            'res': 10,
            'n_states': self.msm.n_states_,
            'lag_time': 1,
            'prior_counts': 1e-5,
            'n_timescales': self.msm.n_timescales,
        }

    def test_multi(self):
        convergence = Multi(
            KL(self.msm, cutoff=7),
            Gmrq(self.msm, cutoff=0.1),
            behavior='all'
        )
        traj1 = self.simulator.simulate(0, self.params)

        with Pool() as pool:
            conv_pipe, model_pipe = Pipe(duplex=False)
            model_fut = pool.apply_async(
                _call_model,
                (self.modeler, [[traj1]], model_pipe, self.params)
            )
            conv_fut = pool.apply_async(
                _call_convergence,
                (convergence, conv_pipe, self.params)
            )
            conv1 = conv_fut.get()

            self.assertGreater(conv1['kl'], 7)
            self.assertGreater(conv1['gmrq'], 0.1)
            self.params.update(res=1000)

            traj2 = self.simulator.simulate(traj1[-1], self.params)
            conv_pipe, model_pipe = Pipe(duplex=False)
            pool.apply_async(
                _call_model,
                (self.modeler, [[traj1, traj2]], model_pipe, self.params)
            )
            conv_fut = pool.apply_async(
                _call_convergence,
                (convergence, conv_pipe, self.params)
            )
            conv2 = conv_fut.get()

        self.assertLess(conv2['kl'], conv1['kl'])
        self.assertLess(conv2['gmrq'], conv1['gmrq'])
        self.assertLess(conv2['kl'], 7)
        self.assertLess(conv2['gmrq'], 0.1)
        self.assertTrue(conv2['converged'])

    def test_kl(self):
        convergence = KL(self.msm, cutoff=7)
        traj1 = self.simulator.simulate(0, self.params)
        model = self.modeler.model([[traj1]], None, self.params)
        conv1 = convergence.convergence(model, self.params)

        self.assertGreater(conv1['kl'], 7)
        self.params.update(res=1000)

        traj2 = self.simulator.simulate(traj1[-1], self.params)
        model = self.modeler.model([[traj1, traj2]], None, self.params)

        conv2 = convergence.convergence(model, self.params)
        self.assertLess(conv2['kl'], conv1['kl'])
        self.assertLess(conv2['kl'], 7)
        self.assertTrue(conv2['converged'])
