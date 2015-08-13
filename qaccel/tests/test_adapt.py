from unittest import TestCase
from qaccel.adapt import Random, TruePopWeighted
import numpy as np


class TestAdapt(TestCase):
    def setUp(self):
        self.params = {
            'n_states': 2,
            'tpr': 10,
        }

    def test_random(self):
        adapt = Random()
        states = adapt.adapt(None, self.params)
        self.assertEqual(len(states), 10)
        for s in states:
            self.assertLess(s, 2)

        self.assertLess(np.abs(np.mean(states) - 0.5), 0.4)

    def test_truepopweight(self):
        # TODO
        pass

