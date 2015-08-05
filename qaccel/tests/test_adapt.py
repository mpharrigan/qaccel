from unittest import TestCase
from qaccel.adapt import Random
import numpy as np


class TestRandom(TestCase):
    def setUp(self):
        self.adapt = Random(n_states=2)
        self.params = {
            'n_states': 2,
            'tpr': 10,
        }

    def test_adapt(self):
        states = self.adapt.adapt(None, self.params)
        self.assertEqual(len(states), 10)
        for s in states:
            self.assertLess(s, 2)

        self.assertLess(np.abs(np.mean(states) - 0.5), 0.4)
