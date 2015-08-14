from unittest import TestCase

from qaccel.runs.ala_rnd import Run
import pandas as pd


class TestAlaRnd(TestCase):
    def test_round(self):
        rungen = Run()
        run = rungen.make_run({'tpr': 2, 'rep_spt': 5})

        self.assertEqual(run.params['param_str'], "rep_spt-5_tpr-2")

        results = run.main_loop()

        df = pd.DataFrame(results)
        self.assertGreaterEqual(df['converged'].sum(), 1)
