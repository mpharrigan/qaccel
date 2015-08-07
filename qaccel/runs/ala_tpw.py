from ..simulator import TMatSimulator
from ..model import MSMFromLabtraj
from ..convergence import Gmrq, KL, Multi
from ..adapt import TruePopWeighted

from ..reference.alanine import get_ref_msm

from ..dag import DAG


class Run:
    def __init__(self, lbv):
        self.lbv = lbv
        self.ref = get_ref_msm()

        pass

    def make_dag(self, params):
        def_params = {
            'tpr': 1,
            'res': 10,
            'res_spt': 1,
            'n_states': self.ref.n_states_,
            'lag_time': 1,
            'prior_counts': 1e-5,
            'n_timescales': self.ref.n_timescales,
            'clone': 1,
            'gmrq_cutoff': 0.05,
            'kl_cutoff': 5.0,
            'conv_behavior': 'all',
        }
        def_params.update(params)
        dag = DAG(self.lbv,
                  simulator=TMatSimulator(self.ref),
                  modeler=MSMFromLabtraj(),
                  convergence=Multi(
                      Gmrq(self.ref, cutoff=def_params['gmrq_cutoff']),
                      KL(self.ref, cutoff=def_params['kl_cutoff']),
                      behavior=def_params['conv_behavior']
                  ),
                  adapter=TruePopWeighted(self.ref),
                  params=def_params,
                  )
        return dag
