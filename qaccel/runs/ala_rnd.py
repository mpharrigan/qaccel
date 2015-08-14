from ..simulator import TMatSimulator
from ..model import MSMFromLabtraj
from ..convergence import Gmrq, KL, Multi
from ..adapt import Random

from ..reference.alanine import get_ref_msm

from ..multip import Files


class Run:
    def __init__(self):
        self.ref = get_ref_msm()
        self.param_str = ""

    def make_run(self, params):
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
        params['param_str'] = "_".join(
            "{}-{}".format(k, params[k]) for k in sorted(params))
        def_params.update(params)
        dag = Files(simulator=TMatSimulator(self.ref),
                    modeler=MSMFromLabtraj(),
                    convergence=Multi(
                        Gmrq(self.ref, cutoff=def_params['gmrq_cutoff']),
                        KL(self.ref, cutoff=def_params['kl_cutoff']),
                        behavior=def_params['conv_behavior']
                    ),
                    adapter=Random(),
                    params=def_params,
                    )
        return dag
