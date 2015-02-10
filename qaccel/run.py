"""Do a run of simulation

Author: Matthew Harrigan
"""

from itertools import zip_longest

import numpy as np
import pandas as pd


def get_map(*, param_gen, ref_msm, simulator, adapt, builder, convergence,
            initial_func):
    """Get function and arguments to pass to `map`

    :param param_gen: An iterator that yields Params
    :param ref_msm: Reference MSM
    :param simulator: Object responsible for running simulation
    :param adapt: Object responsible for generating new starting states
    :param builder: Object responsible for building models
    :param convergence: Object responsible for checking for convergence
    :param initial_func: Function for generating initial starting states
    """

    def map_func(param):
        r = Run(simulator, builder, convergence, adapt, initial_func, param)
        return r.run()

    return map_func, param_gen


class Run:
    def __init__(self, simulator, builder, conv, adapter, initial_func, param):
        self.simulator = simulator
        self.builder = builder
        self.conv = conv
        self.adapter = adapter
        self.param = param
        self.initial_func = initial_func


    def run(self):
        results = []
        steps_left = self.param.post_converge // self.param.res
        running_counts = np.zeros((self.conv.true_n, self.conv.true_n))
        sstate = self.initial_func()

        while True:
            # States will be a list of trajectories, where each trajectory
            # is a 1d sequence of state labels.
            states = self.simulator.sample_states(self.param, sstate=sstate)

            # Get first states of each trajectory
            step_it = enumerate(zip_longest(*states))
            i, prev_step = next(step_it)

            for i, step in step_it:
                # step will be tuple of length tpr

                # aggregate counts
                for pr, ne in zip(prev_step, step):
                    if pr is not None and ne is not None:
                        running_counts[pr, ne] += 1

                if i % self.param.res == 0:
                    # make msm
                    msm = self.builder.build_msm(running_counts)

                    # calculate fro
                    err, converged = self.conv.check_conv(msm)
                    results += [dict(
                        step_i=i, err=err, converged=converged
                    )]

                    if converged:
                        steps_left -= 1

                    # Exit condition
                    if steps_left < 1:
                        break

                prev_step = step

            # Set starting states
            sstate = self.adapter.adapt(running_counts)

        # Return a data frame
        res_df = pd.DataFrame(results, index='step_i')
        return res_df


