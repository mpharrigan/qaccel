"""Do a run of simulation

Author: Matthew Harrigan
"""

from itertools import zip_longest

import numpy as np
import pandas as pd
import logging


class Run:
    """A single adaptive run until convergence

    :param simulator: Object responsible for running simulation
    :param adapter: Object responsible for generating new starting states
    :param builder: Object responsible for building models
    :param convergence: Object responsible for checking for convergence
    :param initial_func: Function for generating initial starting states. It
                         will be passed the run object and a param object.
    """

    def __init__(self, *, simulator, builder, convergence, adapter,
                 initial_func):
        self.simulator = simulator
        self.builder = builder
        self.conv = convergence
        self.adapter = adapter
        self.initial_func = initial_func

    def run(self, param):
        """Run the adaptive loop

        :param param: Values which change for each run.
        """
        log = logging.getLogger(param.unique_string())
        log.addHandler(logging.FileHandler(param.unique_string()))
        results = []
        steps_left = param.post_converge // param.res
        running_counts = np.zeros((self.conv.true_n, self.conv.true_n))
        sstate = self.initial_func(param)

        log.debug("Logging debug")
        log.info("Logging info")
        log.warning("Logging warning")
        log.error("Logging error")

        while True:
            # States will be a list of trajectories, where each trajectory
            # is a 1d sequence of state labels.
            states = self.simulator.sample_states(param, sstate=sstate)

            # Get first states of each trajectory
            step_it = enumerate(zip_longest(*states))
            i, prev_step = next(step_it)

            for i, step in step_it:
                # step will be tuple of length tpr

                # aggregate counts
                for pr, ne in zip(prev_step, step):
                    if pr is not None and ne is not None:
                        running_counts[pr, ne] += 1

                if i % param.res == 0:
                    # make msm
                    msm = self.builder.build(running_counts)

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


