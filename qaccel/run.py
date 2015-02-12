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

    @classmethod
    def _set_up_logger(cls, param):
        parm_str = param.unique_string()
        log_fn = "run-{}.log".format(parm_str)
        log = logging.getLogger(parm_str)
        formatter = logging.Formatter(style='{')
        filehandler = logging.FileHandler(log_fn, mode='w')
        filehandler.setFormatter(formatter)
        log.addHandler(filehandler)
        return log

    def safe_run(self, param, out_fn_fmt="results-{param_str}.h5"):
        """Run the adaptive loop where a problem won't destroy everything

        Catch exceptions, save results here.
        """
        log = logging.getLogger(param.unique_string())
        try:
            res_df, res_param = self.run(param)
            out_fn = out_fn_fmt.format(param_str=param.unique_string())
            if out_fn.endswith(".h5"):
                res_df.to_hdf(out_fn, key='results')
            else:
                log.warn("Didn't save an individual run output.")
                log.warn("Unknown file format: %s", out_fn)

            return res_df, res_param

        except:
            log.exception("A problem occurred")

    def run(self, param):
        """Run the adaptive loop

        :param param: Values which change for each run.
        """
        log = self._set_up_logger(param)
        log.info("Starting simulation %s", param.unique_string())
        results = []
        steps_left = param.post_converge // param.res
        running_counts = np.zeros((self.conv.true_n, self.conv.true_n))
        sstate = self.initial_func(self, param)
        round_i = 0

        while True:
            # States will be a list of trajectories, where each trajectory
            # is a 1d sequence of state labels.
            states = self.simulator.sample_states(param, sstate)

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
                        round_i=round_i, step_i=i, err=err, converged=converged
                    )]

                    if converged:
                        steps_left -= 1

                    log.info(
                        "Round %4d Step %10d Error: %10g Converged %s",
                        round_i, i, err, converged
                    )

                prev_step = step

            # Exit condition
            if steps_left < 1:
                break

            # Set starting states
            sstate = self.adapter.adapt(param, running_counts)
            round_i += 1

        # Return a data frame
        res_df = pd.DataFrame(results)
        log.info("Done")
        return res_df, param


