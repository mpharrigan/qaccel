from multiprocessing import Pool, Pipe
from itertools import repeat as rpt


# [
#   [ [chunk], [chunk], [chunk] ],
#   [ [chunk], [chunk], [chunk] ]
# ]

def update(sims, iterresults):
    for simchunks, result in zip(sims, iterresults):
        simchunks += [result]


def update_new(sims, newstates):
    sims += [
        [[ns]] for ns in newstates
        ]
    return sims


def last_of(sims):
    for simchunks in sims:
        yield simchunks[-1][-1]


def filter_futs(futs):
    new_futs = set()
    for fut in futs:
        if not fut.ready():
            new_futs.add(fut)

    return new_futs


def _call_simulate(args):
    simulator, *rest = args
    return simulator.simulate(*rest)


def _call_model(modeler, *args):
    return modeler.model(*args)


def _call_convergence(convergence, *args):
    return convergence.convergence(*args)


def _call_adapt(adapter, *args):
    return adapter.adapt(*args)


class Files:
    def __init__(self, *,
                 simulator,
                 modeler,
                 convergence,
                 adapter,
                 params):

        self.simulator = simulator
        self.modeler = modeler
        self.convergence = convergence
        self.adapter = adapter

        # Params
        params['spt'] = params['res_spt'] * params['res']
        # res_i   : within a 'trajectory', how many chunks we've done.
        # adapt_i : how many 'full' trajectories we've done.
        params['res_i'] = 0
        params['adapt_i'] = 0
        self.params = params

        self.results = []

    def any_converged(self, conv_futs):
        new_futs = set()
        for cfut in conv_futs:
            if cfut.ready():
                if cfut.successful():
                    result = cfut.get()
                    self.results += [result]
                    if result['converged']:
                        return True, None
                else:
                    # exception!
                    print("Error: Problem with convergence future")
                    cfut.get()
            else:
                new_futs.add(cfut)
        return False, new_futs

    def main_loop(self):
        sims = self.simulator.init(self.params)
        with Pool() as pool:
            while True:
                model_futs = set()
                conv_futs = set()
                model_fut = None

                for _ in range(self.params['res_spt']):
                    update(
                        sims,
                        pool.imap(_call_simulate,
                                  zip(
                                      rpt(self.simulator),
                                      last_of(sims),
                                      rpt(self.params)
                                  ))
                    )

                    conv_pipe, model_pipe = Pipe(duplex=False)
                    model_fut = pool.apply_async(
                        _call_model,
                        (self.modeler, sims, model_pipe, self.params)
                    )
                    model_futs.add(model_fut)
                    conv_fut = pool.apply_async(
                        _call_convergence,
                        (self.convergence, conv_pipe, self.params)
                    )
                    conv_futs.add(conv_fut)

                    model_futs = filter_futs(model_futs)
                    done, conv_futs = self.any_converged(conv_futs)

                    if done:
                        return self.results

                # Adapt
                sims = update_new(sims,
                                  self.adapter.adapt(model_fut.get(),
                                                     self.params)
                                  )
