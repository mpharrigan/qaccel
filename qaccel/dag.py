import time

from IPython.parallel import Client


# The following are helper functions to call a method on an object.
# Using anything fancier (e.g. a function that returns the appropriate
# function) will break with IPython.parallel

def _call_simulate_init(simulator, *args):
    return simulator.init(*args)


def _call_simulate_from_simulate(simulator, *args):
    return simulator.simulate_from_simulate(*args)


def _call_simulate_from_adapt(simulator, *args):
    return simulator.simulate_from_adapt(*args)


def _call_model(modeler, *args):
    return modeler.model(*args)


def _call_convergence(convergence, *args):
    return convergence.convergence(*args)


def _call_adapt(adapter, *args):
    return adapter.adapt(*args)


class DAG:
    """Run rounds as a directed acyclic graph of dependencies.
    """

    def __init__(self, lbv, *,
                 simulator,
                 modeler,
                 convergence,
                 adapter,
                 params):

        self.lbv = lbv

        self.simulator = simulator
        self.modeler = modeler
        self.convergence = convergence
        self.adapter = adapter

        # Results tracker
        self.cars_unknown = set()
        self.cars_all = list()

        # Params
        params['spt'] = params['res_spt'] * params['res']
        # res_i   : within a 'trajectory', how many chunks we've done.
        # adapt_i : how many 'full' trajectories we've done.
        params['res_i'] = 0
        params['adapt_i'] = 0
        self.params = params

        # Initialize simulation
        self.simulate_ars = [
            self.lbv.apply(_call_simulate_init, self.simulator, i)
            for i in range(self.params['tpr'])]
        self.all_simulate_mids = [[ar.msg_ids[0]] for ar in self.simulate_ars]

        # Initialize other values
        self.model_ar = None
        self.adapt_ar = None

    def _submit_simulate_from_simulate(self):
        tpr = self.params['tpr']
        simulate_ars = []
        for i in range(tpr):
            with self.lbv.temp_flags(after=[self.simulate_ars[i]]):
                sar = self.lbv.apply(_call_simulate_from_simulate,
                                     self.simulator,
                                     self.simulate_ars[i].msg_ids[0],
                                     self.params)

                # add to AR's for dependencies
                simulate_ars += [sar]
                # as well as *all* trajectory data
                #  --> add this chunk to one that has already started
                all_i = tpr * self.params['adapt_i'] + i
                self.all_simulate_mids[all_i].append(sar.msg_ids[0])

        # Save the list we constructed
        self.simulate_ars = simulate_ars

    def _submit_simulate_from_adapt(self):
        tpr = self.params['tpr']
        simulate_ars = []
        for i in range(tpr):
            with self.lbv.temp_flags(after=[self.adapt_ar]):
                sar = self.lbv.apply(_call_simulate_from_adapt,
                                     self.simulator,
                                     self.adapt_ar.msg_ids[0], i, self.params)
                # add AR's for dependencies
                simulate_ars += [sar]
                # as well as *all* trajectory data
                #  --> this will be the first chunk
                self.all_simulate_mids.append([sar.msg_ids[0]])

        self.simulate_ars = simulate_ars

    def res_round(self):
        with self.lbv.temp_flags(after=self.simulate_ars):
            model_ar = self.lbv.apply(_call_model, self.modeler,
                                      self.all_simulate_mids, self.params)
        self.model_ar = model_ar

        with self.lbv.temp_flags(after=[model_ar]):
            convergence_ar = self.lbv.apply(_call_convergence,
                                            self.convergence,
                                            model_ar.msg_ids[0],
                                            self.params)

        self.cars_unknown.add(convergence_ar)
        self.cars_all.append(convergence_ar)
        self.params['res_i'] += 1

    def adapt_round(self):
        with self.lbv.temp_flags(after=[self.model_ar]):
            adapt_ar = self.lbv.apply(_call_adapt,
                                      self.adapter,
                                      self.model_ar.msg_ids[0],
                                      self.params)
        self.adapt_ar = adapt_ar
        self.params['adapt_i'] += 1
        self.params['res_i'] = 0

    def round(self):
        if self.params['res_i'] < self.params['res_spt']:
            self._submit_simulate_from_simulate()
            self.res_round()
        else:
            self.adapt_round()
            self._submit_simulate_from_adapt()
            self.res_round()

    def is_converged(self):
        updated_cars_unknown = set()
        for car in self.cars_unknown:
            if car.ready():
                result = car.get()
                if result['converged']:
                    return True
                else:
                    pass
            else:
                updated_cars_unknown.add(car)
        self.cars_unknown = updated_cars_unknown
        return False


def _multiround(dags, max_per=10):
    all_done = True
    for i, dag in enumerate(dags):
        all_done = all_done and dag.is_converged()
        if len(dag.cars_unknown) < max_per:
            dag.round()
    return all_done


def loop(dags, sleep=2):
    while True:
        all_done = _multiround(dags)
        if all_done:
            break
        time.sleep(sleep)
