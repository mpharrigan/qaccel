import time

from IPython.parallel import Client

c = Client()
lbv = c.load_balanced_view()


def simulate(start, params):
    import numpy

    import time
    time.sleep(10)

    return numpy.random.randint(0, 10, (50, 2))


def transform(traj_id, params):
    from IPython.parallel import Client
    c = Client()
    traj = c.get_result(traj_id).get()
    return traj[:, 0]


def update(model_id, traj_id, params):
    import numpy
    from IPython.parallel import Client
    c = Client()
    traj = c.get_result(traj_id).get()
    model = c.get_result(model_id).get()

    lu, lc = numpy.unique(traj, return_counts=True)
    for u, c in zip(lu, lc):
        model[u] += c

    return model


def convergence(model_id, params):
    from IPython.parallel import Client
    c = Client()
    model = c.get_result(model_id).get()

    import time
    time.sleep(10)

    return max(v for v in model.values()) / 100


def init_update():
    return dict((i, 0) for i in range(11))


def init_simulate():
    pass


class DAG:
    def __init__(self, *, simulate_from_simulate, simulate_from_adapt, model,
                 convergence, adapt, params):

        # Functions
        self.simulate_from_simulate = simulate_from_simulate
        self.simulate_from_adapt = simulate_from_adapt
        self.model = model
        self.convergence = convergence
        self.adapt = adapt

        # Results tracker
        self.cars_unknown = set()
        self.cars_all = list()

        # Params
        params['spt'] = params['res_spt'] * params['res']
        self.params = params

        # Initialize
        self.simulate_pars = [lbv.apply(init_simulate) for _ in
                              range(self.params['tpr'])]
        self.model_par = None
        self.adapt_par = None
        self.res_i = 0

    def _submit_simulate_from_simulate(self):
        simulate_ars = []
        for simulate_par in self.simulate_pars:
            with lbv.temp_flags(after=[simulate_par]):
                simulate_ars.append(
                    lbv.apply(self.simulate_from_simulate,
                              simulate_par.msg_ids[0],
                              self.params)
                )
        self.simulate_pars = simulate_ars

    def _submit_simulate_from_adapt(self):
        tpr = self.params['tpr']
        simulate_ars = []
        for i in range(tpr):
            with lbv.temp_flags(after=[self.adapt_par]):
                simulate_ars.append(
                    lbv.apply(self.simulate_from_adapt,
                              self.adapt_par.msg_ids[0], i, self.params)
                )

    def res_round(self):
        with lbv.temp_flags(after=self.simulate_pars):
            model_ar = lbv.apply(self.model,
                                 [sar.msg_ids[0] for sar in self.simulate_pars],
                                 self.params)
        self.model_par = model_ar

        with lbv.temp_flags(after=[model_ar]):
            convergence_ar = lbv.apply(self.convergence, model_ar.msg_ids[0],
                                       self.params)

        self.cars_unknown.add(convergence_ar)
        self.cars_all.append(convergence_ar)
        self.res_i += 1

    def adapt_round(self):
        with lbv.temp_flags(after=[self.model_par]):
            adapt_ar = lbv.apply(self.adapt, self.model_par.msg_ids[0],
                                 self.params)
        self.adapt_par = adapt_ar

    def round(self):
        if self.res_i < self.params['res_spt']:
            self._submit_simulate_from_simulate()
            self.res_round()
        else:
            self.adapt_round()
            self._submit_simulate_from_adapt()
            self.res_i = 0
            self.res_round()

    def is_converged(self):
        updated_cars_unknown = set()
        for car in self.cars_unknown:
            if car.ready():
                converged, _ = car.get()
                if converged:
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
        all_done = all_done and dag.is_conv()
        if len(dag.cars_unknown) < max_per:
            dag.round()
    return all_done


def loop(dags, sleep=2):
    while True:
        all_done = _multiround(dags)
        if all_done:
            break
        time.sleep(sleep)
