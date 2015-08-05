import time

from IPython.parallel import Client


class Deref:
    def __init__(self, parallel):
        self.ll = parallel
        self.client = None

    def __call__(self, var):
        if self.ll:
            if self.client is None:
                self.client = Client()
            return self.client.get_result(var).get()
        else:
            return var


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
    def __init__(self, *,
                 simulator,
                 modeler,
                 convergence,
                 adapter,
                 params):

        c = Client()
        self.lbv = c.load_balanced_view()

        self.simulator = simulator
        self.modeler = modeler
        self.convergence = convergence
        self.adapter = adapter

        # Results tracker
        self.cars_unknown = set()
        self.cars_all = list()

        # Params
        params['spt'] = params['res_spt'] * params['res']
        self.params = params

        # Initialize
        self.simulate_pars = [
            self.lbv.apply(_call_simulate_init, self.simulator, i)
            for i in range(self.params['tpr'])]
        self.model_par = None
        self.adapt_par = None
        self.res_i = 0

    def _submit_simulate_from_simulate(self):
        simulate_ars = []
        for simulate_par in self.simulate_pars:
            with self.lbv.temp_flags(after=[simulate_par]):
                simulate_ars.append(
                    self.lbv.apply(_call_simulate_from_simulate,
                                   self.simulator,
                                   simulate_par.msg_ids[0],
                                   self.params)
                )
        self.simulate_pars = simulate_ars

    def _submit_simulate_from_adapt(self):
        tpr = self.params['tpr']
        simulate_ars = []
        for i in range(tpr):
            with self.lbv.temp_flags(after=[self.adapt_par]):
                simulate_ars.append(
                    self.lbv.apply(_call_simulate_from_adapt,
                                   self.simulator,
                                   self.adapt_par.msg_ids[0], i, self.params)
                )

    def res_round(self):
        with self.lbv.temp_flags(after=self.simulate_pars):
            model_ar = self.lbv.apply(_call_model,
                                      self.modeler,
                                      [sar.msg_ids[0] for sar in
                                       self.simulate_pars],
                                      self.params)
        self.model_par = model_ar

        with self.lbv.temp_flags(after=[model_ar]):
            convergence_ar = self.lbv.apply(_call_convergence,
                                            self.convergence,
                                            model_ar.msg_ids[0],
                                            self.params)

        self.cars_unknown.add(convergence_ar)
        self.cars_all.append(convergence_ar)
        self.res_i += 1

    def adapt_round(self):
        with self.lbv.temp_flags(after=[self.model_par]):
            adapt_ar = self.lbv.apply(_call_adapt,
                                      self.adapter,
                                      self.model_par.msg_ids[0],
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
