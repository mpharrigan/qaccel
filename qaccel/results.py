"""Parse result files."""

import pickle
import pandas as pd


class Results:
    """Represents a collection of runs.

    :param fns: Pickle file names to load
    :param window: passed to Result.find_first_convergence()
    :param cutoff: passed to Result.find_first_convergence()
    """

    def __init__(self, fns, window=12, cutoff=0.8):
        self.results = []
        for fn in fns:
            with open(fn, 'rb') as f:
                _results = pickle.load(f)

            self.results += [Result(df, param, window, cutoff) for df, param in
                             _results]

        self._param_df = None
        self._df = None

    def __iter__(self):
        yield from self.results

    def _get_param_df(self):
        params = set()
        for result in self:
            params.add(result.param)
        self._param_df = pd.DataFrame(p.__dict__ for p in params)

    def _get_df(self):
        self._df = pd.DataFrame(r.record() for r in self)

    @property
    def param_df(self):
        """A pandas data frame of parameters over all the results."""
        if self._param_df is None:
            self._get_param_df()
        return self._param_df

    @property
    def df(self):
        """A pandas data frame of convergence information."""
        if self._df is None:
            self._get_df()
        return self._df

    def where(self, **kv):
        """Pandas rows that satisfy conditions

        :param kv: Conditions of type df[key] == value. `and`ed together
        """
        condition = True
        for k, v in kv.items():
            condition &= (self.df[k] == v)
        return self.df[condition]


class Result:
    """Represents a single run.

    :param df: Data frame of error over time
    :param window: passed to find_first_convergence()
    :param cutoff: passed to find_first_convergence()

    If window or cutoff is None, you must call find_first_convergence()
    yourself.
    """

    def __init__(self, df, param, window=None, cutoff=None):
        self.df = df
        self.param = param
        self.steps = None
        self.rounds = None

        if window is not None and cutoff is not None:
            self.find_first_convergence(window, cutoff)

    def find_first_convergence(self, window=12, cutoff=0.8):
        """Use a rolling average to find first convergence.

        Specifically, we compute a rolling average over the `converged`
        boolean values (False -> 0; True -> 1) so fluctuations into and
        out of convergence can be mitigated.

        :param window: Rolling average window (in units of data frame rows)
        :param cutoff: Cutoff after which we are converged. Should be in the
                       range (0, 1]
        """
        rolling_df = self.df.copy()
        rolling_df['rolling_conv'] = pd.rolling_mean(rolling_df['converged'],
                                                     window).fillna(0)
        loci = (rolling_df['rolling_conv'] >= cutoff).argmax()
        rounds = rolling_df['round_i'].loc[loci] + 1
        steps = self.param.spt * rounds
        self.steps = steps
        self.rounds = rounds
        return steps, rounds

    def record(self):
        """Return a record for constructing a pandas data frame."""
        return dict(steps=self.steps, rounds=self.rounds, **self.param.__dict__)