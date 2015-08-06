from IPython.parallel import Client

client = Client()
lbv = client.load_balanced_view()

import itertools
import pandas as pd
from qaccel.dag import loop
from qaccel.deref import Deref
from qaccel.runs.ala_rnd import Run

tpr = [2 ** i for i in range(5)]
spt = [4 ** i for i in range(5)]
n_clones = 2
out_fn = 'dag.pickl'

run = Run(lbv)
dags = [run.make_dag({'tpr': tpr, 'res_spt': rspt, 'clone': clone})
        for tpr, rspt, clone
        in itertools.product(tpr, spt, range(n_clones))]

loop(dags, sleep=1)

dref = Deref(True)
convs = [[dref(ar.msg_ids[0]) for ar in dag.cars_all] for dag in dags]
df = pd.concat([pd.DataFrame(conv) for conv in convs])
df.tail()

df['steps'] = df['adapt_i'] * df['spt'] + df['res_i'] * df['res']
df['agg'] = df['steps'] * df['tpr']

df.to_pickle(out_fn)
