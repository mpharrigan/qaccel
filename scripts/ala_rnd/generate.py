import itertools
import pickle
import os

from jinja2 import Environment, FileSystemLoader

from qaccel.runs.ala_rnd import Run

os.makedirs("runs/")
os.makedirs("jobs/")
os.makedirs("outs/")

env = Environment(loader=FileSystemLoader("."))
job_templ = env.get_template("job.pbs.template")

tpr = [2 ** i for i in range(5)]
spt = [4 ** i for i in range(5)]
n_clones = 2

run = Run()
runs = [run.make_run({'tpr': tpr, 'res_spt': rspt, 'clone': clone})
        for tpr, rspt, clone
        in itertools.product(tpr, spt, range(n_clones))]

for r in runs:
    run_fn = "runs/{}.pickl".format(r.params['param_str'])
    job_fn = "jobs/{}.pbs".format(r.params['param_str'])
    out_fn = "outs/{}.pickl".format(r.params['param_str'])
    with open(run_fn, 'wb') as f:
        pickle.dump(r, f)
    with open(job_fn, 'w') as f:
        f.write(job_templ.render(run_fn=run_fn, out_fn=out_fn))
