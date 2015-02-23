#!/usr/bin/env python3
# PBS -l nodes=1:ppn=16
# PBS -l walltime=72:00:00
# PBS -N ala_rnd
# PBS -j oe
# PBS -m bea
# PBS -M harrigan@stanford.edu

import os
import itertools
import pickle
import logging
from multiprocessing import Pool

import numpy

import qaccel
from qaccel.reference.alanine import get_ref_msm
from qaccel.adapt import Random
from qaccel.simulator import TMatSimulator
from qaccel.builder import MSMBuilder
from qaccel.convergence import Frobenius


os.chdir(os.environ.get("PBS_O_WORKDIR", "."))
qaccel.init_logging(logging.DEBUG)

# Define the search space
def get_params():
    spts = [2 ** i for i in range(4, 10)]
    tprs = [10 ** i for i in range(4)]
    for spt, tpr in itertools.product(spts, tprs):
        yield qaccel.Param(spt=spt, tpr=tpr, res=8, post_converge=1000)


# Define initial conditions
def initial(run, param):
    """Start in random states"""
    return numpy.random.randint(run.conv.true_n, size=param.tpr)


def run_clone(clone_i):
    folder = "clone-{}".format(clone_i)
    os.mkdir(folder)
    os.chdir(folder)

    # Prepare the calculation
    ref_msm = get_ref_msm()
    run = qaccel.Run(
        adapter=Random(ref_msm),
        simulator=TMatSimulator(ref_msm),
        builder=MSMBuilder(),
        convergence=Frobenius(ref_msm, cutoff=1e-1),
        initial_func=initial
    )

    # Run the calculation
    with Pool(16) as pool:
        results = pool.map(run.safe_run, get_params())

    # Save the results
    with open("results.pickl", 'wb') as f:
        pickle.dump(results, f)

    os.chdir('..')


for i in range(10):
    run_clone(i)
