"""Src Kinase Transition matrix system from Diwakar

Author: Matthew Harrigan
"""

import logging
import tarfile
import os
import urllib
import pickle
import shutil

import scipy.io
import mdtraj as md
import numpy as np
from msmbuilder.msm import MarkovStateModel
from msmbuilder.decomposition import PCA
from msmbuilder.featurizer import DihedralFeaturizer

from .util import get_fn


log = logging.getLogger(__name__)

SRC = dict(
    SRC_URL="https://stacks.stanford.edu/file/druid:cm993jk8755/",
    SRC_FILE="MSM_2000states_csrc.tar.gz",
    SRC_DIR="srckinase",
)


def get_ref_msm():
    """Load and return a saved MSM."""
    with open(get_fn('src.msm.pickl'), 'rb') as f:
        return pickle.load(f)


def _download(source, tar_dest, untar_dest):
    """Download a tar file and extract it."""
    with urllib.request.urlopen(source) as tmat_tar_url:
        with open(tar_dest, 'wb') as tmat_tar_f:
            tmat_tar_f.write(tmat_tar_url.read())
        with tarfile.open(tar_dest) as tmat_tar_f:
            tmat_tar_f.extractall(untar_dest)


def get_src_kinase_data(dirname, cleanup=True):
    """Get the 2000 state msm from Stanford's SDR."""
    fmt = dict(dirname=dirname, **SRC)

    try:
        os.mkdir("{dirname}/{SRC_DIR}".format(**fmt))
    except OSError:
        pass

    # Fetch data
    _download(
        "{SRC_URL}/{SRC_FILE}".format(**fmt),
        "{dirname}/{SRC_DIR}/{SRC_FILE}".format(**fmt),
        "{dirname}/{SRC_DIR}".format(**fmt)
    )

    # Load and convert
    msm, centers = generate_srckinase_msm(
        tmat_fn="{dirname}/{SRC_DIR}/Data_l5/tProb.mtx".format(**fmt),
        pops_fn="{dirname}/{SRC_DIR}/Data_l5/Populations.dat".format(**fmt),
        mapping_fn="{dirname}/{SRC_DIR}/Data_l5/Mapping.dat".format(**fmt),
        gens_fn="{dirname}/{SRC_DIR}/Gens.lh5".format(**fmt)
    )

    np.save("{dirname}/src.centers.npy".format(**fmt), centers)


    # Save MSM Object
    with open("{dirname}/src.msm.pickl".format(**fmt), 'wb') as f:
        pickle.dump(msm, f)

    # Optionally, delete all data
    if cleanup:
        shutil.rmtree("{dirname}/{SRC_DIR}".format(**fmt))


def generate_srckinase_msm(tmat_fn, pops_fn, mapping_fn, gens_fn):
    msm = _generate_msm(tmat_fn, pops_fn)
    centers = _generate_centers(mapping_fn, gens_fn)
    return msm, centers


def _generate_msm(tmat_fn, populations_fn):
    tmat_sparse = scipy.io.mmread(tmat_fn)
    tmat_dense = tmat_sparse.toarray()

    populations = np.loadtxt(populations_fn)

    # Data indicates a lag time of 5. Note this is not really relevant
    # Make sure in convergence checks to divide timescales by this
    # for an accurate comparison
    msm = MarkovStateModel(lag_time=5)
    msm.n_states_ = tmat_dense.shape[0]
    msm.mapping_ = dict(zip(np.arange(msm.n_states_), np.arange(msm.n_states_)))
    msm.transmat_ = tmat_dense
    msm.populations_ = populations

    # Force eigensolve and check consistency
    computed_pops = msm.left_eigenvectors_[:, 0]
    computed_pops /= np.sum(computed_pops)

    np.testing.assert_allclose(computed_pops, msm.populations_)

    return msm


def _generate_centers(mapping_fn, gens_fn):
    mapping = np.loadtxt(mapping_fn)

    gens = md.load(gens_fn)
    gens = gens[mapping != -1]

    dihed = DihedralFeaturizer(['phi', 'psi'])
    dihedx = dihed.fit_transform([gens])

    pca = PCA(n_components=2)
    pcax = pca.fit_transform(dihedx)[0]

    return pcax


