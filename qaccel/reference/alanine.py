"""Make a small transition matrix from alanine dipeptide trajectories.

Author: Matthew Harrigan
"""

import logging
import pickle

import numpy as np

from msmbuilder.msm import MarkovStateModel
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.example_datasets import fetch_alanine_dipeptide


log = logging.getLogger(__name__)


def ref_msm():
    """Load and return a saved MSM."""

    # TODO


def make_alanine_reference_data(dirname):
    """Make a small transition matrix from Alanine trajectories

    We featurize using phi / psi angles

    Note: This function is not-deterministic, although it would be useful
    if it were, so testing could be conducted.
    """
    fmt = dict(dirname=dirname)

    ala = fetch_alanine_dipeptide()
    msm, kmeans = generate_alanine_msm(ala)

    # Save cluster centers
    np.save("{dirname}/ala.centers.npy".format(**fmt), kmeans.cluster_centers_)

    # Save MSM Object
    with open("{dirname}/ala.msm.pickl".format(**fmt), 'wb') as f:
        pickle.dump(msm, f)


def generate_alanine_msm(ala):
    """Make a small transition matrix from Alanine trajectories

    We featurize using phi / psi angles

    Note: This function is not-deterministic, although it would be useful
    if it were, so testing could be conducted.
    """

    # Featurize
    dihed = DihedralFeaturizer()
    feat_trajs = dihed.transform(ala['trajectories'])

    # Cluster
    kmeans = MiniBatchKMeans(n_clusters=20, random_state=52)
    kmeans.fit(feat_trajs)

    # Build MSM
    msm = MarkovStateModel(lag_time=3, verbose=False)
    msm.fit(kmeans.labels_)

    return msm, kmeans


