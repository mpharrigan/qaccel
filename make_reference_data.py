"""A script to generate reference data for testing.

This can be run from the command line.
"""

import os
import argparse

from qaccel.reference import make_muller_reference_data, \
    make_alanine_reference_data, get_src_kinase_data


def make_reference_data(dirname, alanine=True, muller=True, srckinase=True):
    """Make reference data into a given directory.

    :param dirname: Where to put the files.
    :param alanine: Whether to make alanine data
    :param muller: Whether to make muller data
    :param srckinase: Whether to make srckinase data
    """
    try:
        os.mkdir(dirname)
    except OSError:
        pass

    if alanine:
        print('Making Muller Data')
        make_muller_reference_data(dirname)

    if muller:
        print('Making Alanine Data')
        make_alanine_reference_data(dirname)

    if srckinase:
        print("Getting Src Kinase Data")
        get_src_kinase_data(dirname, powers=[1, 10])


def parse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reference data for quant accel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dirname', help='Directory to write data',
                        default='./reference')

    # Flags
    parser.add_argument('--alanine', action='store_true', default=False)
    parser.add_argument('--muller', action='store_true', default=False)
    parser.add_argument('--srckinase', action='store_true', default=False)

    args = parser.parse_args()

    if (not args.alanine) and (not args.muller) and (not args.srckinase):
        make_reference_data(args.dirname)
    else:
        make_reference_data(args.dirname, alanine=args.alanine,
                            muller=args.muller, srckinase=args.srckinase)


if __name__ == "__main__":
    parse()
