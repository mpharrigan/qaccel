"""Generate a particle on a Muller potential surface.

Author: Matthew Harrigan
"""

import mdtraj as md
import numpy as np
from simtk import unit
from simtk import openmm


def _serialize(system, integrator, sys_fn, int_fn):
    """Serialize openmm system and integrator to files."""
    with open(sys_fn, 'w') as sys_f:
        sys_f.write(openmm.XmlSerializer.serialize(system))

    with open(int_fn, 'w') as int_f:
        int_f.write(openmm.XmlSerializer.serialize(integrator))


def make_muller_reference_data(dirname):
    """Make OpenMM system.xml and integrator.xml files.

    :param dirname: Where to put the files.
    """
    fmt = dict(dirname=dirname)
    system, integrator = generate_muller_sysint()
    _serialize(
        system, integrator,
        "{dirname}/muller.sys.xml".format(**fmt),
        "{dirname}/muller.int.xml".format(**fmt)
    )


def generate_muller_sysint():
    """Set up muller potential."""

    mass = 12.0 * unit.dalton
    temperature = 2000 * unit.kelvin
    friction = 100 / unit.picosecond
    timestep = 5.0 * unit.femtosecond

    # Prepare the system
    system = openmm.System()
    mullerforce = MullerForce()
    system.addParticle(mass)
    mullerforce.addParticle(0, [])
    system.addForce(mullerforce)

    # Prepare the integrator
    integrator = openmm.LangevinIntegrator(temperature, friction, timestep)

    return system, integrator


def make_traj_from_coords(xyz):
    """Take numpy array and turn it into a one particle trajectory

    :param xyz: (n_frames, 3) np.ndarray
    """

    top = md.Topology()
    chain = top.add_chain()
    resi = top.add_residue(None, chain)
    top.add_atom("C", md.element.carbon, resi)

    xyz = np.asarray(xyz)
    xyz = xyz[:, np.newaxis, :]
    traj = md.Trajectory(xyz, top)
    return traj


class MullerForce(openmm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    strength = 0.5

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], bb=self.bb[j], cc=self.cc[j],
                       AA=self.AA[j], XX=self.XX[j], YY=self.YY[j])
            expression += """+ {AA}*exp({aa} *(x - {XX})^2 +
                                {bb} * (x - {XX}) * (y - {YY}) +
                                {cc} * (y - {YY})^2)""".format(**fmt)

        # Include scaling expression
        expression = ("{strength}*(".format(strength=self.strength) +
                      expression + ")")

        super().__init__(expression)

    @classmethod
    def term(cls, x, y, j):
        return np.exp(cls.aa[j] * (x - cls.XX[j]) ** 2 +
                      cls.bb[j] * (x - cls.XX[j]) * (y - cls.YY[j]) +
                      cls.cc[j] * (y - cls.YY[j]) ** 2)

    @classmethod
    def potential(cls, x, y):
        """Compute the potential at a given point x, y"""
        value = np.zeros_like(x)
        for j in range(4):
            value += cls.AA[j] * cls.term(x, y, j)
        return value * cls.strength

    @classmethod
    def get_bounds(cls):
        # xmin, xmax, ymin, ymax
        bounds = [-1.5, 1.2, -0.2, 2.0]
        return bounds
