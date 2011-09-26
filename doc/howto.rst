Yaff Howto
##########

The description below is the way we would like Yaff to be. We are not there
yet. This is work in progress.


First steps
===========

Yaff is in fact a Python library that can be used to implement all sorts of
force field simulations. The conventional input file is therefore replaced by
an input script. This means that you must write a (small) main program that
specifies what type of simulation is carried out. On top of such a script, one
has to import the Yaff library. Numpy is used extensively in Yaff, and is in
general also useful in the input scripts::

    import numpy as np
    from yaff import *

A useful simulation typically consists of four steps:

1. Specification of the molecular system that will be simulated.
2. Specification of the force field model used to compute energies and forces.
3. An iterative procedure, such as a Molecular Dynanics or a Monte Carlo
   simulation, a geometry optimization, or yet something else.
4. Analysis of the output to extract meaningful numbers from the simulation.

These steps will be discussed briefly in the following sections. Yaff internally
works with atomic units, although other unit systems can be used in input and
output files.


Setting up a molecular system
=============================

Besides the positions of the atoms (or nuclei or pseudo-atoms) and the periodic
boundary conditions, two additional pieces of information are needed to define a
force field energy:

1. the bonds between the atoms (in the form of a list of atom pairs) and 
2. atom types.

Other properties such as valence angles, dihedral angles, assignment of energy
terms, exclusion rules, and so on, can be derived from these basic properties.
Some force-field models (especially those used for biomolecular simulations) 
have more topological features. However, we will stick to the basics discussed
here.

In Yaff, the ``System`` class is used to keep track of the following data:

* Positions of the atoms (in Cartesian coordinates, Bohr unit).
* 0, 1, 2 or 3 Cell vectors (in Cartesian coordinates, Bohr unit).
* Bonds (an array with pairs of atoms) and some derived properties.
* Atom types and atomic numbers.

The latter two are fixed during a simulation. If these need to be changed for
some purpose, just make a new instance of the System class. Atomic numbers are
also stored because they are useful for some output formats.

Creating a System instance
~~~~~~~~~~~~~~~~~~~~~~~~~~

The system class has the following signature:
    
.. autoclass:: yaff.System

The constructor arguments can be specified with some python code::

    system = System(
        numbers=np.array([8, 1, 1]*2),
        pos=np.array([[-4.583, 5.333, 1.560], [-3.777, 5.331, 0.943]])*angstrom,
        ffatypes=['O', 'H', 'H']*2,
        bonds=np.array([[(i/3)*3,i] for i in xrange(6) if i%3!=0]),
        rvecs=np.array([[9.865, 0.0, 0.0], [0.0, 9.865, 0.0], [0.0, 0.0, 9.865]])*angstrom,
    )

where the ``*angstrom`` converts the numbers from angstrom to atomic units.
Alternatively, one can load the system from one or more files::

    system = System.from_file('initial.xyz', 'topology.psf', cell=np.identity(3)*9.865*angstrom)
    
The ``from_file`` class method accepts one or more files and any constructor
argument from the System class. A system can be easily stored to a file using
the ``to_file`` method::

    system.to_file('last.chk')

where the ``.chk``-format is the standard text-based checkpoint file format for
systems in Yaff. It can also be used in the ``from_file`` method. A checkpoint
file may also contain other information, e.g. atomic velocities, which are
returned in a dictionary by the ``from_file`` method. (These are not assigned
as attributes of the system class.)

**TODO:**

1. import units in Yaff namespace
2. implement ``to_file`` and ``from_file``
3. provide a simple tool to assign atom types using rules.


Setting up an FF
================

Once the system is defined, one can continue with the specification of the force
field model. Such a model typically consists of several major term, hereafter
called `parts` that sum up to the total energy. One may have a valence part, a
electrostatic part, a repulsion part, and so on. In Yaff, these are all instance
of subclasses of the ``ForcePart`` class.

An example::

    vpart = ValencePart(system)
    scalings = Scalings(system.topology, scale1=0, scale2=1, scale3=1)
    nlists = NeighborLists(system)
    pair_pot_lj = PairPotLJ(sigmas, epsilons, rcut, smooth=True)
    pair_part_lj = PairPart(system, nlists, scalings, pair_pot_lj)
    ForceField(system, [valence_part, pair_part_lj], nlists)


Running an FF simulation
========================


Analyzing the results
=====================


