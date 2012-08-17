Exploring the phase space
#########################

Introduction
============

This section assumes that one has defined a force-field model as explained in
the previous section, :ref:`ug_sec_forcefield`. The tools discussed in this
section allow one to explore the phase space of a system (and derive its
thermodynamic properties) using a force field model.

All algorithms are implemented such that they assume very little about the
internals of the force field models. The force field takes atomic positions and
cell vectors as input, and returns the energy (and optionally forces and a
virial tensor). All algorithms below are only relying on this basic interface.

Most of the algorithms are extensible through so-called `hooks`. These hooks are
pieces of code that can be plugged into a basic algorithm (like an NVE
integrator) to add functionality like writing trajectory files, sampling other
ensembles or computing statistical properties on the fly.

One important aspect of :mod:`yaff.analysis` is that that trajectory data can
be written to an HDF5 file. In short, HDF5 is a cross-platform format to store
efficiently any type of binary array data. A HDF5 file stores arrays
in a tree sturcture, which is similar to files and directories in a regular file
system. More details about HDF5 can be found on `wikipedia
<http://en.wikipedia.org/wiki/Hdf5>`_ and on the `non-profit HDF Group website
<http://www.hdfgroup.org/>`_. This format is designed to handle huge amounts of
binary data and it greatly facilitates post-processing analysis of the
trajectory data. By convention, Yaff stores all data in HDF5 files in atomic
units.


Molecular Dynacmis
==================

The NVE (microcanonical) ensemble
---------------------------------

The equations of motion in the NVE ensemble can be integrated as follows::

    nve = NVEIntegrator(ff, 1*femtosecond, temp0=300)
    nve.run(5000)

This example just propagates the system with 5000 steps of 1 fs, but does nearly
nothing else. After calling the ``run`` method, one can inspect atomic positions
and velocities of the final time step:

    print nve.vel
    print nve.pos
    print ff.system.pos  # equivalent to the previous line
    print nve.ekin/kjmol # the kinetic energy in kJ/mol.

By default all information from past steps is discarded. If one is interested
in writing a trajectory file, one must add a hook to do so. The following
example writes a HDF5 trajectory file:

    hdf5_writer = HDF5Writer(h5py.File('output.h5', mode='w'))
    nve = NVEIntegrator(ff, 1*femtosecond, hooks=hdf5_writer, temp0=300)
    nve.run(5000)

The parameters of the integrator can be tuned with several optional arguments of
the ``NVEIntegrator`` constructor. See
:class:``yaff.sampling.nve.NVEIntegrator`` for more details. The exact contents
of the HDF5 file depends on the integrator used and the optional arguments. The
typical tree structure of a trajectory HDF5 file is as follows::

    xxx

The ``hooks`` argument can be used to specify callback routines that are called
after every iteration or, using the ``start`` and ``step`` arguments, at
selected iterations. For example, the following HDF5 hook will write data every
100 steps, after the first 1000 iterations are carried out::

    hdf5_writer = HDF5Writer(h5py.File('output.h5', mode='w'), start=1000, step=100)

The hooks argument may also be a list of hook objects, e.g. to reset the
velocities every 200 steps, one may include the ``AndersenThermostat`` to
sample the NVT ensemble and ``XYZWriter`` to write a trajectory of the atomic
positions in XYZ format::

    hooks=[
        HDF5Writer(h5py.File('output.h5', mode='w')),
        AndersenThermostat(temp=300, step=200),
        XYZWriter('trajectory.xyz'),
    ]

By default a screen logging hook is added (if not yet present) to write one
line per iteration with some critical integrator parameters.

Other integrators are implemented such as ``NVTNoseIntegrator``,
``NVTLangevinIntegrator``, and so on.


**Geometry optimization**

One may also use a geometry optimizer instead of an integrator::

    hdf5 = HDF5Writer(h5py.File('output.h5', mode='w'))
    opt = CGOptimizer(ff, CartesianDOF(), hooks=hdf5)
    opt.run(5000)

The ``CartesianDOF()`` argument indicates that only the positions of the nuclei
will be optimized. The convergence criteria are controlled through optional
arguments of the ``CartesianDOF`` class. The ``run`` method has the maximum
number of iterations as the only optional argument. If ``run`` is called without
arguments, the optimization continues until convergence is reached.

One may also perform an optimization of the nuclei and the cell parameters is
follows::

    hdf5 = HDF5Writer(h5py.File('output.h5', mode='w'))
    opt = CGOptimizer(ff, CellDOF(FullCell()), hooks=hdf5)
    opt.run(5000)

This will transform the degrees of freedom (DOF's) of the system (cell vectors
and cartesian atomic coordinates) into a new set of DOF's (scaled cell vectors
and reduced coordinates) to allow an efficient optimization of both cell
parameters atomic positions. One may replace ``FullCell`` by ``AnisoCell`` or
``IsoCell``. The optional arguments of ``CellDOF`` also include convergence
criteria for the cell parameters.


**Vibrational analysis**

The Hessian is computed as follows::

    hessian = estimate_hessian(ff)

This function uses the symmetric finite difference approximation to estimate the
Hessian using many gradient computations. Further vibrational analysis based on
this Hessian can be carried out with TAMkin::

    hessian = estimate_hessian(ff)
    gpos = np.zeros(ff.system.pos.shape, float)
    epot = ff.compute(gpos)

    import tamkin
    mol = tamkin.Molecule(system.numbers, system.pos, system.masses, epot, gpos, hessian)
    nma = tamkin.NMA(mol)
    invcm = lightspeed/centimeter
    print nma.freqs/invcm

One may also compute the Hessian of a subsystem, e.g. for the first three atoms,
as follows::

    hessian = estimate_hessian(ff, select=[0, 1, 2])
