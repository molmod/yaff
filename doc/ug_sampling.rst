..
    : YAFF is yet another force-field code.
    : Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
    : Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
    : (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
    : stated.
    :
    : This file is part of YAFF.
    :
    : YAFF is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : YAFF is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>
    :
    : --

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
pieces of code that can be plugged into a basic algorithm (like a Verlet
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

Overview of the Verlet algorithms
---------------------------------

The equations of motion in the NVE ensemble can be integrated as follows::

    verlet = VerletIntegrator(ff, 1*femtosecond, temp0=300)
    verlet.run(5000)

This example just propagates the system with 5000 steps of 1 fs, but does nearly
nothing else. After calling the ``run`` method, one can inspect atomic positions
and velocities of the final time step::

    print verlet.vel
    print verlet.pos
    print ff.system.pos     # equivalent to the previous line
    print verlet.ekin/kjmol # the kinetic energy in kJ/mol.

By default all information from past steps is discarded. If one is interested
in writing a trajectory file, one must add a hook to do so. The following
example writes a HDF5 trajectory file::

    hdf5_writer = HDF5Writer(h5.File('output.h5', mode='w'))
    verlet = VerletIntegrator(ff, 1*femtosecond, hooks=hdf5_writer, temp0=300)
    verlet.run(5000)

The parameters of the integrator can be tuned with several optional arguments of
the ``VerletIntegrator`` constructor. See
:class:`yaff.sampling.verlet.VerletIntegrator` for more details. The exact contents
of the HDF5 file depends on the integrator used and the optional arguments of
the integrator and the :class:`yaff.sampling.io.HDF5Writer`. The typical tree
structure of a trajectory HDF5 file is as follows. (Comments were added manually
to the output of h5dump to describe all the arrays.)::

    $ h5dump -n production.h5
    HDF5 "production.h5" {
    FILE_CONTENTS {
     group      /
     group      /system                          # The 'system' group contains most attributes of the System class.
     dataset    /system/bonds
     dataset    /system/charges
     dataset    /system/ffatype_ids
     dataset    /system/ffatypes
     dataset    /system/masses
     dataset    /system/numbers
     dataset    /system/pos
     dataset    /system/rvecs
     group      /trajectory                      # The 'trajectory' group contains the time-dependent data.
     dataset    /trajectory/cell                 # cell vectors
     dataset    /trajectory/cons_err             # the root of the ratio of the variance on the conserved quantity
                                                 #     and the variance on the kinetic energy
     dataset    /trajectory/counter              # an integer counter for the integrator steps
     dataset    /trajectory/dipole               # the dipole moment
     dataset    /trajectory/dipole_vel           # the time derivative of the dipole moment
     dataset    /trajectory/econs                # the conserved quantity
     dataset    /trajectory/ekin                 # the kinetic energy
     dataset    /trajectory/epot                 # the potential energy
     dataset    /trajectory/epot_contribs        # the contributions to the potential energy from the force field parts.
     dataset    /trajectory/etot                 # the total energy (kinetic + potential)
     dataset    /trajectory/pos                  # the atomic positions
     dataset    /trajectory/rmsd_delta           # the RMSD change of the atomic positions
     dataset    /trajectory/rmsd_gpos            # the RMSD value of the Cartesian energy gradient (forces if you like)
     dataset    /trajectory/temp                 # the instantaneous temperature
     dataset    /trajectory/time                 # the time
     dataset    /trajectory/vel                  # the atomic velocities
     dataset    /trajectory/volume               # the (generalized) volume of the unit cell
     }
    }

The hooks argument may also be a list of hook objects. For example, one may
include the :class:`yaff.sampling.nvt.AndersenThermostat` to reset the velocities
every 200 steps. The :class:`yaff.sampling.io.XYZWriter` can be added to write a
trajectory of the atomic positions in XYZ format::

    hooks=[
        HDF5Writer(h5.File('output.h5', mode='w')),
        AndersenThermostat(temp=300, step=200),
        XYZWriter('trajectory.xyz'),
    ]

By default a screen logging hook is added (if not yet present) to print one line
per iteration with some critical integrator parameters. The output of the
``VerletIntegrator`` is as follows::

 VERLET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 VERLET Cons.Err. = the root of the ratio of the variance on the conserved
 VERLET             quantity and the variance on the kinetic energy.
 VERLET d-rmsd    = the root-mean-square displacement of the atoms.
 VERLET g-rmsd    = the root-mean-square gradient of the energy.
 VERLET counter  Cons.Err.       Temp     d-RMSD     g-RMSD   Walltime
 VERLET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 VERLET       0    0.00000      299.5     0.0000       93.7        0.0
 VERLET       1    0.15231      286.4     0.0133      100.1        0.0
 VERLET       2    0.17392      297.8     0.0132       90.6        0.0
 VERLET       3    0.19803      306.8     0.0137       82.1        0.0

The screen output is geared towards detecting simulation errors. The
parameters ``Cons.Err.``, ``Temp``, ``d-RMSD``, ``g-RMSD`` should exhibit only
minor fluctuations in a proper MD run, except when the system only consists of
just a few atoms. The wall time should increase at a somewhat constant rate.

It is often desirable to control the amount of data processed by the hooks, e.g.
to limit the size of the trajectory files and the amount of screen output.
Most hooks have ``start`` and ``step`` arguments for this purpose. Consider
the following example::

    hooks=[
        VerletScreenLog(step=100)
        HDF5Writer(h5.File('output.h5', mode='w'), start=5000, step=10),
        XYZWriter('trajectory.xyz', step=50),
        AndersenThermostat(temp=300, step=1000),
    ]

In this example, the screen output contains only one line per 100 NVE iterations.
The HDF5 trajectory only contains trajectory data starting from step 5000 with
intervals of 10 steps. The ``XYZwriter`` only contains the positions of the atoms
every 50 steps. The Andersen thermostat only resets the atomic velocities every
1000 steps.

For a detailed description of all options of the VerletIntegrator and the supported
hooks, we refer to the reference documentation:

* :class:`yaff.sampling.verlet.VerletIntegrator`: Generic Verlet integrator, whose
  functionality can be extended through hooks.
* :class:`yaff.sampling.io.HDF5Writer`: Writes HDF5 trajectory files and is
  compatible with most other algorithms discussed below.
* :class:`yaff.sampling.io.XYZWriter`: Writes XYZ trajectory files, which may be
  useful for visualization purposes.
* :class:`yaff.sampling.verlet.VerletScreenLog`: The Verlet screen logger.
* :class:`yaff.sampling.nvt.AndersenThermostat`: Switch from NVE to NVT with the
  Andersen thermostat.
* :class:`yaff.sampling.nvt.NHCThermostat`: Switch from NVE to NVT with the
  Nose-Hoover chains thermostat.
* :class:`yaff.sampling.nvt.LangevingThermostat`: Switch from NVE to NVT with
  the Langevin thermostat.
* :class:`yaff.sampling.npt.AndersenMcDonaldBarostat`: experimental
  support for the NpT ensemble.
* :class:`yaff.sampling.verlet.KineticAnnealing`: simulated annealing based on
  slow dissipation of the kinetic energy.


Initial atomic velocities
-------------------------

When no initial velocities are given to the constructor of the
``VerletIntegrator`` constructor, these velocities are randomly sampled from a
Poisson-Boltzmann distribution. The temperature of the distribution is
controlled by the ``temp0`` argument and if needed, the velocities can be
rescaled by using the ``scalevel0=True`` argument.

The default behavior is to not remove center-of-mass and global angular momenta.
However, for the Nose-Hoover thermostat, this is mandatory and done
automatically. For the computation of the instantanuous temperature, one must
know the number of degrees of freedom (``ndof``) in which the kinetic energy is
distributed. The default value for ``ndof`` is in line with the default initial
velocities. ``ndof`` is always set to 3N, except for the Nose-Hoover thermostat,
where ndof is set to the number of internal degrees of freedom.

One may specify custom initial velocities and ndof by using the ``vel0`` and
``ndof`` arguments of the ``VerletIntegrator`` constructor. The module
:mod:`yaff.samplling.utils` contains various functions to set up initial
velocities.


Geometry optimization
=====================

A basic geometry optimization (with trajectory output in an HDF5 file) is
implemented as follows::

    hdf5 = HDF5Writer(h5.File('output.h5', mode='w'))
    opt = CGOptimizer(CartesianDOF(ff), hooks=hdf5)
    opt.run(5000)

The ``CartesianDOF()`` argument indicates that only the positions of the nuclei
will be optimized. The convergence criteria are controlled through optional
arguments of the :class:`yaff.sampling.dof.CartesianDOF` class. The ``run`` method has the maximum
number of iterations as the only optional argument. If ``run`` is called without
arguments, the optimization continues until convergence is reached.

One may also perform an optimization of the nuclei and the cell parameters as
follows::

    hdf5 = HDF5Writer(h5.File('output.h5', mode='w'))
    opt = CGOptimizer(FullCellDOF(ff), hooks=hdf5)
    opt.run(5000)

This will transform the degrees of freedom (DOFs) of the system (cell vectors
and Cartesian coordinates) into a new set of DOF's (scaled cell vectors
and reduced coordinates) to allow an efficient optimization of both cell
parameters atomic positions. One may replace :class:`yaff.sampling.dof.FullCellDOF` by any of the following:

* :class:`yaff.sampling.dof.StrainCellDOF`: like ``FullCellDOF``, but constrains
  cell rotations. This should be equivalent to ``FullCellDOF`` and even more
  robust in practice.
* :class:`yaff.sampling.dof.IsoCellDOF`: only allows isotropic scaling of the
  unit cell.
* :class:`yaff.sampling.dof.AnisoCellDOF`: like ``FullCellDOF``, but fixes the
  angles between the cell vectors.
* :class:`yaff.sampling.dof.ACRatioCellDOF`: special case designed to study the
  breathing of MIL-53(Al).

The optional arguments of any ``CellDOF`` variant includes convergence criteria
for the cell parameters and the ``do_frozen`` option to freeze the fractional
coordinates of the atoms.


Harmonic approximations
=======================


Yaff can compute matrices of second order derivatives of the energy based on
symmetric finite differences of analytic gradients for an arbitrary DOF object.
This is the most general approach to compute such a generic Hessian::

    hessian = estimate_hessian(dof)

where ``dof`` is a DOF object like CellDOF and others discussed in the previous
section. The routines discussed in the following subsections are based on this
generic Hessian routine. See :mod:`yaff.sampling.harmonic` for a
description of the harmonic approximation routines.


Vibrational analysis
--------------------

The `Cartesian` Hessian is computed as follows::

    hessian = estimate_cart_hessian(ff)

This function uses the symmetric finite difference approximation to estimate the
Hessian using many analytic gradient computations. Further vibrational
analysis based on this Hessian can be carried out with TAMkin::

    hessian = estimate_cart_hessian(ff)
    gpos = np.zeros(ff.system.pos.shape, float)
    epot = ff.compute(gpos)

    import tamkin
    mol = tamkin.Molecule(system.numbers, system.pos, system.masses, epot, gpos, hessian)
    nma = tamkin.NMA(mol)
    invcm = lightspeed/centimeter
    print nma.freqs/invcm

One may also compute the Hessian of a subsystem, e.g. for the first three atoms,
as follows::

    hessian = estimate_cart_hessian(ff, select=[0, 1, 2])


Elastic constants
-----------------

Yaff can estimate the elastic constants of a system at zero Kelvin. Just like the
computation of the Hessian, the elastic constants are obtained from symmetric
finite differences of analytic gradient computations. The standard approach
is::

    elastic = estimate_elastic(ff)

where ``elastic`` is a symmetric 6 by 6 matrix with the elastic constants stored
in Voight notation. If the system under scrutiny does not change its relative
coordinates when the cell is deformed, one may use a faster approach:

    elastic = estimate_elastic(ff, do_frozen=True)

A detailed description of this routine can be found here:
:func:`yaff.sampling.harmonic.estimate_elastic`.
