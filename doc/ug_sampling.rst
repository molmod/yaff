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


Molecular Dynamics
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


Advanced sampling methods
=========================

Yaff can be used for some advanced sampling methods such as umbrella sampling,
metadynamics, or variationally enhanced sampling. Such methods require the use
of a bias potential, which can be achieved with the
:class:`yaff.pes.ff.ForcePartBias` class, contributing to the total force-field
energy. This class supports two kinds of contributions, either from the
:class:`yaff.pes.vlist.ValenceTerm` class, or from the
:class:`yaff.pes.bias.BiasPotential` class.

In many cases, the bias potentials are very similar to expressions appearing
in the covalent part of the force field, because the collective variable is
simply a function of interatomic vectors. A bias can therefore be constructed
by combining an instance of :class:`yaff.pes.vlist.ValenceTerm` with an
instance of :class:`yaff.pes.iclist.InternalCoordinate`. Consider for example a
bias potential which is the cosine of a dihedral angle of atoms 0, 1, 2 and 3::

    ff = ForceField(...)
    part_bias = ForcePartBias(ff.system)
    ff.add_part(part_bias)
    cv = DihedralAngle(0,1,2,3) # Instance of InternalCoordinate
    m, a, phi0 = 1, 2.0, np.pi/4.0
    bias = Cosine(m, a, phi0, cv) # Instance of ValenceTerm
    part_bias.add_term(bias)

By making use of the :class:`yaff.pes.comlist.COMList` class, it is possible
to construct more complicated collective variables and use them for the bias
potential. Suppose that a plane is defined by atom 0, atom 1, and the average
position of atoms 2 and 3. The collective variable is the distance from this
plane to the average position of atoms 4, 5 and 6. A harmonic bias of this
collective variable can be achieved as follows::

    ff = ForceField(...)
    # Construct COMList; each group is a collection of atoms from which a
    # position is calculated as a weighted average of atomic positions
    groups = [ (np.array([0]), np.array([1.0])),
               (np.array([1]), np.array([1.0])),
               (np.array([2,3]), np.array([1.0/2.0,1.0/2.0])),
               (np.array([4,5,6]), np.array([1.0/3.0,1.0/3.0,1.0/3.0])) ]
    comlist = COMList(system, groups)
    part_bias = ForcePartBias(system, comlist=comlist)
    ff.add_part(part_bias)
    # Define a bias potential, harmonic in the out-of plane distance
    # from group 3 to the plane spanned by groups 0, 1 and 2
    cv = OopDist(0,1,2,3)
    K, cv0 = 1.0, 0.3
    bias = Harmonic(K, cv0, cv)
    # use_comlist=True makes that the positions of the groups are used
    part_bias.add_term(bias, use_comlist=True)

In some occasions, the collective variable cannot be expressed as a function
of interatomic vectors, for instance the volume of the simulation cell. In such
a case, an instance of :class:`yaff.pes.bias.BiasPotential` can be combined
with an instance of :class:`yaff.pes.colvar.CollectiveVariable`. For instance
a harmonic restraint on the volume can be constructed as follows::

    ff = ForceField(...)
    part_bias = ForcePartBias(ff.system)
    ff.add_part(part_bias)
    cv = CVVolume(ff.system) # Instance of CollectiveVariable
    K, V0 = 0.3, 12000.0
    bias = HarmonicBias(K, V0, cv) # Instance of BiasPotential
    part_bias.add_term(bias)

It can be necessary to keep track of the values of collective variables or the
bias potentials for postprocessing. This can be accomplished by making use of
the :class:`yaff.sampling.iterative.CVStateItem` and
:class:`yaff.sampling.iterative.BiasStateItem` classes as follows::

    cv_tracker = CVStateItem([cv0, cv1, ...])
    bias_tracker = BiasStateItem(part_bias)
    verlet = VerletIntegrator(..., state=[cv_tracker, bias_tracker])

Such a construction will write the values of the requested collective variables
and the contributions to the bias potential during a simulation to a HDF5 file.
Note that the ``bias_tracker`` will not work if terms are added during the
simulation.


Interface with PLUMED
---------------------

PLUMED is an open-source, community-developed library that provides a wide
range of different methods, including  enhanced-sampling algorithms and
free-energy methods. Just as many popular MD engines, Yaff works together with
PLUMED using the :class:`yaff.external.libplumed.ForcePartPlumed` class. This
class acts as a :class:`yaff.pes.ff.ForcePart` in the sense that it computes
energies, forces, and virials by making use of PLUMED.

A typical setup could look as follows::

    # Construct the unbiased PES
    ff = ForceField(...)
    # Construct the PLUMED contribution to the PES
    plumed = ForcePartPlumed(ff.system, fn='plumed.dat')
    ff.add_part(plumed)
    # Construct an integrator
    verlet = VerletIntegrator(ff, timestep)


Monte Carlo simulations
=======================

Yaff offers support for some basic Monte Carlo (MC) simulations. To quickly get
started, it is advisable to take a look at the examples discussed in
:ref:`tu_sec_montecarlo`. Below, a slightly more detailed discussion of the
inner workings of the MC code is provided.

Currently all MC routines assume pairwise-additive force fields and rigid
molecules/frameworks.


Changes compared to the standard ForceField
-------------------------------------------

An important distinction between molecular dynamics and MC simulations, is that
for MC simulations only the energy difference resulting from moving a few atoms
(for instance a single molecule) is required. This makes the standard
:class:`yaff.pes.ff.ForceField` class inadequate for computationally efficient
MC simulations, as its default behavior is to compute interactions between all
atoms. For pairwise-additive force fields (which is assumed to be the case
here), only interactions involving the moved atoms need to be calculated during
MC simulations.

This can be achieved using the `nlow` and `nhigh` keywords when constructing a
force field. When `nlow` is equal to M, interactions within the first M atoms
of the system are not computed. When `nhigh` is equal to M, interactions within
the other atoms of the system are not computed. By setting both `nlow` and
`nhigh` equal to M, only interactions between atoms with index smaller than
M and atoms with index higher than or equal to M are considered.

Suppose that you would like to perform an MC simulation of N molecules, each
consisting of n atoms. During the MC simulation, we need to compute for
instance the energy difference of translating a single molecule. A force field
that allows to do this can be constructed as follows::

    ff_guestguest = ForceField.generate(..., nlow=(N-1)*n, nhigh=(N-1)*n)

This ensures that interactions between the first N-1 molecules are not
computed. Of course, this only works if the translated molecule is the last
one, but luckily the trial moves (such as
:class:`yaff.sampling.mctrials.TrialTranslation`) are
implemented this way.

When MC simulations of guest molecules inside a framework are considered, an
additional force field describing interactions between a single guest and the
host need to be constructed. Similarly as before::

    ff_hostguest = ForceField.generate(..., nlow=framework.natom, nhigh=framework.natom)

where it is crucial that the guest molecule appears last in the System.

Because molecules/frameworks are assumed to be rigid, the covalent interactions
are irrelevant for these types of simulations. If covalent terms are present in
the force field, this should not influence simulation results as they do not
induce energy differences.


Ensembles
---------

The first supported ensemble is the canonical or NVT ensemble, implemented in
:class:`yaff.sampling.mc.CanonicalMC`. It is required that you provide a System
containing N guest molecules (inside the periodic cell that you want to
consider) and a ForceField describing the interactions of the last guest
molecule with all previous guests. Optionally, you can provide an external
potential such as interactions with a host framework. The `eguest` keyword
should contain the intramolecular energy of a single guest, but is actually
not necessary to provide here (the intramolecular energy automatically cancels
when considering only translations and rotations of a single molecule). Setting
up the simulation can be done as follows::

    guests = System.from_file(...)
    ff_guestguest = ForceField.generate(guests, nlow=(N-1)*n, nhigh=(N-1)*n)
    ff_hostguest = ForceField.generate(..., nlow=framework.natom, nhigh=framework.natom)
    mc = CanonicalMC(guests, ff_guestguest, external_potential=ff_hostguest)

Now it is time to set the external conditions (the temperature in this case)
and optionally specify trial moves and their relative probability before we
can run the MC simulation::

    mc.set_external_conditions(300*kelvin)
    mc.run(nsteps, mc_moves={'translation':1.0, 'rotation':1.0})

Simulations in other ensembles are performed in a very similar fashion, as the
main difference is just the nature of allowed MC moves. For instance the
extension to the ensemble commonly referred to as NPT, can be done by including
a :class:`yaff.sampling.mctrials.TrialVolumechange` move::

    mc = NPTMC(guests, ff_guestguest)
    mc.set_external_conditions(300*kelvin, 1*bar)
    mc.run(nsteps, mc_moves={'translation':1.0, 'rotation':1.0, volumechange=0.1},
        volumechange_stepsize=...)

An external potential can not be included in this case, because normally the
effect of a volume change on the external potential is not very useful. The
`volumechange_stepsize` keyword sets the maximal change in volume that is
attempted. Together with the `volumechange` probability this can be an
important parameter for these kind of simulations.

Finally, simulations in the grand-canonical (muVT) ensemble are supported by
:class:`yaff.sampling.mc.GCMC`. Compared to the canonical ensemble it is now
necessary to also include insertion and deletion moves. This poses some
technical problems, as Yaff ForceFields can not simply change their number
of atoms. Therefore, a separate force field for each number of adsorbed
molecules has to be constructed. Setting up such a simulation therefore
requires a method to generate ForceFields for arbitray number of guest
molecules::

    # A single guest molecule
    guest = System.from_file(...)
    # The single guest molecule is placed in the periodic box
    guest.cell = Cell(rvecs)
    # Return a ForceField for N guests (in system)
    def ff_generator(system, guest):
        return ForceField.generate(system, 'pars.txt', nlow=max(0,system.natom-guest.natom),
                nlow=max(0,system.natom-guest.natom), rcut=..., ...)
    # Keep track of average number of molecules
    screenlog = MCScreenLog(step=500)
    # Actual GCMC class
    gcmc = GCMC(guest, ff_generator, external_potential=ff_hostguest,
                hooks=[screenlog])

Instead of controlling the chemical potential of the external gas reservoir, it
is more intuitive to think about controlling the pressure. If the gas reservoir
is far from ideal-gas behavior, the fugacity should be used instead of the
pressure. This can be obtained using an equation of state, such as the van der
Waals description (:class:`yaff.pes.eos.vdWEOS`). The adsorbed number of guest
molecules can be simulated as follows::

    gcmc.set_external_conditions(300*kelvin, 1.0*bar)
    gcmc.run(500000, mc_moves={'insertion':1.0, 'deletion':1.0,
                 'translation':1.0, 'rotation':1.0})

Note that a large number of steps might be required to reach converged results.
