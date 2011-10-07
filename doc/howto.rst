Yaff Howto
##########

The description below is a design document for the future version of Yaff. We are not
there yet. A bunch of code still needs to be written.


Introduction
============

Yaff is a Python library that can be used to implement all sorts of
force-field simulations. A useful simulation typically consists of four steps:

1. Specification of the molecular system that will be simulated.
2. Specification of the force field model used to compute energies and forces.
3. An (iterative) simulation protocol, such as a Molecular Dynamics or a Monte
   Carlo simulation, a geometry optimization, or yet something else.
4. Analysis of the output to extract meaningful numbers from the simulation.

In Yaff, the conventional input file is replaced by an input script. This means
that you must write a (small) main program that specifies what type of
simulation is carried out. This is a minimalistic example that includes the
four steps given above::

    # import the yaff library
    from yaff import *
    # control the amount of screen output and the unit system
    log.set_level(log.medium)
    log.set_unitsys(log.joule)

    # 1) specify the system
    system = System.from_file('system.chk')
    # 2) specify the force field
    ff = ForceField.generate(system, 'parameters.txt')
    # 3) Integrate Newton's equation of motion and write the trajectory in HDF5 format.
    nve = NVEIntegrateor(ff, 1*femtosecond, hooks=HDF5TrajectoryHook('output.h5'), temp0=300)
    nve.run(5000)
    # 4) perform an analysis, e.g. RDF computation for O_W O_W centers.
    indexes = ssytem.get_indexes('O_W')
    rdf = RDFAnalysis('output.h5', indexes)
    rdf.result.plot('rdf.png')

These steps will be discussed in more detail in the following sections.

Yaff internally works with atomic units, although other unit systems can be used
in input and (some) output files. The units used for the screen output are
controlled with the ``log.set_unitsys`` method. Output written in (binary) HDF5
files will always be in atomic units. When output is written to a format from
other projects/programs, the units of that program/project will be used.

Numpy and Cython are used extensively in Yaff for numerical efficiency. The
examples below often use Numpy too, assuming the following import statement::

    import numpy as np


Setting up a molecular system
=============================

A ``System`` instance in Yaff contains all the physical properties of a
molecular system plus some extra information that is needed to define a force
field.

**Basic physical properties:**

#. Atomic numbers
#. Positions of the atoms
#. 0, 1, 2, or 3 Cell parameters (optional)
#. Atomic charges (optional)
#. Atomic masses (optional)

**Basic auxiliary properties:** (needed to define FF)

#. The bonds between the atoms (in the form of a list of atom pairs)
#. Atom types.

Other properties such as valence angles, dihedral angles, assignment of energy
terms, exclusion rules, and so on, can be derived from these basic properties.

The positions and the cell parameters
may change during the simulation. All other properties (including the number of
atoms and cell vectors) do not change during a simulation. If such changes seem
to be necessary, one should create a new System class instead.

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

where the ``.chk``-format is the standard text-based checkpoint file format in
Yaff. It can also be used in the ``from_file`` method.

**TODO:**

#. Add possibility to read system from a HDF5 output file.

#. [LOW PRIORITY] Introduce fragment name spaces in the atom types, e.g. instead
   of using O_W and H_W, we should have WATER:O, WATER:H. In the system object,
   we should have an extra fragment dictionary to know which atom is part of
   what sort of fragment, e.g. something like: ``system.fragments = {'WATER':,
   np.array([0, 1, 2, 3, 4, 5, ...])}``. The corresponding atom types can simply
   be ``system.ffatypes=['O', 'H', 'H, 'O', 'H', 'H, ...]``. It is OK that
   different atoms in different fragments have coinciding atom type names. This
   approach has the following advantages:

   * It allows us to develop separate parameter files with sections for
     different sorts of fragments, e.g. WATER, CO2, ALANINE, GLYCINE, MIL-53,
     ZEO, IONS, ...

   * A simple concatenation of parameter files for different fragments gives
     us a big parameter file that can used to model mixed systems.

   * The atom types can be kept short because they only have to be different
     within one fragment.

   It also introduces some (minor) extra difficulties:

   * In some cases, e.g. peptides, chemical bonds connect different fragments.
     In such cases, we should allow fragments to overlap.

   * We must introduce mixing rules for all types of non-bonding interactions
     or we have to introduce cross-parameter files. (The latter may be very
     annoying when pursuing more advanced simulations where molecules are
     gradually switched on and off.)

   Final thought: we can make the entire thing optional, i.e. when
   system.fragments is None, we can have the behavior without separate
   namespaces. This is convenient when testing a new FF for one sort of
   fragment.

#. [LOW PRIORITY] Provide a simple tool to automatically assign bonds and atom
   types using rules. (For the moment we hack our way out with the ``molmod``
   package.)


Setting up an FF
================

Once the system is defined, one can continue with the specification of the force
field model. The simplest way to create a force-field is as follows::

    ff = ForceField.generate(system, 'parameters.txt')

where the file ``parameters.txt`` contains all force field parameters. See XXX
for more details on the format of the parameters file. Additional `technical`
parameters that determine the behavior of the force field, such as the
real-space cutoff, the verlet skin, and so on, may be specified as keyword
arguments in the ``generate`` method. See XXX for a detailed description of the
``generate`` method.

Once an ``ff`` object is created, it can be used to evaluate the energy (and
optionally the forces) for a given set of Cartesian coordinates and/or cell
parameters::

    # change the atomic positions and cell parameters
    ff.update_pos(new_pos)
    ff.update_rvecs(new_rvecs)
    # compute the energy
    new_energy = ff.compute()

One may also allocate arrays to store the derivative of the energy towards
the atomic positions and uniform deformations of the system::

    # allocate arrays for the Cartesian gradient of the energy and the virial
    # tensor.
    gpos = np.zeros(system.pos.shape, float)
    vtens = np.zeros((3,3), float)
    # change the atomic positions and cell parameters
    ff.update_pos(new_pos)
    ff.update_rvecs(new_rvecs)
    # compute the energy
    new_energy = ff.compute(gpos, vtens)

This will take a little more CPU time because the presence of the optional
arguments implies that a lot of partial derivatives must be computed.

After the ``compute`` method is called, one can obtain a lot of intermediate
results by accessing attributes of the ``ff`` object. Some examples::

    print ff.part_pair_ei.energy/kjmol
    print ff.part_valence.gpos
    print ff.part_ewald_cor.vtens

Depending on the system and the contents of the file ``parameters.txt`` some
``part_*`` attributes may not be present. All parts are also accessible through
the list ``ff.parts``.

Instead of using the ``ForceField.generate`` method, one may also construct all
the parts of the force field manually. However, this can become very tedious.
This is a simple example of a Lennard-Jones force field::

    system = System(
        numbers=np.array([18]*10),
        pos=np.random.uniform(0, 10*angstrom, (10,3)),
        ffatypes=['Ar']*10,
        bonds=None,
        rvecs=np.identity(3)*10*angstrom,
    )
    sigmas = np.array([3.98e-4]*10),
    epsilons = np.array([6.32]*10),
    pair_pot_lj = PairPotLJ(sigmas, epsilons, rcut=15*angstrom, smooth=True)
    nlists = NeighborLists(system)
    scalings = Scalings(system.topology)
    part_pair_lj = ForcePartPair(system, nlists, scalings, pair_pot_lj)
    ff = ForceField(system, [part_pair_lj], nlists)


**TODO:**

#. Document the format of ``parameters.txt``. This should be done very
   carefully. I'm currently thinking of something along the lines of the CHARMM
   parameter file, but with a few extra features to make the format more
   general:

    a. Introduce sections for different namespaces (see above, low priority)
    b. Include charges based on reference charges and charge-transfers over
       bonds. Dielectric background for fixed charge models.
    c. prefix each line with a keyword that fixes the interpretation of the
       parameters that follow, e.g. ``EXPREP:PARS O H 100.0 4.4``
    d. Configurable units, e.g. ``EXPREP:UNIT A au``.
    e. Allow comments with #
    f. Put multiple related parameters on a single line for the sake of
       compactness.
    g. Make the format very simple, such that it can be easily written/modified
       manually in a text editor.
    h. Make it doable to convert existing sets of parameters to our file format.
    i. Make the format easily extensible, in case we come up with new energy
       terms. (or things like ACKS2)
    j. Specification of mixing rules.
    k. Specification of exclusion/scaling rules.

   We must keep in mind that not all parameters come from MFit2, or even FFit2
   in general. We just have to make sure that all FFit2 components (and other
   scripts) can write parameters in this format.

   I've made a tentative example for a (reasonable) non-polarizable water FF:

   .. literalinclude:: ../input/parameters_water.txt

#. [PARTIALLY DONE, TODO: TORSION, DAMPDISP, LJ, MM3, GRIMME] The generate method.


Running an FF simulation
========================


Molecular Dynacmis
------------------

The equations of motion in the NVE ensemble can be integrated as follows::

    nve = NVEIntegrateor(ff, 1*femtosecond, hooks=HDF5TrajectoryHook('output.h5'), temp0=300)
    nve.run(5000)

The parameters of the integrator can be tuned with several optional arguments of
the ``NVEIntegrator`` constructor. See XXX for more details. Once the integrator
is created, the ``run`` method can be used to compute a given number of time
steps. The trajectory output is written to a HDF5 file. The exact contents of
the HDF5 file depends on the integrator used and the optional arguments. All
data in the HDF5 file is stored in atomic units.

The ``hooks`` argument can be used to specify callback routines that are called
after every iteration or, using the ``start`` and ``step`` arguments, at
selected iterations. For example, this HDF5 hook will write data every 100
steps, after the first 1000 iterations are carried out::

    HDF5TrajectoryHook('output.h5', start=1000, step=100)

The hooks argument may also be a list of hook objects, e.g. to reset the
velocities every 200 steps, one may include the ``AndersonTHook``::

    hooks=[
        HDF5TrajectoryHook('output.h5', start=1000, step=100),
        AndersonTHook(temp=300, step=200)
    ]

By default a screen logging hook is added (if not yet present) to write one
line per iteration with some critical integrator parameters.

Other integrators are implemented such as NVTNoseIntegrator,
NVTLangevinIntegrator, and so on.

Geometry optimization
---------------------

One may also use a geometry optimizer instead of an integrator::

    opt = CGOptimizer(ff, hooks=HDF5TrajectoryHook('output.h5', start=1000, step=100))
    opt.run(5000)

Again, convergence criteria are controlled through optional arguments of the
constructor. the ``run`` method has the maximum number of iterations as the only
argument. By default the positions of the atoms or optimized, without changing
the cell vectors. This behavior can be changed through the ``dof_transform``
argument::

    opt = CGOptimizer(ff, dof_transform=cell_opt, hooks=HDF5TrajectoryHook('output.h5', start=1000, step=100))
    opt.run(5000)

This will transform the degrees of freedom (DOF's) of the system (cell vectors
and cartesian atomic coordinates) into a new set of DOF's (scaled cell vectors
and reduced coordinates) to allow an efficient optimization of both cell
parameters atomic positions. Several other dof_transform options are discussed
in XXX.


**TODO:**

#. Check if we can do something like the Andersen thermostat to simulate a
   constant pressure ensemble.

#. Check how to append data efficiently in HDF5 file. Add rows one by one or
   add rows in blocks.

#. ``RefTraj`` derivative of the Iterative class.

#. Optimizer stuff. We should use the molmod optimizer, but change it such
   that the main loop of the optimizer is done in Yaff instead of in molmod.

#. Numerical (partial) Hessian


Analyzing the results
=====================

The analysis of the results is (in the first place) based on the output
file ``output.h5``. On-line analysis (during the iterative algorithm, without
writing data to disk) is also possible.

Slicing the data
----------------

All the analysis routines below have at least the following four optional
arguments:

* ``start``: the first sample to consider for the analysis
* ``end``: the last sample to consider for the analysis
* ``step``: consider only a sample each ``step`` iterations.
* ``max_sample``: consider at most ``max_sample`` number of samples.

The last option is only possible when ``step`` is not specified and the total
number of samples (or ``end``) is known. The optimal value for ``step`` will be
derived from ``max_sample``.

**TODO:**

#. Support these arguments in all analysis routines.


Basic analysis
--------------

A few basic analysis routines are provided to quickly check the sanity of an MD
simulation:

* ``plot_energies`` makes a plot of the kinetic and the total energy as function
  of time. For example::

    plot_energies('output.h5')

  makes a figure ``energies.png``.

* ``plot_temperate`` is similar, but plots the temperature as function of time.

* ``plot_temp_dist`` plots the distribution (both pdf and cdf) of the
  instantaneous atomic and system temperatures and compares these with the
  expected analytical result for a constant-temperature ensemble. For example:

    plot_temp_dist('output.h5')

  makes a figure ``temp_dist.png``

All these functions accept optional arguments to tune their behavior. See XXX
for more details.

**TODO:**

#. Add cdf and system temperature dist to ``plot_temp_dist``.


Advanced analysis
-----------------

Yaff also includes analysis tools that can extract relevant macroscopic
properties from a simulation. These analysis tools require some additional
computations that can either be done in a post-processing step, or on-line.

* A radial distribution function is computed as follows::

    indexes = system.get_indexes('O_W')
    rdf = RDFAnalysis('output.h5', indexes)
    rdf.result.plot('rdf.png')

  The results are included in the HDF5 file, and optionally plotted using
  matplotlib. Alternatively, the same ``RDFAnalysis`` class can be used for
  on-line analysis, without the need to store huge amounts of data on disk::

    indexes = system.get_indexes('O_W')
    rdf = RDFAnalysis(None, indexes)
    nve = NVEIntegrator(ff, hooks=rdf, temp0=300)
    nve.run(5000)
    rdf.result.plot('rdf.png')

  The analysis keyword must obviously also accept a list of analysis objects.


**TODO:**

#. Implement RDF. Check how we can write things to files in the on-line case.
   Is it OK that both RDFAnalysis and HDF5TrajectoryHook open the same HDF5 file
   for writing data? Is this OK or not? ::

    indexes = system.get_indexes('O_W')
    rdf = RDFAnalysis('output.h5', indexes, on_line=True)
    hdf5 = HDF5TrajectoryHook('output.h5', start=1000, step=100)
    nve = NVEIntegrator(ff, hooks=[rdf, hdf5], temp0=300)
    nve.run(5000)
    rdf.result.plot('rdf.png')

   The RDF analysis must have a real-space cutoff that is smaller than the
   smallest spacing of the periodic cells.

#. Implement spectral analysis.

#. Implement autocorrelation function.

#. Port other things from MD-Tracks, including the conversion stuff.
