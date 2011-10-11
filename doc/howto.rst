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

    # import the h5py library to write output in the HDF5 format.

    # 1) specify the system
    system = System.from_file('system.chk')
    # 2) specify the force field
    ff = ForceField.generate(system, 'parameters.txt')
    # 3) Integrate Newton's equation of motion and write the trajectory in HDF5 format.
    f = h5py.File('output.h5', mode='w')
    hdf5_writer = HDF5Writer(f)
    nve = NVEIntegrateor(ff, 1*femtosecond, hooks=hdf5_writer, temp0=300)
    nve.run(5000)
    # 4) perform an analysis, e.g. RDF computation for O_W O_W centers.
    indexes = ssytem.get_indexes('O_W')
    rdf = RDFAnalysis(f, indexes)
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

#. [LOW PRIORITY] Add possibility to read system from a HDF5 output file.

#. Introduce fragment name spaces in the atom types, e.g. instead
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
   name spaces. This is convenient when testing a new FF for one sort of
   fragment.

#. Make the checkpoint format more compact.

#. Provide a simple tool to automatically assign bonds and atom types using
   rules. (For the moment we hack our way out with the ``molmod`` package.) See
   also TODO item below.


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

    hdf5_writer = HDF5Writer(h5py.File('output.h5', mode='w'))
    nve = NVEIntegrateor(ff, 1*femtosecond, hooks=hdf5_writer, temp0=300)
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

    hdf5_writer = HDF5Writer(h5py.File('output.h5', mode='w'), start=1000, step=100)

The hooks argument may also be a list of hook objects, e.g. to reset the
velocities every 200 steps, one may include the ``AndersonThermostat``::

    hooks=[
        HDF5Writer(h5py.File('output.h5', mode='w'))
        AndersonThermostat(temp=300, step=200)
    ]

By default a screen logging hook is added (if not yet present) to write one
line per iteration with some critical integrator parameters.

Other integrators are implemented such as NVTNoseIntegrator,
NVTLangevinIntegrator, and so on.


Geometry optimization
---------------------

One may also use a geometry optimizer instead of an integrator::

    opt = CGOptimizer(ff, hooks=HDF5Writer(h5py.File('output.h5', mode='w')))
    opt.run(5000)

Again, convergence criteria are controlled through optional arguments of the
constructor. the ``run`` method has the maximum number of iterations as the only
argument. By default the positions of the atoms or optimized, without changing
the cell vectors. This behavior can be changed through the ``dof_transform``
argument::

    opt = CGOptimizer(ff, dof_transform=cell_opt, hooks=HDF5Writer(h5py.File('output.h5', mode='w')))
    opt.run(5000)

This will transform the degrees of freedom (DOF's) of the system (cell vectors
and cartesian atomic coordinates) into a new set of DOF's (scaled cell vectors
and reduced coordinates) to allow an efficient optimization of both cell
parameters atomic positions. Several other dof_transform options are discussed
in XXX.


**TODO:**

#. Check if we can do something like the Andersen thermostat to simulate a
   constant pressure ensemble.

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
* ``step``: consider only a sample each ``step`` iterations.
* ``max_sample``: consider at most ``max_sample`` number of samples.

The last option is only possible when ``step`` is not specified and the total
number of samples (or ``end``) is known. The optimal value for ``step`` will be
derived from ``max_sample``. Some analysis may not have the max_sample argument,
e.g. the spectrum analysis, because the choice of the step size for such
analysis is a critical parameter that needs to be set carefully.


Basic analysis
--------------

A few basic analysis routines are provided to quickly check the sanity of an MD
simulation:

* ``plot_energies`` makes a plot of the kinetic and the total energy as function
  of time. For example::

    f = h5py.File('output.h5')
    plot_energies(f)

  makes a figure ``energies.png``.

* ``plot_temperate`` is similar, but plots the temperature as function of time.

* ``plot_temp_dist`` plots the distribution (both pdf and cdf) of the
  instantaneous atomic and system temperatures and compares these with the
  expected analytical result for a constant-temperature ensemble. For example:

    plot_temp_dist(f)

  makes a figure ``temp_dist.png``

All these functions accept optional arguments to tune their behavior. See XXX
for more details.


Advanced analysis
-----------------

Yaff also includes analysis tools that can extract relevant macroscopic
properties from a simulation. These analysis tools require some additional
computations that can either be done in a post-processing step, or on-line.

* A radial distribution function is computed as follows::

    indexes = system.get_indexes('O_W')
    f = h5py.File('output.h5')
    rdf = RDFAnalysis(f, indexes)
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

#. A tool to select certain atoms, e.g. based on type or more complex rules. The
   graph code in ``molmod`` is OK, but it takes too much typing to use it. The
   SMARTS system is very compact, but it has a few disadvantages that make it
   poorly applicable in the Yaff context: e.g. it assumes that the hybridization
   state of first-row atoms and bond orders are known. The only real `knowns` in
   the Yaff context are: ``numbers``, (optionally) ``ffatypes``, (optionally)
   ``fragments`` and bonds. It would be good to have a simplified SMARTS-like
   one-line syntax that only uses the four types of information above:

     a. The following rules are available to specify an atom. The specifiers are discussed below.

      * ``spec`` -- the atom must match specifier ``spec``
      * ``=N[%spec]`` -- the atom must be bonded to N atoms (that match specifier ``spec``)
      * ``>N[%spec]`` -- the atom must be bonded to more than N atoms (that match specifier ``spec``)
      * ``<N[%spec]`` -- the atom must be bonded to less N atoms (that match specifier ``spec``)
      * ``@N`` -- the atom must be part of a strong ring of size N

     b. The ``[fragment:]kind`` specifier, in short ``[F:]X``, is used to specify an
        atom in fragment F that has atomic number or atom type X.

     c. ``&`` (and), ``|`` (or) and ``!`` (not) operators to combine rules, in that
        order of precedence

     d. The compound specifiers: a specifier may also consist of a rule
        enclosed in curly brackets.

     e. parenthesis to modify operator precedence

   We should make a compiler that transforms these rules into a a function that
   returns ``True`` or ``False``, given a system object and an atom index. This
   implies a few things:

    a. We must first figure out how to represent fragments in the System class.
       The following things should be taken into account.
        1. An atom can be part of multiple fragments at the same time, i.e.
           fragments may overlap.
        2. It must be fast to determine of an atom is in a certain fragment
        3. It must be simple to dump and load fragment information into the
           array-based checkpoint format.
        4. Fragments must be implementable in the generators. These contain
           loops over relevant internal coordinates or atoms, and determine for
           each case the atom types involved to select the proper parameters.
           Just like an atom, an internal coordinate may be part of one or more
           fragments. All matching parameters must be translated into energy
           terms. This may be problematic for the non-bonding interactions,
           which typically have a single set of parameters per atom or per
           atom-pair. Adding two terms per atom pair is not possible. Hmm.
           Do we really need overlapping fragments?
    b. The ffatypes must become an optional argument of the ``System`` class.
    c. A ``System`` method must be provided to assign ffatypes based on a list
       of (ffatype, selector) pairs. These are trivial to load from a txt file.
       An error should be raised in this method, if there is an atom without
       a matching ffatype.
    d. The generators should raise an error if no ffatypes are present.
    e. Atom types should not contain the following symbols: ``:``, ``%``, ``=``,
       ``<``, ``>``, ``@``, ``(``, ``)``, ``&``, ``|``, ``!``, ``{``, ``}``, and
       should not start with a digit.
    f. The last four rules will only work of the system has a topology object.


   Some examples of atom selectors:

 * ``6`` -- any carbon atom
 * ``TPA:6`` -- a carbon atom in the TPA fragment
 * ``C3`` -- any atom with type C3
 * ``TPA:C3`` -- an atom with type C3 in the TPA fragment
 * ``!1`` -- anything that is not a hydrogen
 * ``C2|C3`` -- an atom of type C2 or C3
 * ``6|7&=1%1`` or ``(6|7)&=1%1`` -- a carbon or nitrogen bonded to exactly one hydrogen
 * ``>0%{6|=4}`` -- an atom bonded to at least one carbon atom or bonded to at least one atom with four bonds.
 * ``6&@6`` -- a Carbon atom that is part of a six-membered ring

#. Implement RDF. The RDF analysis must have a real-space cutoff that is smaller
   than the smallest spacing of the periodic cells.

#. Something to estimate diffusion constants.

#. Port other things from MD-Tracks, including the conversion stuff.
