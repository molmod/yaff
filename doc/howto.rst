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
-----------------------------

A ``System`` instance in Yaff contains all the physical properties of a
molecular system plus some extra information that is usfule to define a force
field. Most properties are optional.

**Basic physical properties:**

#. Atomic numbers
#. Positions of the atoms
#. 0, 1, 2, or 3 Cell parameters (optional)
#. Atomic charges (optional)
#. Atomic masses (optional)

**Basic auxiliary properties:** (useful to define FF)

#. The bonds between the atoms (in the form of a list of atom pairs, optional)
#. Scopes. This data is stored as an ordered list of unique scope names and a
   numpy array with a scope index for each atom. (Optional) Each atom can only
   be part of one scope.
#. Atom types. This data is stored as an ordered list of unique atom type names
   and a numpy array with an atom type index for each atom. (Optional) Each atom
   can only have one atom type.

Other properties such as valence angles, dihedral angles, assignment of energy
terms, exclusion rules, and so on, can be derived from these basic properties.

The positions and the cell parameters may change during the simulation. All
other properties (including the number of atoms and the number of cell vectors)
do not change during a simulation. If such changes seem to be necessary, one
should create a new System class instead.

A scope is a part of the system in which atom types and force field parameters
are consistent. The same atom types in different scopes may have a different
meaning and bonds between the same atom types from different scopes, may have
different parameters. For example, when simulating a mixture of water and
methanol, it makes sense to put all water molecules in the ``WATER`` scope and
all methanol molecules in the ``METHANOL`` scope. The ``WATER`` scope contains
only two atom types (``WATER:O``, ``WATER:H``), while the ``METHANOL`` scope may
contain four atom types (``METHANOL:C``, ``METHANOL:H_C``, ``METHANOL:O``,
``METHANOL:O_H``). It is OK to have the ``O`` atom type in both the ``WATER``
and ``METHANOL`` scopes.

The ``System`` constructor arguments can be specified with some python code::

    system = System(
        numbers=np.array([8, 1, 1]*2),
        pos=np.array([[-4.583, 5.333, 1.560], [-3.777, 5.331, 0.943]])*angstrom,
        scopes=['WAT']*6,
        ffatypes=['O', 'H', 'H']*2,
        bonds=np.array([[(i/3)*3,i] for i in xrange(6) if i%3!=0]),
        rvecs=np.array([[9.865, 0.0, 0.0], [0.0, 9.865, 0.0], [0.0, 0.0, 9.865]])*angstrom,
    )

where the ``*angstrom`` converts the numbers from angstrom to atomic units. The
scopes and atom types may be given as ordinary lists with a single string for
each atom. Such lists are converted automatically to a unique list of strings
and arrays with scope and atom type indexes for each atom. The following is
equivalent::

    system = System(
        numbers=np.array([8, 1, 1]*2),
        pos=np.array([[-4.583, 5.333, 1.560], [-3.777, 5.331, 0.943]])*angstrom,
        scopes=['WAT'],
        scope_ids=[0]*6
        ffatypes=['O', 'H'],
        ffatype_ids=[0, 1, 1]*2
        bonds=np.array([[(i/3)*3,i] for i in xrange(6) if i%3!=0]),
        rvecs=np.array([[9.865, 0.0, 0.0], [0.0, 9.865, 0.0], [0.0, 0.0, 9.865]])*angstrom,
    )

The latter constructor initializes the scope and atom type information in the
native form of the ``System`` class.

One can also load the system from one or more files::

    system = System.from_file('initial.xyz', 'topology.psf', cell=np.identity(3)*9.865*angstrom)

The ``from_file`` class method accepts one or more files and any constructor
argument from the System class. A system can be easily stored to a file using
the ``to_file`` method::

    system.to_file('last.chk')

where the ``.chk``-format is the standard text-based checkpoint file format in
Yaff. It can also be used in the ``from_file`` method.

**TODO:**

#. [LOW PRIORITY] Add possibility to read system from a HDF5 output file.

#. [LOW PRIORITY] Make the checkpoint format more compact.


Setting up an FF
----------------

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

#. [PARTIALLY DONE, TODO: TORSION, GRIMME] The generate method.

#. [LOW PRIORITY] Replace hammer by taper. Check which one converges the
   quickest as the real-space cutoff is increased, without compromising the
   conserved quantity.


Running an FF simulation
------------------------


**Molecular Dynacmis**

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


**Geometry optimization**

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
   constant pressure ensemble. (Is possible, see Andersen J. Chem. Phys. 1980,
   72, 2384-2393.)

#. [LOW PRIORITY] ``RefTraj`` derivative of the Iterative class.

#. [LATER] Optimizer stuff. We should use the molmod optimizer, but change it
   such that the main loop of the optimizer is done in Yaff instead of in
   molmod.

#. [LATER] Numerical (partial) Hessian


Analyzing the results
---------------------

The analysis of the results is (in the first place) based on the output
file ``output.h5``. On-line analysis (during the iterative algorithm, without
writing data to disk) is also possible.


**Slicing the data**

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


**Basic analysis**

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


**Advanced analysis**

Yaff also includes analysis tools that can extract relevant macroscopic
properties from a simulation. These analysis tools require some additional
computations that can either be done in a post-processing step, or on-line.

* A radial distribution function is computed as follows::

    f = h5py.File('output.h5')
    select = system.get_indexes('O')
    rdf = RDF(f, 4.8*angstrom, 0.1*angstrom, max_sample=100, select0=select)
    rdf.plot()
    rdf.plot_crdf()

  In this example, the cutoff for the RDF is 4.8 Å and the spacing of the
  bins 0.1 Å. At most 100 samples are used to compute the RDF. The results are
  included in the HDF5 file, and optionally plotted using matplotlib.
  Alternatively, the same ``RDFAnalysis`` class can be used for on-line
  analysis, without the need to store huge amounts of data on disk::

    select = system.get_indexes('O')
    rdf = RDF(None, 4.8*angstrom, 0.1*angstrom, max_sample=100, select0=select)
    nve = NVEIntegrator(ff, hooks=rdf, temp0=300)
    nve.run(5000)
    rdf.plot()
    rdf.plot_crdf()


**TODO:**

#. [LOW PRIORITY] Port other things from MD-Tracks, including the conversion stuff.


ATSELECT: Selecting atoms
=========================

In several parts of the introduction, one can provide a list of atom indexes to
limit an analysis or a hook to a subset of the complete system. To facilitate
the creation of these lists, yaff introduces an atom-selection language similar
to SMARTS patterns. This language can also be used to define atom types.

The SMARTS system has the advantage of being very compact, but it has a few
disadvantages that make it poorly applicable in the Yaff context: e.g. it
assumes that the hybridization state of first-row atoms and bond orders are
known. The only real `knowns` in the Yaff context are: ``numbers``, (optionally)
``ffatypes``, (optionally) ``scopes`` and (optionally) ``bonds``. Therefore
we introduce a new language, hereafter called `ATSELECT`, to select atoms in a
system.

The syntax of the ATSELECT language is defined as follows. An ATSELECT
expression consists of a single line and is case-sensitive. White-space is
completely ignored. An ATSELECT expression can be any of the following:

``[scope:]number``
    Matches an atom with the given number, optionally part of the given scope.

``[scope:]ffatype``
    Matches an atom with the given atop type, optionally part of the given scope.

``scope:*``
    Matches any atom in the given scope.

``expr1 & expr2 [& ...]``
    Matches an atom the satisfies all the given expressions.

``expr1 | expr2 [| ...]``
    Matches an atom the satisfies any of the given expressions.

``!expr``
    Matches an atom that does not satisfy the given expression.

``=N[%expr]``
    Matches an atom that has exactly N neighbors, that optionally match the
    given expression.

``>N[%expr]``
    Matches an atom that has more than N neighbors, that optionally match the
    given expression.

``<N[%expr]``
    Matches an atom that has less than N neighbors, that optionally match the
    given expression.

``@N``
    Matches an atom that is part of a strong ring with N atoms.

``(expr)``
    Round brackets are part of the syntax, used to override operator precedence.
    The precedence of the operators corresponds to the order of this list.

In the list above, ``expr`` can be any valid ATSELECT expression. Atom types and
scope names should not contain the following symbols: ``:``, ``%``, ``=``,
``<``, ``>``, ``@``, ``(``, ``)``, ``&``, ``|``, ``!``, and should
not start with a digit. Some examples of atom selectors:

 * ``6`` -- any carbon atom.
 * ``TPA:6`` -- a carbon atom in the TPA fragment.
 * ``C3`` -- any atom with type C3.
 * ``TPA:C3`` -- an atom with type C3 in the TPA fragment.
 * ``!1`` -- anything that is not a hydrogen.
 * ``C2|C3`` -- an atom of type C2 or C3.
 * ``6|7&=1%1`` or ``(6|7)&=1%1`` -- a carbon or nitrogen bonded to exactly one
   hydrogen.
 * ``>0%(6|=4)`` -- an atom bonded to at least one carbon atom or bonded to at
   least one atom with four bonds.
 * ``6&@6`` -- a Carbon atom that is part of a six-membered ring.

There are currently two ways to use the ATSELECT strings in Yaff:

1. Compile the string into a function and use it directly::

    from yaff import *
    fn = atsel_compile('C&=4')
    system = System.from_file('test.chk')
    if fn(systen, 0):
        pass
        # Do something if the first atom is a carbon with four neighbors.
        # ...

2. Get all atom indexes in a system that match a certain ATSELECT string::

    from yaff import *
    system = System.from_file('test.chk')
    indexes = system.get_indexes('C&=4')
    # The array indexes is now contains all indexes of the carbon atoms with
    # four neighbors.

Whenever one uses a compiled expression on a system that does not have
sufficient attributes, a ``ValueError`` is raised.

**TODO:**

#. [LATER] Add a method to the System class to assign ffatypes based on ATSELECT filters.
   If an atom does not have a matching filter, raise an error.

#. [LATER2] Add support for atomic numbers in the parameter files.

#. [LATER2] Make an FF for methanol, and a methanol-water system to facilitate the
   testing.

#. [LATER2] Add support for scopes to the Generator classes.

   The parameter file contains two sections::

     BEGIN SCOPE WATER
     ...
     END SCOPE

     BEGIN SCOPE METHANOL
     ...
     END SCOPE

   Each section has its own default scope, although it is OK to use other scopes
   too when defining the parameters. (Examples will be given below.)

#. [LATER2] Allow ``scope:ffatype`` and ``scope:number`` combinations in the parameter
   files.

#. [LOW PRIORITY] Add support for ``@N`` feature to ATSELECT.

#. [LOW PRIORITY] Add caching to the ATSELECT compiler.
