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
    # control the amount of screen output
    log.set_level(log.medium)

    # 1) specify the system
    system = System.from_file('system.chk')
    # 2) specify the force field
    ff = ForceField.generate(system, 'parameters.txt')
    # 3) Integrate Newton's equation of motion and write the trajectory in HDF5 format.
    nve = NVEIntegrateor(ff, 'output.h5', temp_init=300)
    nve.run(5000)
    # 4) perform an analysis, e.g. RDF computation for O_W O_W centers.
    rdf = RDFAnalysis(system, 'O_W', 'O_W', 'output.h5')
    rdf.result.plot('rdf.png')

These steps will be discussed in more detail in the following sections.

Yaff internally works with atomic units, although other unit systems can be used
in input and (some) output files. Numpy and Cython are used extensively in Yaff
for numerical efficiency. The examples below often use Numpy too, assuming
the following import statement::

    import numpy as np

**TODO:**

1. Implement the log object. Usage should be as follows::

    if log.do_medium:
        log('This is a message printed at the medium log level and at higher log levels, e.g. \'high\' and \'debug\'.')

   It would be good to stick to standard terminal width (80 chars, automatic
   line wrapping) and fix some format conventions that make it easy to recognize
   which output comes from which part of the program. This could be done by
   reserving the first 8 characters for a location specifier, e.g. ::

    if log.do_medium:
        log('FOO', 'This is a message printed at the medium log level and at higher log levels, e.g. \'high\' and \'debug\'.')

   would result in::

    ____FOO This is a message printed at the medium log level and at higher log
    ____FOO levels, e.g. 'high' and 'debug'.

   The following levels are useful: silent, error, warning, low, medium, high,
   debug.

Setting up a molecular system
=============================

Besides the positions of the atoms (or nuclei or pseudo-atoms) and the periodic
boundary conditions, two additional pieces of information are needed to be able
to define a force field energy:

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

where the ``.chk``-format is the standard text-based checkpoint file format in
Yaff. It can also be used in the ``from_file`` method.

**TODO:**

1. Import units in the Yaff name space.

2. Implement ``to_file`` and ``from_file``

3. Introduce fragment name spaces in the atom types, e.g. instead of using O_W
   and H_W, we should have WATER:O, WATER:H. In the system object, we should
   have an extra fragment dictionary to know which atom is part
   of what sort of fragment, e.g. something like: ``system.fragments =
   {'WATER':, np.array([0, 1, 2, 3, 4, 5, ...])}``. The corresponding atom types
   can simply be ``system.ffatypes=['O', 'H', 'H, 'O', 'H', 'H, ...]``. It is OK
   that different atoms in different fragments have coinciding atom type names.
   This approach has the following advantages:

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

4. Lower priority: provide a simple tool to automatically assign bonds and atom
   types using rules. (For the moment we hack our way out with the ``molmod``
   package.)


Setting up an FF
================

Once the system is defined, one can continue with the specification of the force
field model. The simplest way to create a force-field as as follows::

    ff = ForceField.generate(system, 'parameters.txt')

where the file ``parameters.txt`` contains all force field parameters. See XXX
for more details on the format of the parameters file. Additional `technical`
parameters that determine the behavior of the force field, such as the
real-space cutoff, the verlet skin, and so on, may be specified as keyword
arguments in the ``generate`` method. See XXX for a detailed description of the
``generate`` method.

Once an ``ff`` object is created, it can be used to evaluate the energy (and
optionally the forces) for a given set of Cartesian coordinates::

    # change the atomic positions in the system object
    system.pos[:] = new_pos
    # compute the energy
    new_energy = ff.compute()

One may also allocate arrays to store the derivative of the energy towards
the atomic positions and the virial tensor::

    # allocate arrays for the Cartesian gradient of the energy and the virial
    # tensor.
    gpos = np.zeros(system.pos.shape, float)
    vtens = np.zeros((3,3), float)
    # change the atomic positions in the system object
    system.pos[:] = new_pos
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
    pot_pair_lj = PotPairLJ(sigmas, epsilons, rcut=15*angstrom, smooth=True)
    nlists = NeighborLists(system)
    scalings = Scalings(system.topology)
    part_part_lj = PartPair(system, nlists, scalings, pot_pair_lj)
    ff = ForceField(system, [part_part_lj], nlists)


**TODO:**

1. Change name conventions (``pair_part_*`` and ``pot_pair_*``) to make things
   easier to read. Create ``pair_*`` attributes automatically.

2. Document the format of ``parameters.txt``. This should be done very
   carefully. I'm currently thinking of something along the lines of CHARMM
   parameter file, but then with a few extra features to make the format more
   general:

    a. Introduce sections for different namespaces (see above)
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

   .. literalinclude:: parameters.txt

3. The generate method.

4. If the generate method is slow, we may need a checkpoint file for the
   ForceField class.


Running an FF simulation
========================

Given a ``ForceField`` instance, it is trivial to run several types of basic
simulations. For example, the equations of motion in the NVE ensemble can be
integrated as follows::

    nve = NVEIntegrator(ff, 'output.h5', temp_init=300)
    nve.run(5000)

The parameters of the integrator can be tuned with several optional arguments of
the ``NVEIntegrator`` constructor. See XXX for more details. Once the integrator
is created, the ``run`` method can be used to compute a given number of time
steps. The trajectory output is written to a HDF5 file. The exact contents of
the HDF5 file depends on the integrator used and the optional arguments. All
data is stored in atomic units.

Other integrators are implemented such as NVTNoseIntegrator,
NVTAndersenIntegrator and NVTLangevinIntegrator. One may also use a geometry
optimizer instead of an integrator::

    opt = BFGSOptimizer(ff, 'output.h5')
    opt.run(5000)

Again, convergence criteria are controlled through optional arguments of the
constructor. the ``run`` method has the maximum number of iterations as the only
argument. I would implement the Hessian computation with finite differences also
at this level, mainly because there are different ways of doing this, e.g. using
different constraints etc.

**TODO:** implement or transform existing code


Analyzing the results
=====================

The analysis of the results is (in the first place) based on the output
file ``output.h5``. Several analysis routines are implemented, e.g. for the
computation of an RDF, the following can be used::

    rdf = RDFAnalysis(system, 'WATER:O', 'WATER:O', 'output.h5')
    rdf.result.plot('rdf.png')

The results are included in the HDF5 file, and optionally plotted using
matplotlib. Alternatively, we want the same code to be usable for on-line
analysis, without the need to store huge amounts of data on disk::

    rdf = RDFAnalysis(system, 'WATER:O', 'WATER:O', 'analysis.h5')
    nve = NVEIntegrator(ff, None, temp_init=300, analysis=rdf)
    nve.run(5000)
    rdf.result.plot('rdf.png')

The analysis keyword must obviously also accept a list of analysis objects.

In the former case, the ``RDFAnalsysis`` class will detect that the file
'output.h5' already contains a trajectory and hence immediately performs the
analysis. In the latter case, the file ``analysis.h5`` must be a new file or at
least it may not contain a trajectory. For an on-line analysis, the integrator
class will make the necessary calls to the analysis object.


**TODO:**

1. The main idea is to port MD-tracks to a new HDF5-based analysis system.
