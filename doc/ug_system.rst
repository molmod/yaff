.. _ug_system:

Representation of a molecular system
####################################


Mote: the reference documentation (based on the source code) of the System class
can be found here: :mod:`yaff.system`.


Introduction
============


A ``System`` instance in Yaff contains all the physical properties of a
molecular system plus some extra information that is useful to define a force
field. Most properties are optional.

**Basic physical properties:**

#. Atomic numbers
#. Positions of the atoms
#. 0, 1, 2, or 3 Cell vectors (optional)
#. Atomic charges (optional)
#. Atomic masses (optional)

**Basic auxiliary properties:** (useful to define FF)

#. Bonds (optional). The bonds are representated as an array with two columns.
   Each row contains the atom indexes of a pair of bonded atoms.
#. Atom types (optional). Each atom can be given a specific atom type. Atoms of the
   same type should also be the same elements, but two atoms with the same
   atomic number do not need to be of the same type. This data is stored as an
   ordered list of unique atom type names and a numpy array with an atom type
   index for each atom. The atom type index refers to an item in the list of
   unique atom types.
#. Scopes (optional). Each atom can be part of one scope.  A scope
   is a part of the system for which a certain force field must be used.
   This information will be used in future versions of
   Yaff to combine different force fields in a single simulation. It is similar
   to the treatment of residues in biochemical force fields. This data is stored
   as an ordered list of unique scope names and a numpy array with a scope index
   for each atom. The scope index refers to one of the unique scope names.

Other properties such as valence angles, dihedral angles, assignment of energy
terms, exclusion rules, and so on, can be derived from these basic properties.

The positions and the cell parameters may change during the simulation. All
other properties (including the number of atoms and the number of cell vectors)
do not change during a simulation. If such changes seem to be necessary, one
should create a new System instance instead of modifying an existing one.

..  A scope is a part of the system in which atom types and force field parameters
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
        pos=np.array([[-4.583, 5.333, 1.560], [-3.777, 5.331, 0.943],
                      [-5.081, 4.589, 1.176], [-0.083, 4.218, 0.070],
                      [-0.431, 3.397, 0.609], [0.377, 3.756, -0.688]])*angstrom,
        scopes=['WAT']*6,
        ffatypes=['O', 'H', 'H']*2,
        bonds=np.array([[(i/3)*3,i] for i in xrange(6) if i%3!=0]),
        rvecs=np.array([[9.865, 0.0, 0.0], [0.0, 9.865, 0.0], [0.0, 0.0, 9.865]])*angstrom,
    )

where the ``*angstrom`` converts the numbers from angstrom to atomic units. The
scopes and atom types may be given as ordinary lists with a single string for
each atom. Such lists are converted automatically to a list of unique strings
and arrays with scope and atom type indexes for each atom. The following is
equivalent to the previous example::

    system = System(
        numbers=np.array([8, 1, 1]*2),
        pos=np.array([[-4.583, 5.333, 1.560], [-3.777, 5.331, 0.943],
                      [-5.081, 4.589, 1.176], [-0.083, 4.218, 0.070],
                      [-0.431, 3.397, 0.609], [0.377, 3.756, -0.688]])*angstrom,
        scopes=['WAT'],
        scope_ids=[0]*6
        ffatypes=['O', 'H'],
        ffatype_ids=[0, 1, 1]*2
        bonds=np.array([[(i/3)*3,i] for i in xrange(6) if i%3!=0]),
        rvecs=np.array([[9.865, 0.0, 0.0], [0.0, 9.865, 0.0], [0.0, 0.0, 9.865]])*angstrom,
    )

The second example initializes the scope and atom type information in the
native form of the ``System`` class.

One can also load the system from one or more files::

    system = System.from_file('initial.xyz', cell=np.identity(3)*9.865*angstrom)

The ``from_file`` class method accepts one or more files and any constructor
argument from the System class as keyword arguments. A system can be easily
stored to a file using the ``to_file`` method::

    system.to_file('last.chk')

where the ``.chk``-format is the standard text-based checkpoint file format in
Yaff. It can also be used in the ``from_file`` method.


Working with the ``System`` class
=================================

For production runs, we recommend that one writes a separate script to prepare
a systems instance, which is then written to the ``.chk`` format for later use
in scripts that perform the actual simulation and/or analysis. The example
below shows how this can be done, starting from a simple ``.xyz`` file with
coordinates for a water box with 32 molecules.

The first step is to load the ``.xyz`` file and add some extra information, cell
parameters in this example, through keyword arguments. ::

    sys = System.from_file('waterbox.xyz', cell=np.identity(3)*9.865*angstrom)

In order to run a force field simulation, one has to identify covalent bonds in
the system. We could have added these via keyword arguments of the ``from_file``
method. In this example, the :meth:`yaff.system.System.detect_bonds` method is
used::

    sys.detect_bonds()
    print 'The number of bonds:', len(sys.bonds)
    print sys.bonds

For the analysis of some simulations on crystals, it may be useful to align
the unit cell vectors with the Cartesian frame. This can be
done with the :meth:`yaff.system.System.align_cell` method. The following
will allign the 110 vector with the x-axis and the 001 vector with the z-axis::

    sys.align_cell(np.array([[1,1,0], [0,0,1]]))

On several occasions, it is also useful to construct a supercell::

    sys2 = sys.supercell(np.array([3,3,3]))

For most force fields, one has to define atom types. This can be done on the
basis of ATSELECT rules. (See :ref:`ug_sec_atselect` for details.) The following
will assign ``O_W`` and ``H_W`` to oxygen and hydrogen atoms, respectively::

    sys2.detect_ffatypes([('O_W', '8'), ('H_W', '1')])

The first string in each tuple is an `ffatype` string. The second string is an
ATSELECT rule. In this case, the rules only inspect the atomic number, but more
complicated rules are possible that also take into account the chemical
environment of the atom.

Although one can assign arbitrary masses to each atom, one is typically interested
in assigning standard atomic weights. This is done as follows::

    sys2.set_standard_masses()

When the system is finally ready to be used as a starting point for a Yaff
simulation, it is convenient to write it as a ``.chk`` file that can be easily
loaded in subsequent scripts::

    sys2.to_file('waterbox333.chk')

It is instructive to open this ``.chk`` file with a text editor. One will see
that all attributes of the system class are present in this file.
