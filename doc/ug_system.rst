Representation of a molecular system
####################################


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
