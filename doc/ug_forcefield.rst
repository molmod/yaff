Force field models
##################


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



An example for a reasonable non-polarizable water FF:

.. literalinclude:: ../input/parameters_water.txt
