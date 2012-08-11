Sampling the phase space
########################




**Molecular Dynacmis**

The equations of motion in the NVE ensemble can be integrated as follows::

    hdf5_writer = HDF5Writer(h5py.File('output.h5', mode='w'))
    nve = NVEIntegrator(ff, 1*femtosecond, hooks=hdf5_writer, temp0=300)
    nve.run(5000)

The parameters of the integrator can be tuned with several optional arguments of
the ``NVEIntegrator`` constructor. See XXX for more details. Once the integrator
is created, the ``run`` method can be used to compute a given number of time
steps. The trajectory output is written to a HDF5 file. The exact contents of
the HDF5 file depends on the integrator used and the optional arguments. All
data in the HDF5 file is stored in atomic units.

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
