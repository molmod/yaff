Overview of YAFF
################

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
    import h5py

    # 1) specify the system
    system = System.from_file('system.chk')
    # 2) specify the force field
    ff = ForceField.generate(system, 'parameters.txt')
    # 3) Integrate Newton's equation of motion and write the trajectory in HDF5 format.
    f = h5py.File('output.h5', mode='w')
    hdf5_writer = HDF5Writer(f)
    nve = NVEIntegrator(ff, 1*femtosecond, hooks=hdf5_writer, temp0=300)
    nve.run(5000)
    # 4) perform an analysis, e.g. RDF computation for O_W O_W centers.
    indexes = system.get_indexes('O_W')
    rdf = RDFAnalysis(f, indexes)
    rdf.result.plot('rdf.png')
    f.close()

These steps will be discussed in more detail in the following sections.

Yaff internally works with atomic units, although other unit systems can be used
in input and (some) output files. The units used for the screen output are
controlled with the ``log.set_unitsys`` method. Output written in (binary) HDF5
files will always be in atomic units. When output is written to a format from
other projects/programs, the units of that program/project will be used.

Numpy, Cython and h5py are used extensively in Yaff for numerical efficiency.
The examples below often use Numpy too, assuming the following import
statement::

    import numpy as np, h5py
