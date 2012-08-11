Trajectory Analysis
###################

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
    f.close()

  makes a figure ``energies.png``.

* ``plot_temperate`` is similar, but plots the temperature as function of time.

* ``plot_temp_dist`` plots the distribution (both pdf and cdf) of the
  instantaneous atomic and system temperatures and compares these with the
  expected analytical result for a constant-temperature ensemble. For example:

    plot_temp_dist(f)

  makes a figure ``temp_dist.png``

All these functions accept optional arguments to tune their behavior. See XXX
for more analysis routines and more details.


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
    f.close()

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

* A vibrational spectrum can be computed as follows::

    spectrum = Spectrum(f, bsize=512)
    spectrum.plot()
    spectrum.plot_ac()

  The ``bsize`` argument determines the size of the blocks used for the
  spectral analysis. The trajectory is cut into blocks of the given size. For
  each block, the spectrum is computed, and then averaged over all blocks. The
  ``plot`` method makes a figure of the spectrum. The ``plot_ac`` method makes
  a figure of the corresponding autocorrelation function. All the results are
  also available as attributes of the spectrum object. Similar to the RDF
  analysis, the spectrum can be computed both on-line and off-line. One can
  also estimate the IR spectrum as follows::

    spectrum = Spectrum(f, bsize=512, path='trajectory/dipole_vel', key='ir')
    spectrum.plot()

* The diffusion constant is computed as follows::

    diff = Diffusion(f, step=10, mult=5, select=select0)
    diff.plot()





Post-processing external trajectory data
========================================

One may also use the analysis module of Yaff to process trajectories generated
with other molecular simulation codes. This typically takes the following three
steps. These steps may be put in a single script, but in practice it is
recommended to have a separate script for the actual analysis.

1. Create a Yaff system object of the molecular system of interest. The
   following example loads the XYZ file of an initial geometry and adds cell
   vectors corresponding to a cubic cell with edge length 20.3 Å. ::

    from yaff import *
    import numpy as np
    system = System.from_file('initial.xyz', rvecs=np.diag([20.3, 20.3, 20.3])*angstrom)

2. Initialize an HDF5 file and load the trajectory in the HDF5 file::

    import h5py
    f = h5py.File('trajectory.h5', mode='w')
    system.to_hdf5(f)
    xyz_to_hdf5(f, 'trajectory.xyz')
    f.close()

3. Perform the actual analysis. In the following example, a radial distribution
   function is computed between the hydrogen and the oxygen atoms. ::

    select0 = system.get_indexes('1')
    select1 = system.get_indexes('8')
    rdf = RDF(10*angstrom, 0.1*angstrom, f, max_sample=100, select0=select0, select1=select1)
    rdf.plot()
