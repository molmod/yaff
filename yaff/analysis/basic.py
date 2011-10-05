# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
# for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
# reserved unless otherwise stated.
#
# This file is part of YAFF.
#
# YAFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# YAFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


import h5py

import numpy as np

from molmod import boltzmann
from yaff.log import log


__all__ = ['plot_energies', 'plot_temp_dist']


def get_hdf5_file(fn_hdf5_traj):
    if isinstance(fn_hdf5_traj, h5py.File):
        f = fn_hdf5_traj
        do_close = False
    else:
        f = h5py.File(fn_hdf5_traj, mode='r')
        do_close = True
    return f, do_close


def get_slice(f, start, end, max_sample):
    nrow = f['trajectory'].attrs['row']
    if end == -1 or end < nrow:
        end = nrow
    if start < 0:
        start = 0
    if max_sample is None or max_sample > (end-start):
        step = 1
    else:
        step = (end-start)/max_sample
    return start, end, step


def plot_energies(fn_hdf5_traj, fn_png='energies.png', start=0, end=-1, max_sample=1000):
    """Make a plot of the potential and the total energy in the trajectory

       **Arguments:**

       fn_hdf5_traj
            The filename of the HDF5 file (or an h5py.File instance) containing
            the trajectory data.

       **Optional arguments:**

       fn_png
            The png file to write the figure to

       start
            The first sample to consider.

       end
            The last sample to consider + 1.

       max_sample
            The maximum number of data points to use for the plot. When set to
            None, all data from the trajectory is used. However, this is not
            recommended.

       The units for making the plot are taken from the yaff screen logger. This
       type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt
    f, do_close = get_hdf5_file(fn_hdf5_traj)
    start, end, step = get_slice(f, start, end, max_sample)

    ekin = f['trajectory/ekin'][start:end:step]/log.energy
    epot = f['trajectory/epot'][start:end:step]/log.energy
    time = f['trajectory/time'][start:end:step]/log.time

    pt.clf()
    pt.plot(time, epot, 'k-', label='E_pot')
    pt.plot(time, epot+ekin, 'r-', label='E_pot+E_kin')
    pt.xlim(time[0], time[-1])
    pt.xlabel('Time [%s]' % log.unitsys.time[1])
    pt.ylabel('Energy [%s]' % log.unitsys.energy[1])
    pt.legend(loc=0)
    pt.savefig(fn_png)

    if do_close:
        f.close()


def plot_temp_dist(fn_hdf5_traj, fn_png='temp_dist.png', start=0, end=-1, max_sample=50):
    """Plots the distribution of the weighted atomic velocities

       **Arguments:**

       fn_hdf5_traj
            The filename of the HDF5 file (or an h5py.File instance) containing
            the trajectory data.

       **Optional arguments:**

       fn_png
            The png file to write the figure to

       start
            The first sample to consider.

       end
            The last sample to consider + 1.

       max_sample
            The maximum number of data points to use for the plot. When set to
            None, all data from the trajectory is used. However, this is not
            recommended.

       This type of plot is essential for checking the sanity of a simulation.
       The empirical cumulative distribution is plotted and overlayed with the
       analytical cumulative distribution one would expect if the data were
       taken from an NVT ensemble.

       This type of plot reveals issues with parts that are relatively cold or
       warm compared to the total average temperature. This helps to determine
       (the lack of) thermal equilibrium.
    """
    # TODO: switch to temperatures.
    # TODO: add mean temperature to plot.
    # TODO: move away from cumulative
    import matplotlib.pyplot as pt
    f, do_close = get_hdf5_file(fn_hdf5_traj)
    start, end, step = get_slice(f, start, end, max_sample)

    # get the mean temperature
    temp_mean = f['trajectory/temp'][start:end:step].mean()
    wvel_sigma = np.sqrt(temp_mean*boltzmann)
    wvel_step = wvel_sigma/100
    wvel_grid = np.arange(-2*wvel_sigma, wvel_sigma*2 + 0.5*wvel_step, wvel_step)

    # build up the distribution
    counts = np.zeros(len(wvel_grid)-1, int)
    total = 0.0
    weights = np.sqrt(np.array(f['system/masses']).reshape(-1,1))
    for i in xrange(start, end, step):
        wvel = f['trajectory/vel'][i]*weights
        counts += np.histogram(wvel.ravel(), bins=wvel_grid)[0]
        total += wvel.size

    # transform into cumulative
    x = wvel_grid[:-1]+0.5*wvel_step
    emp = counts.cumsum()/total

    # the analytical form
    from scipy.special import erf
    ana = (1 + erf(x/wvel_sigma))/2

    # Make the plot of the cumulative histograms
    unit = np.sqrt(log.energy)
    pt.clf()
    pt.plot(x/unit, emp, 'k-', label='Empirical')
    pt.plot(x/unit, ana, 'r-', label='Analytical')
    pt.axvline(-wvel_sigma/unit, color='k', ls='--')
    pt.axvline(0, color='k', ls='-')
    pt.axvline(wvel_sigma/unit, color='k', ls='--')
    pt.xlim(x[0]/unit, x[-1]/unit)
    pt.ylim(0, 1)
    pt.ylabel('Cumulative probability')
    pt.xlabel('Velocity/sqrt(mass) [sqrt(%s)]' % log.unitsys.energy[1])
    pt.title('Distribution of weighted atomic velocities')
    pt.legend(loc=0)
    pt.savefig(fn_png)

    if do_close:
        f.close()
