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


__all__ = ['plot_energies', 'plot_temperature', 'plot_temp_dist']


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
    """Make a plot of the potential and the total energy as function of time

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


def plot_temperature(fn_hdf5_traj, fn_png='temperature.png', start=0, end=-1, max_sample=1000):
    """Make a plot of the temperature as function of time

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

    temp = f['trajectory/temp'][start:end:step]
    time = f['trajectory/time'][start:end:step]/log.time

    pt.clf()
    pt.plot(time, temp, 'k-')
    pt.xlim(time[0], time[-1])
    pt.xlabel('Time [%s]' % log.unitsys.time[1])
    pt.ylabel('Temperature [K]')
    pt.savefig(fn_png)

    if do_close:
        f.close()


def plot_temp_dist(fn_hdf5_traj, fn_png='temp_dist.png', start=0, end=-1, max_sample=200):
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
    import matplotlib.pyplot as pt
    from scipy.special import gammainc, erf
    f, do_close = get_hdf5_file(fn_hdf5_traj)
    start, end, step = get_slice(f, start, end, max_sample)
    temps = f['trajectory/temp'][start:end:step]
    temp_mean = temps.mean()

    # A) ATOMS

    # setup the temperature grid
    temp_step = temp_mean/10
    temp_grid = np.arange(0, temp_mean*5 + 0.5*temp_step, temp_step)

    # build up the distribution for the atoms
    counts = np.zeros(len(temp_grid)-1, int)
    total = 0.0
    weights = np.array(f['system/masses'])/boltzmann
    for i in xrange(start, end, step):
        atom_temps = (f['trajectory/vel'][i]**2).sum(axis=1)*weights
        counts += np.histogram(atom_temps.ravel(), bins=temp_grid)[0]
        total += atom_temps.size

    # transform into empirical pdf and cdf
    x_atom = temp_grid[:-1]+0.5*temp_step
    emp_atom_pdf = counts/total/temp_step
    emp_atom_cdf = counts.cumsum()/total

    # the analytical form
    ana_atom_pdf = np.sqrt(x_atom/temp_mean)*np.exp(-0.5*(x_atom/temp_mean))/np.sqrt(2*np.pi)/temp_mean
    ana_atom_cdf = gammainc(1.5, x_atom/(temp_mean*2))

    # B) SYSTEM

    # setup the temperature grid and make the histogram
    temp_step = temp_mean/25
    temp_grid = np.arange(0, temp_mean*2 + 0.5*temp_step, temp_step)
    counts = np.histogram(temps.ravel(), bins=temp_grid)[0]
    total = float(len(temps))

    # transform into empirical pdf and cdf
    x_sys = temp_grid[:-1]+0.5*temp_step
    emp_sys_pdf = counts/total/temp_step
    emp_sys_cdf = counts.cumsum()/total

    # the analytical form
    ndof = 3*f['system/numbers'].shape[0]
    sigma = temp_mean/np.sqrt(ndof)
    ana_sys_pdf = np.exp(-((x_sys-temp_mean)/sigma)**2)/np.sqrt(2*np.pi)/sigma
    ana_sys_cdf = (erf((x_sys - temp_mean)/sigma)+1)/2


    # C) Make the plots
    pt.clf()

    pt.subplot(2,2,1)
    pt.title('Atom (mean T=%.0fK)' % temp_mean)
    scale = 1/emp_atom_pdf.max()
    pt.plot(x_atom, emp_atom_pdf*scale, 'k-', drawstyle='steps-mid')
    pt.plot(x_atom, ana_atom_pdf*scale, 'r-')
    pt.axvline(temp_mean, color='k', ls='--')
    pt.xlim(x_atom[0], x_atom[-1])
    pt.ylabel('Recaled PDF')
    #pt.legend(loc=0)

    pt.subplot(2,2,2)
    pt.title('System')
    scale = 1/emp_sys_pdf.max()
    pt.plot(x_sys, emp_sys_pdf*scale, 'k-', drawstyle='steps-mid')
    pt.plot(x_sys, ana_sys_pdf*scale, 'r-')
    pt.axvline(temp_mean, color='k', ls='--')
    pt.xlim(x_sys[0], x_sys[-1])

    pt.subplot(2,2,3)
    pt.plot(x_atom, emp_atom_cdf, 'k-', drawstyle='steps-mid')
    pt.plot(x_atom, ana_atom_cdf, 'r-')
    pt.axvline(temp_mean, color='k', ls='--')
    pt.xlim(x_atom[0], x_atom[-1])
    pt.ylim(0, 1)
    pt.ylabel('CDF')
    pt.xlabel('Temperature [K]')

    pt.subplot(2,2,4)
    pt.plot(x_sys, emp_sys_cdf, 'k-', drawstyle='steps-mid')
    pt.plot(x_sys, ana_sys_cdf, 'r-')
    pt.axvline(temp_mean, color='k', ls='--')
    pt.xlim(x_sys[0], x_sys[-1])
    pt.ylim(0, 1)
    pt.xlabel('Temperature [K]')

    pt.savefig(fn_png)

    if do_close:
        f.close()
