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

from yaff.log import log


__all__ = ['plot_energies']


def plot_energies(fn_hdf5_traj, fn_png, max_data=1000):
    """Make a plot of the potential and the total energy in the trajectory

       **Arguments:**

       fn_hdf5_traj
            The filename of the HDF5 file (or an h5py.File instance) containing
            the trajectory data.

       fn_png
            The png file to write the figure to

       **Optional arguments:**

       max_data
            The maximum number of datapoints to use for the plot. When set to
            None, all data from the trajectory is used. However, this is not
            recommended.

       The units for making the plot are taken from the yaff screen logger. This
       type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt

    if isinstance(fn_hdf5_traj, h5py.File):
        f = fn_hdf5_traj
    else:
        f = h5py.File(fn_hdf5_traj, mode='r')

    if max_data is None or max_data > f['trajectory'].attrs['row']:
        step = 1
    else:
        step = f['trajectory'].attrs['row']/max_data
    ekin = f['trajectory/ekin'][::step]/log.energy
    epot = f['trajectory/epot'][::step]/log.energy
    time = f['trajectory/time'][::step]/log.time

    pt.clf()
    pt.plot(time, epot, 'k-', label='E_pot')
    pt.plot(time, epot+ekin, 'r-', label='E_pot+E_kin')
    pt.savefig(fn_png)
