# YAFF is yet another force-field code
# Copyright (C) 2008 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
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
from yaff.analysis.utils import get_slice


__all__ = [
    'plot_energies', 'plot_temperature', 'plot_temp_dist', 'plot_density',
    'plot_cell_pars', 'plot_epot_contribs',
]


def get_time(f, start, end, step):
    if 'trajectory/time' in f:
        label = 'Time [%s]' % log.time.notation
        time = f['trajectory/time'][start:end:step]/log.time.conversion
    else:
        label = 'Step'
        time = f['trajectory/counter'][start:end:step]
    return time, label


def plot_energies(f, fn_png='energies.png', **kwargs):
    """Make a plot of the potential, total and conserved energy as f. of time

       **Arguments:**

       f
            An h5py.File instance containing the trajectory data.

       **Optional arguments:**

       fn_png
            The png file to write the figure to

       The optional arguments of the ``get_slice`` function are also accepted in
       the form of keyword arguments.

       The units for making the plot are taken from the yaff screen logger. This
       type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt
    start, end, step = get_slice(f, **kwargs)

    epot = f['trajectory/epot'][start:end:step]/log.energy.conversion
    time, tlabel = get_time(f, start, end, step)

    pt.clf()
    pt.plot(time, epot, 'k-', label='E_pot')
    if 'trajectory/etot' in f:
        etot = f['trajectory/etot'][start:end:step]/log.energy.conversion
        pt.plot(time, etot, 'r-', label='E_tot')
    if 'trajectory/econs' in f:
        econs = f['trajectory/econs'][start:end:step]/log.energy.conversion
        pt.plot(time, econs, 'g-', label='E_cons')
    pt.xlim(time[0], time[-1])
    pt.xlabel(tlabel)
    pt.ylabel('Energy [%s]' % log.energy.notation)
    pt.legend(loc=0)
    pt.savefig(fn_png)


def plot_temperature(f, fn_png='temperature.png', **kwargs):
    """Make a plot of the temperature as function of time

       **Arguments:**

       f
            An h5py.File instance containing the trajectory data.

       **Optional arguments:**

       fn_png
            The png file to write the figure to

       The optional arguments of the ``get_slice`` function are also accepted in
       the form of keyword arguments.

       The units for making the plot are taken from the yaff screen logger. This
       type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt
    start, end, step = get_slice(f, **kwargs)

    temp = f['trajectory/temp'][start:end:step]
    time, tlabel = get_time(f, start, end, step)

    pt.clf()
    pt.plot(time, temp, 'k-')
    pt.xlim(time[0], time[-1])
    pt.xlabel(tlabel)
    pt.ylabel('Temperature [K]')
    pt.savefig(fn_png)


def plot_temp_dist(f, fn_png='temp_dist.png', select=None, **kwargs):
    """Plots the distribution of the weighted atomic velocities

       **Arguments:**

       f
            An h5py.File instance containing the trajectory data.

       **Optional arguments:**

       fn_png
            The png file to write the figure to

       select
            A list of atom indexes that should be considered for the analysis.
            By default, information from all atoms is combined.

       start, end, step, max_sample
           The optional arguments of the ``get_slice`` function are also
           accepted in the form of keyword arguments.

       This type of plot is essential for checking the sanity of a simulation.
       The empirical cumulative distribution is plotted and overlayed with the
       analytical cumulative distribution one would expect if the data were
       taken from an NVT ensemble.

       This type of plot reveals issues with parts that are relatively cold or
       warm compared to the total average temperature. This helps to determine
       (the lack of) thermal equilibrium.
    """
    import matplotlib.pyplot as pt
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import chi2
    start, end, step = get_slice(f, **kwargs)

    # TODO: plot (and compute) the analytical results with steps, just like the
    # empirical data.

    # Make an array with the weights used to compute the temperature
    if select is None:
        weights = np.array(f['system/masses'])/boltzmann
    else:
        weights = np.array(f['system/masses'])[select]/boltzmann

    if select is None:
        # just load the temperatures from the output file
        temps = f['trajectory/temp'][start:end:step]
    else:
        # compute the temperatures of the subsystem
        temps = []
        for i in xrange(start, end, step):
             temp = ((f['trajectory/vel'][i,select]**2).mean(axis=1)*weights).mean()
             temps.append(temp)
        temps = np.array(temps)

    temp_mean = temps.mean()

    # A) ATOMS
    ndof = 3
    temp_step = temp_mean/10

    # setup the temperature grid
    temp_grid = np.arange(0, temp_mean*5 + 0.5*temp_step, temp_step)

    # build up the distribution for the atoms
    counts = np.zeros(len(temp_grid)-1, int)
    total = 0.0
    for i in xrange(start, end, step):
        if select is None:
            atom_temps = (f['trajectory/vel'][i]**2).mean(axis=1)*weights
        else:
            atom_temps = (f['trajectory/vel'][i,select]**2).mean(axis=1)*weights
        counts += np.histogram(atom_temps.ravel(), bins=temp_grid)[0]
        total += atom_temps.size


    # transform into empirical pdf and cdf
    emp_atom_pdf = counts/total/temp_step
    emp_atom_cdf = counts.cumsum()/total

    # the analytical form
    rv = chi2(ndof, 0, temp_mean/ndof)
    x_atom = temp_grid[:-1]+0.5*temp_step
    ana_atom_pdf = rv.pdf(x_atom)
    ana_atom_cdf = rv.cdf(x_atom)


    # B) SYSTEM
    if select is None:
        ndof = 3*f['system/numbers'].shape[0]
    else:
        ndof = 3*len(select)
    sigma = temp_mean*np.sqrt(2.0/ndof)
    temp_step = sigma/5

    # setup the temperature grid and make the histogram
    temp_grid = np.arange(temp_mean-3*sigma, temp_mean+3*sigma, temp_step)
    counts = np.histogram(temps.ravel(), bins=temp_grid)[0]
    total = float(len(temps))

    # transform into empirical pdf and cdf
    emp_sys_pdf = counts/total/temp_step
    emp_sys_cdf = counts.cumsum()/total

    # the analytical form
    rv = chi2(ndof, 0, temp_mean/ndof)
    x_sys = temp_grid[:-1]+0.5*temp_step
    ana_sys_pdf = rv.pdf(x_sys)
    ana_sys_cdf = rv.cdf(x_sys)


    # C) Make the plots
    pt.clf()
    xconv = log.temperature.conversion
    pt.suptitle('Mean T=%.0f [%s]' % (temp_mean, log.temperature.notation))

    pt.subplot(2,2,1)
    pt.title('Atom (k=3)')
    scale = 1/emp_atom_pdf.max()
    pt.plot(x_atom/xconv, emp_atom_pdf*scale, 'k-', drawstyle='steps-mid')
    pt.plot(x_atom/xconv, ana_atom_pdf*scale, 'r-')
    pt.axvline(temp_mean, color='k', ls='--')
    pt.xlim(x_atom[0]/xconv, x_atom[-1]/xconv)
    pt.ylim(ymin=0)
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    pt.ylabel('Recaled PDF')

    pt.subplot(2,2,2)
    pt.title('System (k=%i)' % ndof)
    scale = 1/emp_sys_pdf.max()
    pt.plot(x_sys/xconv, emp_sys_pdf*scale, 'k-', drawstyle='steps-mid')
    pt.plot(x_sys/xconv, ana_sys_pdf*scale, 'r-')
    pt.axvline(temp_mean, color='k', ls='--')
    pt.ylim(ymin=0)
    pt.xlim(x_sys[0]/xconv, x_sys[-1]/xconv)
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))

    pt.subplot(2,2,3)
    pt.plot(x_atom/xconv, emp_atom_cdf, 'k-', drawstyle='steps-mid')
    pt.plot(x_atom/xconv, ana_atom_cdf, 'r-')
    pt.axvline(temp_mean, color='k', ls='--')
    pt.xlim(x_atom[0]/xconv, x_atom[-1]/xconv)
    pt.ylim(0, 1)
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    pt.ylabel('CDF')
    pt.xlabel('Temperature [%s]' % log.temperature.notation)

    pt.subplot(2,2,4)
    pt.plot(x_sys/xconv, emp_sys_cdf, 'k-', drawstyle='steps-mid')
    pt.plot(x_sys/xconv, ana_sys_cdf, 'r-')
    pt.axvline(temp_mean, color='k', ls='--')
    pt.xlim(x_sys[0]/xconv, x_sys[-1]/xconv)
    pt.ylim(0, 1)
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    pt.xlabel('Temperature [%s]' % log.temperature.notation)

    pt.savefig(fn_png)


def plot_density(f, fn_png='density.png', **kwargs):
    """Make a plot of the mass density as function of time

       **Arguments:**

       f
            An h5py.File instance containing the trajectory data.

       **Optional arguments:**

       fn_png
            The png file to write the figure to

       The optional arguments of the ``get_slice`` function are also accepted in
       the form of keyword arguments.

       The units for making the plot are taken from the yaff screen logger.
    """
    import matplotlib.pyplot as pt
    start, end, step = get_slice(f, **kwargs)

    mass = f['system/masses'][:].sum()
    vol = f['trajectory/volume'][start:end:step]
    rho = mass/vol/log.density.conversion
    time, tlabel = get_time(f, start, end, step)

    pt.clf()
    pt.plot(time, rho, 'k-')
    pt.xlim(time[0], time[-1])
    pt.xlabel(tlabel)
    pt.ylabel('Density [%s]' % log.density.notation)
    pt.savefig(fn_png)


def plot_cell_pars(f, fn_png='cell_pars.png', **kwargs):
    """Make a plot of the cell parameters as function of time

       **Arguments:**

       f
            An h5py.File instance containing the trajectory data.

       **Optional arguments:**

       fn_png
            The png file to write the figure to

       The optional arguments of the ``get_slice`` function are also accepted in
       the form of keyword arguments.

       The units for making the plot are taken from the yaff screen logger.
    """
    import matplotlib.pyplot as pt
    start, end, step = get_slice(f, **kwargs)

    cell = f['trajectory/cell'][start:end:step]/log.length.conversion
    lengths = np.sqrt((cell**2).sum(axis=2))
    time, tlabel = get_time(f, start, end, step)
    nvec = lengths.shape[1]

    def get_angle(i0, i1):
        return np.arccos(np.clip((cell[:,i0]*cell[:,i1]).sum(axis=1)/lengths[:,i0]/lengths[:,i1], -1,1))/log.angle.conversion

    pt.clf()

    if nvec == 3:
        pt.subplot(2,1,1)
        pt.plot(time, lengths[:,0], 'r-', label='a')
        pt.plot(time, lengths[:,1], 'g-', label='b')
        pt.plot(time, lengths[:,2], 'b-', label='c')
        pt.xlim(time[0], time[-1])
        pt.ylabel('Lengths [%s]' % log.length.notation)
        pt.legend(loc=0)

        alpha = get_angle(1, 2)
        beta = get_angle(2, 0)
        gamma = get_angle(0, 1)
        pt.subplot(2, 1, 2)
        pt.plot(time, alpha, 'r-', label='alpha')
        pt.plot(time, beta, 'g-', label='beta')
        pt.plot(time, gamma, 'b-', label='gamma')
        pt.xlim(time[0], time[-1])
        pt.xlabel(tlabel)
        pt.ylabel('Angles [%s]' % log.angle.notation)
        pt.legend(loc=0)
    elif nvec == 2:
        pt.subplot(2,1,1)
        pt.plot(time, lengths[:,0], 'r-', label='a')
        pt.plot(time, lengths[:,1], 'g-', label='b')
        pt.xlim(time[0], time[-1])
        pt.ylabel('Lengths [%s]' % log.length.notation)
        pt.legend(loc=0)

        gamma = get_angle(0, 1)
        pt.subplot(2, 1, 2)
        pt.plot(time, gamma, 'b-', label='gamma')
        pt.xlim(time[0], time[-1])
        pt.xlabel(tlabel)
        pt.ylabel('Angle [%s]' % log.angle.notation)
        pt.legend(loc=0)
    elif nvec == 1:
        pt.plot(time, lengths[:,0], 'k-')
        pt.xlim(time[0], time[-1])
        pt.xlabel(tlabel)
        pt.ylabel('Lengths [%s]' % log.length.notation)
    else:
        raise ValueError('Can not plot cell parameters if the system is not periodic.')

    pt.savefig(fn_png)


def plot_epot_contribs(f, fn_png='epot_contribs.png', size=1.0, **kwargs):
    """Make a plot of the contributions to the potential energy as f. of time

       **Arguments:**

       f
            An h5py.File instance containing the trajectory data.

       **Optional arguments:**

       fn_png
            The png file to write the figure to

       The optional arguments of the ``get_slice`` function are also accepted in
       the form of keyword arguments.

       The units for making the plot are taken from the yaff screen logger. This
       type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt
    start, end, step = get_slice(f, **kwargs)

    epot = f['trajectory/epot'][start:end:step].copy()/log.energy.conversion
    epot -= epot[0]
    epot_contribs = []
    for i, name in enumerate(f['trajectory'].attrs['epot_contrib_names']):
        contrib = f['trajectory']['epot_contribs'][start:end:step,i].copy()/log.energy.conversion
        contrib -= contrib[0]
        epot_contribs.append((name, contrib))
    time, tlabel = get_time(f, start, end, step)

    pt.clf()
    pt.plot(time, epot, 'k-', label='_nolegend_', lw=2)
    for name, epot_contrib in epot_contribs:
        pt.plot(time, epot_contrib, label=name)
    pt.xlim(time[0], time[-1])
    pt.xlabel(tlabel)
    pt.ylabel('Energy [%s]' % log.energy.notation)
    pt.legend(loc=0)
    F=pt.gcf()
    DefaultSize = F.get_size_inches()
    F.set_size_inches(DefaultSize[0]*size, DefaultSize[1]*size)
    pt.savefig(fn_png)
