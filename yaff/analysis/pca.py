# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
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
"""Toolkit for Principal Component Analysis (PCA)"""


from __future__ import division

import h5py as h5
import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as pt
from scipy import random

from molmod.units import *
from molmod.constants import boltzmann
from molmod.io.xyz import XYZWriter
from molmod.periodic import periodic as pd

from yaff.log import log

__all__ = [
    'calc_cov_mat_internal', 'calc_cov_mat', 'calc_pca', 'pca_projection',
    'write_principal_mode', 'pca_similarity', 'pca_convergence'
]


def calc_cov_mat_internal(q, q_ref=None):
    """
        Calculates the covariance matrix of a time-dependent matrix q,
        containing the independent components in its columns. The rows are treated
        as different times at which the components are evaluated.

        **Arguments:**

        q
            The txN matrix for which the NxN covariance matrix is determined

        **Optional arguments:**

        q_ref
            Reference vector of size 1xN. If not provided, the ensemble average is taken.
    """

    # Determine reference value as the ensemble average (along the time axis) if not predefined
    if q_ref is None:
        q_ref = q.mean(axis=0)

    # Calculation of the covariance matrix
    return np.einsum('ij,ik->ijk',q-q_ref, q-q_ref).mean(axis=0), q_ref

def calc_cov_mat(f, q_ref=None, start=0, end=None, step=1, select=None, path='trajectory/pos', mw=True):
    """
        Calculates the covariance matrix of the given trajectory.

        **Arguments:**

        f
            An h5.File instance containing the trajectory data.

        **Optional arguments:**

        q_ref
            Reference vector of the positions. If not provided, the ensemble
            average is taken.

        start
            The first sample to be considered for analysis. This may be
            negative to indicate that the analysis should start from the
            -start last samples.

        end
            The last sample to be considered for analysis. This may be
            negative to indicate that the last -end sample should not be
            considered.

        step
            The spacing between the samples used for the analysis

        select
            A list of atom indexes that are considered for the computation
            of the spectrum. If not given, all atoms are used.

        path
            The path of the dataset that contains the time dependent data in
            the HDF5 file. The first axis of the array must be the time
            axis.

        mw
            If mw is True, the covariance matrix is mass-weighted.
    """

    # Load in the relevant data
    q = f[path][start:end:step,:,:]
    # Select the given atoms
    if select is not None:
        q = q[:,select,:]
    # Reshape such that all Cartesian coordinates are treated equally
    q = q.reshape(q.shape[0],-1)

    # If necessary, weight with the mass
    if mw:
        # Select the necessary masses
        masses = f['system/masses']
        if select is not None:
            masses = masses[select]
        # Repeat d times, with d the dimension
        masses = np.repeat(masses,3)
        # Reweight with the masses
        q *= np.sqrt(masses)
    # Return the covariance matrix
    return calc_cov_mat_internal(q, q_ref)

def calc_pca(f_target, cov_mat=None, f=None, q_ref=None, start=0, end=None, step=1, select=None, path='trajectory/pos', mw=True, temp=None):
    """
        Performs a principle component analysis of the given trajectory.

        **Arguments:**

        f_target
            Path to an h5.File instance to which the results are written.

        **Optional arguments:**

        cov_mat
            The covariance matrix, if already calculated. If not provided,
            the covariance matrix will be calculatd based on the file f.

        f
            An h5.File instance containing the trajectory data.

        q_ref
            Reference vector of the positions. If not provided, the ensemble
            average is taken.

        start
            The first sample to be considered for analysis. This may be
            negative to indicate that the analysis should start from the
            -start last samples.

        end
            The last sample to be considered for analysis. This may be
            negative to indicate that the last -end sample should not be
            considered.

        step
            The spacing between the samples used for the analysis

        select
            A list of atom indexes that are considered for the computation
            of the spectrum. If not given, all atoms are used.

        path
            The path of the dataset that contains the time dependent data in
            the HDF5 file. The first axis of the array must be the time
            axis.

        mw
            If mass_weighted is True, the covariance matrix is mass-weighted.

        temp
            Temperature at which the simulation is carried out, necessary to determine the frequencies
    """
    if cov_mat is None:
        if f is None:
            AssertionError('No covariance matrix nor h5.File instance provided.')
        else:
            with log.section('PCA'):
                log('Calculating covariance matrix')
                cov_mat, q_ref = calc_cov_mat(f, q_ref, start, end, step, select, path, mw)

    with log.section('PCA'):
        log('Diagonalizing the covariance matrix')
        # Eigenvalue decomposition
        eigval, eigvec = np.linalg.eigh(cov_mat)
        # Order the eigenvalues in decreasing order
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:,idx]

        # Create output HDF5 file
        with h5.File(f_target, 'w') as g:
            pca = g.create_group('pca')
            # Output reference structure q_ref
            pca.create_dataset('q_ref', data=q_ref)
            # Output covariance matrix
            pca.create_dataset('cov_matrix', data=cov_mat)
            # Output eigenvectors in columns
            pca.create_dataset('pm', data=eigvec)
            # Output eigenvalues
            pca.create_dataset('eigvals', data=eigval)

            log('Determining inverse of the covariance matrix')
            # Process matrix to determine inverse
            # First, project out the three zero eigenvalues (translations)
            eigvec_reduced = eigvec[:,:-3]
            eigval_reduced = eigval[:-3]

            # Second, calculate the reduced covariance matrix and its inverse
            cov_mat_reduced = np.dot(np.dot(eigvec_reduced, np.diag(eigval_reduced)), eigvec_reduced.T)
            cov_mat_inverse = np.dot(np.dot(eigvec_reduced, np.diag(1/eigval_reduced)), eigvec_reduced.T)
            pca.create_dataset('cov_mat_red', data=cov_mat_reduced)
            pca.create_dataset('cov_mat_inv', data=cov_mat_inverse)

            # Third, if the temperature is specified, calculate the frequencies
            # (the zero frequencies are mentioned last so that their index corresponds to the principal modes)
            if temp is not None:
                log('Determining frequencies')
                frequencies = np.append(np.sqrt(boltzmann*temp/eigval_reduced)/(2*np.pi), np.repeat(0,3))
                pca.create_dataset('freqs', data=frequencies)

    return eigval, eigvec


def pca_projection(f_target, f, pm, start=0, end=None, step=1, select=None, path='trajectory/pos', mw=True):
    """
        Determines the principal components of an MD simulation

        **Arguments:**

        f_target
            Path to an h5.File instance to which the results are written.

        f
            An h5.File instance containing the trajectory data.

        pm
            An array containing the principal modes in its columns

        **Optional arguments:**

        start
            The first sample to be considered for analysis. This may be
            negative to indicate that the analysis should start from the
            -start last samples.

        end
            The last sample to be considered for analysis. This may be
            negative to indicate that the last -end sample should not be
            considered.

        step
            The spacing between the samples used for the analysis

        select
            A list of atom indexes that are considered for the computation
            of the spectrum. If not given, all atoms are used.

        path
            The path of the dataset that contains the time dependent data in
            the HDF5 file. The first axis of the array must be the time
            axis.

        mw
            If mass_weighted is True, the covariance matrix is mass-weighted.
    """
    # Load in the relevant data
    q = f[path][start:end:step,:,:]
    # Select the given atoms
    if select is not None:
        q = q[:,select,:]
    # Reshape such that all Cartesian coordinates are treated equally
    q = q.reshape(q.shape[0],-1)

    # If necessary, weight with the mass
    if mw:
        # Select the necessary masses
        masses = f['system/masses']
        if select is not None:
            masses = masses[select]
        # Repeat d times, with d the dimension
        masses = np.repeat(masses,3)
        # Reweight with the masses
        q *= np.sqrt(masses)

    # Calculation of the principal components: projection of each q_j on the principal modes
    with log.section('PCA'):
        log('Determining principal components')
        prin_comp = np.dot(q, pm)

    # Create output HDF5 file
    with h5.File(f_target, 'a') as g:
        if not 'pca' in g:
            pca = g.create_group('pca')
        else:
            pca = g['pca']
        pca.create_dataset('pc', data=prin_comp)


def write_principal_mode(f, f_pca, index, n_frames=100, select=None, mw=True, scaling=1.):
    """
        Writes out one xyz file per principal mode given in index

        **Arguments:**

        f
            Path to an h5.File instance containing the original data.

        f_pca
            Path to an h5.File instance containing the PCA, with reference structure, eigenvalues
            and principal modes.

        index
            An array containing the principal modes which need to be written out.

        **Optional arguments:**

        n_frames
            The number of frames in each xyz file.

        select
            A list of atom indexes that are considered for the computation
            of the spectrum. If not given, all atoms are used.

        mw
            If mass_weighted is True, the covariance matrix is assumed to be mass-weighted.

        scaling
            Scaling factor applied to the maximum deviation of the principal mode (i.e. the maximum
            principal component for that mode)
    """

    # Load in the relevant data
    # Atom numbers, masses and initial frame
    numbers = f['system/numbers']
    masses = f['system/masses']
    pos = f['trajectory/pos']
    if select is not None:
        numbers = numbers[select]
        masses = masses[select]
        pos = pos[:,select,:]
    masses = np.repeat(masses,3)
    # Data from the PC analysis
    grp = f_pca['pca']
    # The selected principal modes
    pm = grp['pm'][:,index]
    # The corresponding eigenvalues
    eigval = grp['eigvals'][index]
    # And the principal components
    pc = grp['pc'][:,index]

    with log.section('PCA'):
        for i in range(len(index)):
            log('Writing out principal mode %s' %index[i])
            if eigval[i] < 0:
                Warning('Negative eigenvalue encountered, skipping this entry')
                break
            # Determine maximum fluctuation (in units of meter*sqrt(kilogram))
            max_fluct = np.max(np.abs(pc[:,i]))
            # Initialize XYZWriter from molmod package
            xw = XYZWriter('pm_%s.xyz' %index[i], [pd[number].symbol for number in numbers])
            # Determine index in trajectory closest to rest state, and the corresponding positions
            ind_min = np.argmin(np.abs(pc[:,i]))
            r_ref = pos[ind_min,:,:]
            for j in range(n_frames):
                q_var = scaling*pm[:,i]*max_fluct*(2.*j-n_frames)/n_frames
                if mw:
                    q_var /= np.sqrt(masses)
                r = r_ref + q_var.reshape(-1,3)
                xw.dump('Frame%s' %j, r)
            del xw

def pca_similarity(covar_a, covar_b):
    """
        Calculates the similarity between the two covariance matrices

        **Arguments:**

        covar_a
            The first covariance matrix.

        covar_b
            The second covariance matrix.
    """
    # Take the square root of the symmetric matrices
    a_sq = spla.sqrtm(covar_a)
    b_sq = spla.sqrtm(covar_b)

    # Check for imaginary entries
    for mat in [a_sq, b_sq]:
        max_imag = np.amax(np.abs(np.imag(mat)))
        mean_real = np.mean(np.abs(np.real(mat)))
        if(max_imag/mean_real > 1e-6):
            Warning('Covariance matrix is not diagonally dominant')

    # Return the PCA similarity (1 - PCA distance)
    return 1 - np.sqrt(np.trace(np.dot(a_sq-b_sq, a_sq-b_sq))/(np.trace(covar_a+covar_b)))

def pca_convergence(f, eq_time=0*picosecond, n_parts=None, step=1, fn='PCA_convergence', n_bootstrap=50, mw=True):
    """
    Calculates the convergence of the simulation by calculating the pca
    similarity for different subsets of the simulation.

    **Arguments:**

    f
        An h5.File instance containing the trajectory data.

    **Optional arguments:**

    eq_time
        Equilibration time, discarded from the simulation.

    n_parts
        Array containing the number of parts in which
        the total simulation is divided.

    step
        Stepsize used in the trajectory.

    fn
        Filename containing the convergence plot.

    n_bootstrap
        The number of bootstrapped trajectories.

    mw
        If mass_weighted is True, the covariance matrix is mass-weighted.
    """

    # Configure n_parts, the array containing the number of parts in which the total simulation is divided
    if n_parts is None:
        n_parts = np.array([1,3,10,30,100,300])

    # Read in the timestep and the number of atoms
    time = f['trajectory/time']
    timestep = time[1] - time[0]
    time_length = len(time)

    # Determine the equilibration size
    eq_size = int(eq_time/timestep)

    ### ---PART A: SIMILARITY OF THE TRUE TRAJECTORY--- ###

    # Calculate the covariance matrix of the whole production run as golden standard
    covar_total, q_ref = calc_cov_mat(f, start=eq_size, step=step, mw=mw)

    # Initialize the average similarity vector of the divided trajectories
    sim_block = np.zeros(len(n_parts))
    # Calculate this average similarity vector
    for j in range(len(n_parts)):
        # Determine in how many parts the trajectory should be divided and the corresponding block size
        n_part = n_parts[j]
        block_size = (time_length-eq_size)//n_part
        # Calculate the n_part covariance matrices and compare with the total covariance matrix
        tot_sim_block=0
        for i in range(n_part):
            start = eq_size + i*block_size
            covars, tmp = calc_cov_mat(f, start=start, end=start+block_size+1, step=step, mw=mw)
            tot_sim_block += pca_similarity(covars, covar_total)
        # Determine the average similarity
        sim_block[j] = tot_sim_block/n_part


    ### ---PART B: SIMILARITY OF BOOTSTRAPPED TRAJECTORIES --- ###

    # Read in the positions, which will be used to generate bootstrapped trajectories
    pos = f['trajectory/pos'][eq_size:,:,:]
    pos = pos.reshape(pos.shape[0], -1)

    if mw:
        # Read in the masses of the atoms, and replicate them d times (d=dimension)
        masses = f['system/masses']
        masses = np.repeat(masses,3)

        # Create the mass-weighted positions matrix, on which the bootstrapping will be based
        pos *= np.sqrt(masses)

    # Initialize the vector containing the average similarity over all the bootstrapped, divided trajectories
    sim_bt_all = np.zeros(len(n_parts))

    for k in range(n_bootstrap):
        with log.section('PCA'):
            log('Processing %s of %s bootstrapped trajectories' %(k+1,n_bootstrap))
            # Create a bootstrapped trajectory bt
            pos_bt = np.zeros(pos.shape)
            random_time = random.random(time_length)*time_length
            for h in np.arange(time_length):
                pos_bt[h,:] = pos[random_time[h],:]

            # Covariance matrix of the total bootstrapped trajectory
            covar_bt_total, tmp = calc_cov_mat_internal(pos_bt)

            # Initialize the vector containing the average similarity over the different blocks,
            # for the given bootstrapped trajectory
            sim_bt = np.zeros(len(n_parts))

            for j in range(len(n_parts)):
                # Calculate the number of blocks, as well as the block size
                n_part = n_parts[j]
                block_size = (len(time)-eq_size)//n_part
                tot_sim_bt = 0
                # Calculate the total similarity of this number of blocks, for this bootstrapped trajectory
                for i in range(n_part):
                    start = eq_size + i*block_size
                    pos_bt_block = pos_bt[start:start+block_size:step]
                    covars_bt, tmp = calc_cov_mat_internal(pos_bt_block)
                    tot_sim_bt += pca_similarity(covars_bt, covar_bt_total)
                # Calculate the average similarity for this number of blocks, for this bootstrapped trajectory
                sim_bt[j] = tot_sim_bt/n_part
            sim_bt_all += sim_bt
    # Calculate the average similarity over all bootstrapped trajectories
    sim_bt_all /= n_bootstrap

    ### ---PART C: PROCESSING THE RESULTS --- ###

    pt.clf()
    pt.semilogx((time[-1]-time[0])/n_parts/picosecond, sim_block/sim_bt_all, 'r-')
    pt.semilogx((time[-1]-time[0])/n_parts/picosecond, sim_block/sim_bt_all, 'rs')
    pt.xlabel('Block size [ps]')
    pt.ylabel('PCA similarity (1=perfectly similar)')
    pt.title('Convergence assessment via PCA: ' + fn)
    pt.ylim([0,1])
    pt.savefig(fn+'.png')
    pt.savefig(fn+'.pdf', format='pdf')
    return sim_block/sim_bt_all
