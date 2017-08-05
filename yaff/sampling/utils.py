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
'''Auxiliary routines for initial velocities'''


from __future__ import division

from molmod import boltzmann

import numpy as np


__all__ = [
    'get_random_vel', 'remove_com_moment', 'remove_angular_moment',
    'clean_momenta', 'angular_moment', 'get_ndof_internal_md',
    'cell_symmetrize', 'get_random_vel_press', 'get_ndof_baro',
    'stabilized_cholesky_decomp'
]


def get_random_vel(temp0, scalevel0, masses, select=None):
    '''Generate random atomic velocities using a Maxwell-Boltzmann distribution

       **Arguments:**

       temp0
            The temperature for the Maxwell-Boltzmann distribution.

       scalevel0
            When set to True, the velocities are rescaled such that the
            instantaneous temperature coincides with temp0.

       masses
            An (N,) array with atomic masses.

       **Optional arguments:**

       select
            When given, this must be an array of integer indexes. Only for these
            atoms (masses) initial velocities will be generated.

       **Returns:** An (N, 3) array with random velocities. When the select
       option is used, the shape of the results is (M, 3), where M is the length
       of the select array.
    '''
    if select is not None:
        masses = masses[select]
    shape = len(masses), 3
    result = np.random.normal(0, 1, shape)*np.sqrt(boltzmann*temp0/masses).reshape(-1,1)
    if scalevel0 and temp0 > 0:
        temp = (result**2*masses.reshape(-1,1)).mean()/boltzmann
        scale = np.sqrt(temp0/temp)
        result *= scale
    return result


def remove_com_moment(vel, masses):
    '''Zero the linear center-of-mass momentum.

       **Arguments:**

       vel
            An (N, 3) array with atomic velocities. This array is modified
            in-place.

       masses
            An (N,) array with atomic masses

       The zero linear COM momentum is achieved by subtracting translational
       rigid body motion from the atomic velocities.
    '''
    # compute the center of mass velocity
    com_vel = np.dot(masses, vel)/masses.sum()
    # subtract this com velocity vector from each atomic velocity
    vel[:] -= com_vel

def remove_angular_moment(pos, vel, masses):
    '''Zero the global angular momentum.

       **Arguments:**

       pos
            An (N, 3) array with atomic positions. This array is not modified.

       vel
            An (N, 3) array with atomic velocities. This array is modified
            in-place.

       masses
            An (N,) array with atomic masses

       The zero angular momentum is achieved by subtracting angular rigid body
       motion from the atomic velocities. (The angular momentum is measured
       with respect to the center of mass to avoid that this routine
       reintroduces a linear COM velocity. This is also beneficial for the
       numerical stability.)
    '''
    # translate a copy of the positions, such that the center of mass is in the origin
    pos = pos.copy()
    com_pos = np.dot(masses, pos)/masses.sum()
    pos -= com_pos
    # compute the inertia tensor
    iner_tens = inertia_tensor(pos, masses)
    # compute the angular momentum vector
    ang_mom = angular_moment(pos, vel, masses)
    # compute the angular velocity vector
    ang_vel = angular_velocity(ang_mom, iner_tens)
    # subtract the rigid body angular velocities from the atomic velocities
    vel[:] -= rigid_body_angular_velocities(pos, ang_vel)


def clean_momenta(pos, vel, masses, cell):
    '''Remove any relevant external momenta

       **Arguments:**

       pos
            An (N, 3) array with atomic positions. This array is not modified.

       vel
            An (N, 3) array with atomic velocities. This array is modified
            in-place.

       masses
            An (N,) array with atomic masses

       cell
            A Cell instance describing the periodic boundary conditions.
    '''
    remove_com_moment(vel, masses)
    if cell.nvec == 0:
        # remove all angular momenta
        remove_angular_moment(pos, vel, masses)
    elif cell.nvec == 1:
        # TODO: only the angular momentum about the cell vector has to be
        # projected out
        raise NotImplementedError


def inertia_tensor(pos, masses):
    '''Compute the inertia tensor for a given set of point particles

       **Arguments:**

       pos
            An (N, 3) array with atomic positions.

       masses
            An (N,) array with atomic masses.

       **Returns:** a (3, 3) array containing the inertia tensor.
    '''
    return np.identity(3)*(masses.reshape(-1,1)*pos**2).sum() - np.dot(pos.T, masses.reshape(-1,1)*pos)


def angular_moment(pos, vel, masses):
    '''Compute the angular moment of a set of point particles

       **Arguments:**

       pos
            An (N, 3) array with atomic positions.

       vel
            An (N, 3) array with atomic velocities.

       masses
            An (N,) array with atomic masses.

       **Returns:** a (3,) array with the angular momentum vector.
    '''
    lin_moms = masses.reshape(-1,1)*vel
    ang_mom = np.zeros(3, float)
    ang_mom[0] = (pos[:,1]*lin_moms[:,2] - pos[:,2]*lin_moms[:,1]).sum()
    ang_mom[1] = (pos[:,2]*lin_moms[:,0] - pos[:,0]*lin_moms[:,2]).sum()
    ang_mom[2] = (pos[:,0]*lin_moms[:,1] - pos[:,1]*lin_moms[:,0]).sum()
    return ang_mom


def angular_velocity(ang_mom, iner_tens, epsilon=1e-10):
    '''Derive the angular velocity from the angular moment and the inertia tensor

       **Arguments:**

       ang_mom
            An (3,) array with angular momenta.

       iner_tens
            A (3, 3) array with the inertia tensor.

       **Optional arguments:**

       epsilon
            A threshold for the low eigenvalues of the inertia tensor. When an
            eigenvalue is below this threshold, it is assumed to be zero plus
            some (irrelevant) numerical noise.

       **Returns:** An (3,) array with the angular velocity vector.

       In principle these routine should merely return::

           np.linalg.solve(inter_tens, ang_mom).

       However, when the inertia tensor has zero eigenvalues (linear molecules,
       single atoms), this routine will use a proper pseudo inverse of the
       inertia tensor.
    '''
    evals, evecs = np.linalg.eigh(iner_tens)
    # select the significant part of the decomposition
    mask = evals > epsilon
    evals = evals[mask]
    evecs = evecs[:,mask]
    # compute the pseudoinverse
    return np.dot(evecs, np.dot(evecs.T, ang_mom)/evals)


def rigid_body_angular_velocities(pos, ang_vel):
    '''Generate the velocities of a set of atoms that move as a rigid body.

       **Arguments:**

       pos
            An (N, 3) array with atomic positions.

       ang_vel
            An (3,) array with the angular velocity vector of the rigid body.

       **Returns:** An (N, 3) array with atomic velocities in the rigid body.
       Note that the linear momentum is zero.
    '''
    vel = np.zeros(pos.shape, float)
    vel[:,0] = (ang_vel[1]*pos[:,2] - ang_vel[2]*pos[:,1])
    vel[:,1] = (ang_vel[2]*pos[:,0] - ang_vel[0]*pos[:,2])
    vel[:,2] = (ang_vel[0]*pos[:,1] - ang_vel[1]*pos[:,0])
    return vel


def get_ndof_internal_md(natom, nper):
    '''Return the effective number of internal degrees of freedom for MD simulations

       **Arguments:**

       natom
            The number of atoms

       nper
            The number of periodic boundary conditions (0 for isolated systems)
    '''
    if nper == 0:
        # isolated systems
        if natom == 1:
            # single atom
            return 0
        elif natom == 2:
            # the diatomic case
            return 1
        else:
            # No distinction is made between linear and non-linear molecules
            # because a vibrating linear molecule is non-linear.
            return 3*natom - 6
    elif nper == 1:
        # 1D periodic: three translations and one rotation about the cell vector.
        return 3*natom - 4
    else:
        # 2D and 3D periodic
        return 3*natom - 3

def cell_symmetrize(ff, vector_list = None, tensor_list = None):
    '''Symmetrizes the unit cell tensor, and updates the position vectors

    **Arguments:**

    ff
        A ForceField instance

    **Optional arguments:**

    vector_list
        A list of numpy vectors which should be transformed under the
        symmetrization. Note that the positions are already transformed
        automatically

    tensor_list
        A list of numpy tensors of rank 2 which should be transformed
        under the symmetrization.
    '''
    # store the unit cell tensor
    cell = ff.system.cell.rvecs.copy()
    # SVD decomposition of cell tensor
    U, s, Vt = np.linalg.svd(cell)
    # definition of the rotation matrix to symmetrize cell tensor
    rot_mat = np.dot(Vt.T, U.T)
    # symmetrize cell tensor and update cell
    cell = np.dot(cell, rot_mat)
    ff.update_rvecs(cell)
    # also update the new atomic positions
    pos_new = np.dot(ff.system.pos, rot_mat)
    ff.update_pos(pos_new)
    # initialize the new vector and tensor lists
    new_vector_list = []
    new_tensor_list = []
    # update the additional vectors from vector_list
    if vector_list is not None:
        for i in range(len(vector_list)):
            new_vector_list.append(np.dot(vector_list[i], rot_mat))
    # update the additional tensors from tensor_list
    if tensor_list is not None:
        for i in range(len(tensor_list)):
            new_tensor_list.append(np.dot(np.dot(rot_mat.T, tensor_list[i]), rot_mat))
    return new_vector_list, new_tensor_list

def get_random_vel_press(mass, temp):
    '''Generates symmetric tensor of barostat velocities

    **Arguments:**

    mass
        The Barostat mass

    temp
        The temperature at which the velocities are selected
    '''
    shape = 3, 3
    # generate random 3x3 tensor
    rand = np.random.normal(0, np.sqrt(mass*boltzmann*temp), shape)/mass
    vel_press = np.zeros(shape)
    # create initial symmetric pressure velocity tensor
    for i in range(3):
        for j in range(3):
            if i >= j:
                vel_press[i,j] = rand[i,j]
            else:
                vel_press[i,j] = rand[j,i]
            # correct for p_ab = p_ba, hence only 1 dof if a != b
            if i != j:
                vel_press[i,j] /= np.sqrt(2)
    return vel_press


def get_ndof_baro(dim, anisotropic, vol_constraint):
    '''Calculates the number of degrees of freedom associated with the cell fluctuation

    **Arguments:**

    dim
        The dimension of the system

    anisotropic
        Boolean value determining whether anisotropic cell fluctuations are allowed

    vol_constraint
        Boolean value determining whether the cell volume can change
    '''
    baro_ndof = 1
    # degrees of freedom for a symmetric cell tensor
    if anisotropic:
        baro_ndof = dim*(dim+1)//2
    # decrease the number of dof by one if volume is constant
    if vol_constraint:
        baro_ndof -= 1
    # verify at least one degree of freedom is left
    if baro_ndof == 0:
        raise AssertionError('Isotropic barostat called with a volume constraint')
    return baro_ndof

def stabilized_cholesky_decomp(mat):
    '''
    Do LDL^T and transform to MM^T with negative diagonal entries of D put equal to zero
    Assume mat is square and symmetric (but not necessarily positive definite).
    '''
    if np.all(np.linalg.eigvals(mat) > 0):
        return np.linalg.cholesky(mat)  #usual cholesky decomposition
    else:
        n = int(mat.shape[0])
        D = np.zeros(n)
        L = np.zeros(mat.shape)

        for i in np.arange(n):
            L[i, i] = 1

            for j in np.arange(i):
                L[i, j] = mat[i, j]
                for k in np.arange(j):
                    L[i, j] -= L[i, k] * L[j, k] * D[k]

                if abs(D[j]) > 1e-12:
                    L[i, j] *= 1.0/D[j]
                else:
                    L[i, j] = 0

            D[i] = mat[i, i]
            for k in np.arange(i):
                D[i] -= L[i, k] * L[i, k] * D[k]

        D = np.sqrt(D.clip(min=0))
        return L*D
