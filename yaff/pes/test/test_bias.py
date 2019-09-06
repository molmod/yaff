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


from __future__ import division
from __future__ import print_function

import numpy as np
import pkg_resources

from yaff import *

from yaff.pes.test.common import check_gpos_part, check_vtens_part,\
    check_gpos_cv_fd, check_vtens_cv_fd
from yaff.test.common import get_system_quartz, get_alaninedipeptide_amber99ff

from molmod import bend_angle
from molmod import MolecularGraph


def test_bias_harmonicvolume_quartz():
    system = get_system_quartz()
    cv = CVVolume(system)
    K, q0 = 0.4, 0.9*system.cell.volume
    bias = HarmonicBias(K, q0, cv)
    part = ForcePartBias(system)
    part.add_term(bias)
    e = part.compute()
    echeck = 0.5*K*(system.cell.volume-q0)**2
    assert np.abs(e-echeck)<1e-8
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_bias_upperwallvolume_quartz():
    system = get_system_quartz()
    cv = CVVolume(system)
    K, q0 = 0.4, system.cell.volume
    bias = UpperWallBias(K, q0, cv)
    rvecs_orig = system.cell.rvecs.copy()
    part = ForcePartBias(system)
    part.add_term(bias)
    for scale in [0.5,2.0]:
        rvecs = rvecs_orig*scale
        system.cell.update_rvecs(rvecs)
        e = bias.compute()
        if system.cell.volume>q0:
            eref = 0.5*K*(system.cell.volume-q0)**2
        else: eref = 0.0
        assert np.abs(e-eref)<1e-8
        check_gpos_part(system, part)
        check_vtens_part(system, part)


def test_bias_lowerwallvolume_quartz():
    system = get_system_quartz()
    cv = CVVolume(system)
    K, q0 = 0.4, system.cell.volume
    bias = LowerWallBias(K, q0, cv)
    rvecs_orig = system.cell.rvecs.copy()
    part = ForcePartBias(system)
    part.add_term(bias)
    for scale in [0.5,2.0]:
        rvecs = rvecs_orig*scale
        system.cell.update_rvecs(rvecs)
        e = bias.compute()
        if system.cell.volume<q0:
            eref = 0.5*K*(system.cell.volume-q0)**2
        else: eref = 0.0
        assert np.abs(e-eref)<1e-8
        check_gpos_part(system, part)
        check_vtens_part(system, part)


def test_bias_multiple_terms():
    system = get_system_quartz()
    part = ForcePartBias(system)
    # Harmonic volume bias
    cv0 = CVVolume(system)
    K, q0 = 0.4, 0.9*system.cell.volume
    bias0 = HarmonicBias(K, q0, cv0)
    part.add_term(bias0)
    # Cosine of bending angle
    cv1 = BendAngle(0,1,2)
    m, a, phi0 = 1, 2.0, np.pi/4.0
    bias1 = Cosine(m, a, phi0, cv1)
    part.add_term(bias1)
    # Check energy
    e = part.compute()
    contributions = part.get_term_energies()
    assert np.abs(e-np.sum(contributions))<1e-5
    assert contributions[0]==bias0.compute()
    # Check derivatives
    check_gpos_part(system, part)
    check_vtens_part(system, part)
    # Check collective variable values
    cv_values0 = part.get_term_cv_values(0)
    assert cv_values0.shape[0]==1
    assert cv_values0[0]==cv0.compute()
    cv_values1 = part.get_term_cv_values(1)
    assert cv_values1.shape[0]==1
    delta0 = system.pos[1] - system.pos[0]
    system.cell.mic(delta0)
    delta2 = system.pos[1] - system.pos[2]
    system.cell.mic(delta2)
    phi = bend_angle([delta0, np.zeros(3, float), delta2])[0]
    assert np.abs(cv_values1[0]-phi)<1e-5

def test_bias_distance2plane_dummysystem():
    # Test a construction for determining the distance between a point and a
    # plane; the point is defined as an average of two points, the plane is
    # defined by three points, some again defined as the average of more than
    # one point.
    numbers = np.ones((6,), dtype=int)
    pos = np.zeros((numbers.shape[0],3))
    system = System(numbers, pos)
    # Plane defined from points 0,1 and average of 2 and 3:
    # We choose z=1, so the distance is simply the z coordinate of the point
    pos[0] = [0.0,0.0,1.0]
    pos[1] = [1.0,0.0,1.0]
    pos[2] = [0.0,1.0,1.0]
    pos[3] = [0.0,1.0,1.0]
    # Construct COMList; each group is a collection of atoms from which a
    # position is calculated as a weighted average of atomic positions
    groups = [ (np.array([0]), np.array([1.0])),
               (np.array([1]), np.array([1.0])),
               (np.array([2,3]), np.array([0.5,0.5])),
               (np.array([4,5]), np.array([0.5,0.5])) ]
    comlist = COMList(system, groups)
    bias = ForcePartBias(system, comlist=comlist)
    # Define a bias potential, harmonic in the distance from group 3 to the
    # plane spanned by groups 0, 1 and 2
    ic = OopDist(0,1,2,3)
    term = Harmonic(0.2,0.1,ic)
    bias.add_term(term, use_comlist=True)
    # Define different positions of group 3
    pointpos = [
        ([0.0,0.0,2.0],[0.0,0.0,0.0], 0.0),# Point in plane
        ([6.0,0.0,2.0],[0.0,3.0,2.0], 1.0),# Point above plane
        ([2.0,-3.,4.0],[0.0,3.0,-8.0], -3.0),# Point below plane
    ]
    for r0,r1,distance in pointpos:
        pos[4] = r0
        pos[5] = r1
        bias.compute()
        d = bias.get_term_cv_values(0)
        assert d[0]==distance

def test_bias_pathdeviation_mof5():
    # Load the system
    fn_system = pkg_resources.resource_filename(__name__, '../../data/test/system_mof5.chk')
    system = System.from_file(fn_system)
    # Groups that define the COM
    atypes = set(['C_B', 'C_B_BR_O', 'O_p', 'B_p','C_HTTP', 'C_O_BR_O', 'C_O'])
    graph = MolecularGraph(system.bonds, system.numbers)
    indices = graph.independent_vertices
    groups = []
    for layer in [0,1]:
        groups.append( [iatom for iatom in indices[layer] if system.get_ffatype(iatom) in atypes] )
    # Collective Variables
    cv0 = CVCOMProjection(system, groups, 0)
    cv1 = CVCOMProjection(system, groups, 1)
    # The path, rather random nonsense
    # This potential has a discontinuous derivative when there is a jump from
    # one nearest point on the path to the next. This can lead to failure of
    # the check_gpos_part and check_vtens_part tests when such a jump occurs
    # in the finite difference approximation. We avoid this by taking a rather
    # coarse path, so we are always close to the same point of the path.
    npoints = 20
    path = np.zeros((npoints,3))
    path[:,0] = np.cos(np.linspace(0,np.pi,npoints))*0.1
    path[:,1] = np.sin(np.linspace(0,np.pi,npoints))*-0.1
    path[:,2] = 0.2*path[:,0]+path[:,1]**2
    # Test without the harmonic restraint
    bias = PathDeviationBias([cv0,cv1], path, 0.0)
    part = ForcePartBias(system)
    part.add_term(bias)
    check_gpos_part(system, part)
    check_vtens_part(system, part)
    # Test without the harmonic restraint, closest point on path is starting
    # point of the path
    bias = PathDeviationBias([cv0,cv1], path[5:], 0.0)
    index, _, _ = bias.find_nearest_point(np.array([cv.compute() for cv in bias.cvs]))
    assert index==0
    part = ForcePartBias(system)
    part.add_term(bias)
    check_gpos_part(system, part)
    check_vtens_part(system, part)
    # Test with the harmonic restraint
    bias = PathDeviationBias([cv0,cv1], path, 0.5)
    part = ForcePartBias(system)
    part.add_term(bias)
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_bias_gaussianhills_quartz():
    system = get_system_quartz()
    # Volume
    cv0 = CVVolume(system)
    q0 = system.cell.volume
    # Distance between 2 and 8
    cv1 = CVInternalCoordinate(system, Bond(2,8))
    delta = system.pos[2]-system.pos[8]
    system.cell.mic(delta)
    q1 = np.linalg.norm(delta)
    # Widths of the Gaussians
    sigmas = np.array([10.0*angstrom**3, 0.2*angstrom])
    bias = GaussianHills([cv0, cv1], sigmas)
    # No hills added, energy should be zero
    e = bias.compute()
    assert e==0.0
    # Add one hill
    K0 = 5*kjmol
    bias.add_hill(np.array([q0+sigmas[0],q1-2*sigmas[1]]), K0)
    e = bias.compute()
    assert np.abs(e - K0*np.exp(-2.5)) < 1e-10
    # Add another hill
    K1 = 2*kjmol
    bias.add_hill(np.array([q0-3*sigmas[0],q1-1*sigmas[1]]), K1)
    e = bias.compute()
    assert np.abs(e - K0*np.exp(-2.5) - K1*np.exp(-5.0)) < 1e-10
    # Test derivatives
    bias.add_hill(np.array([q0-3*sigmas[0],q1-1*sigmas[1]]), K1)
    part = ForcePartBias(system)
    part.add_term(bias)
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_bias_gaussianhills_alanine():
    ff = get_alaninedipeptide_amber99ff()
    cv = CVInternalCoordinate(ff.system, DihedAngle(4,6,8,14))
    sigma = 0.35*rad
    bias = GaussianHills(cv, sigma)
    e = bias.compute()
    assert e==0.0
    K, phi0 = 3*kjmol, -2.5*rad
    bias.add_hill(phi0, K)
    e = bias.compute()
    phi = cv.compute()
    eref = K*np.exp(-(phi-phi0)**2/2/sigma**2)
    assert np.abs(e-eref)<1e-10*kjmol
    part = ForcePartBias(ff.system)
    part.add_term(bias)
    check_gpos_part(ff.system, part)
    check_vtens_part(ff.system, part)
