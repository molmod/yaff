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
import os
import unittest

from molmod.test.common import tmpdir
from molmod.units import kjmol, angstrom, pascal
mpa = 1e6*pascal

from yaff import *

from yaff.test.common import get_system_water32, get_system_quartz
from yaff.pes.test.common import get_part_water32_9A_lj

try:
    from lammps import lammps
except:
    from nose.plugins.skip import SkipTest
    raise SkipTest('Could not import lammps, skipping all LAMMPS related tests')

def get_water32_ff(lj=True,ei=False,gaussian_charges=False):
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_9A_lj()
    parts = []
    if lj: parts.append(part_pair)
    if ei:
        rcut = part_pair.pair_pot.rcut
        alpha = 4.0/rcut
        if gaussian_charges:
            radii = np.tile([1.0,0.6,0.6],32)
        else: radii = None
        # Construct the ewald real-space potential and part
        ewald_real_pot = PairPotEI(system.charges, alpha, rcut=rcut,radii=radii)
        part_pair_ewald_real = ForcePartPair(system, nlist, scalings, ewald_real_pot)
        # Construct the ewald reciprocal and correction part
        part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, gcut=1.2*alpha)
        part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings)
        parts += [part_pair_ewald_real, part_ewald_reci, part_ewald_corr]
    ff = ForceField(system, parts, nlist)
    return ff


def compare_lammps_yaff_ff(ff, thresh=1.0, do_ei=True, do_vdw=True):
    lammps_ffa, lammps_ffa_ids = get_lammps_ffatypes(ff)
    gpos_yaff, vtens_yaff = np.zeros(ff.system.pos.shape), np.zeros((3,3))
    eyaff = ff.compute(gpos=gpos_yaff, vtens=vtens_yaff)
    with tmpdir(__name__, 'test_liblammps') as dirname:
        # Write LAMMPS data file
        fn_system = os.path.join(dirname,'system.dat')
        write_lammps_system_data(ff.system, ff=ff,fn=fn_system)
        # Write LAMMPS table file
        fn_table = os.path.join(dirname,'table.dat')
        write_lammps_table(ff,fn=fn_table)
        # Construct the LAMMPS force-field contribution
        fn_log = os.path.join(dirname,'lammps.log')
        part_lammps = ForcePartLammps(ff,fn_log=fn_log,scalings_table=np.zeros(3),scalings_ei=np.zeros(3),
            fn_system=fn_system,fn_table=fn_table,do_ei=do_ei, do_table=do_vdw)
        gpos_lammps, vtens_lammps = np.zeros(ff.system.pos.shape), np.zeros((3,3))
        elammps = part_lammps.compute(gpos=gpos_lammps, vtens=vtens_lammps)
        rmsd_gpos = np.std(gpos_yaff-gpos_lammps)
        rmsd_vtens = np.std(vtens_yaff-vtens_lammps)
#        print("E_YAFF = %12.4f E_LAMMPS = %12.4f E_DIFF = %12.4f kJ/mol"%(eyaff/kjmol,elammps/kjmol,(eyaff-elammps)/kjmol))
#        print("RMSD GPOS  = %12.8f kJ/mol/A"%(rmsd_gpos/kjmol*angstrom))
#        print("RMSD VTENS = %12.8f kJ/mol"%(rmsd_vtens/kjmol))
        assert np.abs(eyaff-elammps)<1e-3*kjmol*thresh
        assert rmsd_gpos<1e-4*kjmol/angstrom*thresh
        assert rmsd_vtens<1e-2*kjmol*thresh


def test_liblammps_water32():
    # Only LJ
    ff = get_water32_ff(lj=True,ei=False)
    compare_lammps_yaff_ff(ff, do_ei=False, do_vdw=True)
    # Only point-charge electrostatics (note that larger errors are expected
    # when electrostatics are included
    ff = get_water32_ff(lj=False,ei=True)
    compare_lammps_yaff_ff(ff, do_ei=True, do_vdw=False,thresh=10)
    # Only Gaussian-charge electrostatics
    ff = get_water32_ff(lj=False,ei=True,gaussian_charges=True)
    compare_lammps_yaff_ff(ff, do_ei=True, do_vdw=True,thresh=10)
    # LJ+electrostatics
    ff = get_water32_ff(lj=True,ei=True,gaussian_charges=True)
    compare_lammps_yaff_ff(ff, do_ei=True, do_vdw=True,thresh=10)


def compare_lammps_yaff_swap_noncovalent(name, thresh=1.0):
    # Load the system
    fn_system = pkg_resources.resource_filename(__name__, '../../data/test/system_%s.chk'%name)
    system = System.from_file(fn_system)
    # Generate the YAFF ForceField
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_%s.txt'%name)
    ff = ForceField.generate(system, fn_pars, alpha_scale=3.2, gcut_scale=1.5, rcut=15.0*angstrom, smooth_ei=True)
    gpos_yaff, vtens_yaff = np.zeros(ff.system.pos.shape), np.zeros((3,3))
    eyaff = ff.compute(gpos=gpos_yaff, vtens=vtens_yaff)
    pyaff = np.trace(vtens_yaff)/3.0/ff.system.cell.volume
    # Replace noncovalent contributions with LAMMPS table
    with tmpdir(__name__, 'test_liblammps_swap') as dirname:
        fn_system = os.path.join(dirname,'system.dat')
        fn_table = os.path.join(dirname,'table.dat')
        ff_lammps = swap_noncovalent_lammps(ff, fn_system=fn_system,
            fn_table=fn_table)
        gpos_lammps, vtens_lammps = np.zeros(ff.system.pos.shape), np.zeros((3,3))
        elammps = ff_lammps.compute(gpos=gpos_lammps, vtens=vtens_lammps)
        plammps = np.trace(vtens_lammps)/3.0/ff.system.cell.volume
        rmsd_gpos = np.std(gpos_yaff-gpos_lammps)
        rmsd_vtens = np.std(vtens_yaff-vtens_lammps)
        print("E_YAFF = %12.4f E_LAMMPS = %12.4f E_DIFF = %12.4f kJ/mol"%(eyaff/kjmol,elammps/kjmol,(eyaff-elammps)/kjmol))
        print("P_YAFF = %12.4f P_LAMMPS = %12.4f P_DIFF = %12.4f MPa"%(pyaff/mpa,plammps/mpa,(pyaff-plammps)/mpa))
        print("RMSD GPOS  = %12.8f kJ/mol/A"%(rmsd_gpos/kjmol*angstrom))
        print("RMSD VTENS = %12.8f kJ/mol"%(rmsd_vtens/kjmol))
        assert np.abs(eyaff-elammps)<1e-1*kjmol*thresh
        assert np.abs(pyaff-plammps)<1*mpa*thresh
        assert rmsd_gpos<1e-2*kjmol/angstrom*thresh
        assert rmsd_vtens<0.5*kjmol*thresh


def test_liblammps_mil53_quickff():
    compare_lammps_yaff_swap_noncovalent("mil53", thresh=100.0)


def test_liblammps_mil47_quickff():
    compare_lammps_yaff_swap_noncovalent("mil47", thresh=100.0)


def make_quartz_bks_ff(system, scaling_factors0, scaling_factors1):
    nlist = NeighborList(system)
    rcut = 9.0*angstrom
    nffa = system.ffatypes.shape[0]
    parts = []

    def fill_cross(ffatypes, pars):
        crosspars = np.zeros((ffatypes.shape[0],ffatypes.shape[0]))
        for iffa, ffa0 in enumerate(ffatypes):
            for jffa, ffa1 in enumerate(ffatypes):
                crosspars[iffa,jffa] = pars[(ffa0,ffa1)]
        return crosspars

    # Damped dispersion
    c6_dict = {('O','O'):175.0*electronvolt*angstrom**6,
               ('Si','O'):1.3353810000e+02*electronvolt*angstrom**6,
               ('O','Si'):1.3353810000e+02*electronvolt*angstrom**6,
               ('Si','Si'):120.0*electronvolt*angstrom**6,}
    b_dict = {('O','O'):0.0,
               ('Si','O'):0.0,
               ('O','Si'):0.0,
               ('Si','Si'):0.0}
    cn_cross = fill_cross(system.ffatypes,c6_dict)
    b_cross = fill_cross(system.ffatypes,b_dict)
    pair_pot0 = PairPotDampDisp(system.ffatype_ids,cn_cross,b_cross,rcut)
    scalings0 = Scalings(system,scale1=scaling_factors0[0],scale2=scaling_factors0[1],
        scale3=scaling_factors0[2],scale4=scaling_factors0[3])
    part_pair0 = ForcePartPair(system, nlist, scalings0, pair_pot0)

    # Exponential repulsion
    a_dict = {('O','O'):1.3887730000e+03*electronvolt,
               ('Si','O'):1.8003757200e+04*electronvolt,
               ('O','Si'):1.8003757200e+04*electronvolt,
               ('Si','Si'):0.0*electronvolt,}
    be_dict = {('O','O'):2.7600000000e+00/angstrom,
               ('Si','O'):4.8731800000e+00/angstrom,
               ('O','Si'):4.8731800000e+00/angstrom,
               ('Si','Si'):0.0}
    a_cross = fill_cross(system.ffatypes,c6_dict)
    be_cross = fill_cross(system.ffatypes,be_dict)
    pair_pot1 = PairPotExpRep(system.ffatype_ids,a_cross,be_cross,rcut)
    scalings1 = Scalings(system,scale1=scaling_factors1[0],scale2=scaling_factors1[1],
        scale3=scaling_factors1[2],scale4=scaling_factors1[3])
    part_pair1 = ForcePartPair(system, nlist, scalings1, pair_pot1)
    return ForceField(system, [part_pair0, part_pair1], nlist=nlist)


def test_liblammps_quartz_bks():
    def compare_lammps_yaff_quartz(system, scaling_factors0, scaling_factors1):
        ff = make_quartz_bks_ff(system, scaling_factors0, scaling_factors1)
        gpos_yaff, vtens_yaff = np.zeros(ff.system.pos.shape), np.zeros((3,3))
        eyaff = ff.compute(gpos=gpos_yaff, vtens=vtens_yaff)
        pyaff = np.trace(vtens_yaff)/3.0/ff.system.cell.volume
        # Replace noncovalent contributions with LAMMPS table
        with tmpdir(__name__, 'test_liblammps_swap') as dirname:
            fn_system = os.path.join(dirname,'system.dat')
            fn_table = os.path.join(dirname,'table.dat')
            ff_lammps = swap_noncovalent_lammps(ff, fn_system=fn_system,
                fn_table=fn_table)
            gpos_lammps, vtens_lammps = np.zeros(ff.system.pos.shape), np.zeros((3,3))
            elammps = ff_lammps.compute(gpos=gpos_lammps, vtens=vtens_lammps)
            plammps = np.trace(vtens_lammps)/3.0/ff.system.cell.volume
            rmsd_gpos = np.std(gpos_yaff-gpos_lammps)
            rmsd_vtens = np.std(vtens_yaff-vtens_lammps)
            print("E_YAFF = %12.4f E_LAMMPS = %12.4f E_DIFF = %12.4f kJ/mol"%(eyaff/kjmol,elammps/kjmol,(eyaff-elammps)/kjmol))
            print("P_YAFF = %12.4f P_LAMMPS = %12.4f P_DIFF = %12.4f MPa"%(pyaff/mpa,plammps/mpa,(pyaff-plammps)/mpa))
            print("RMSD GPOS  = %12.8f kJ/mol/A"%(rmsd_gpos/kjmol*angstrom))
            print("RMSD VTENS = %12.8f kJ/mol"%(rmsd_vtens/kjmol))
            assert np.abs(eyaff-elammps)<0.5*kjmol
            assert np.abs(pyaff-plammps)<0.1*mpa
            assert rmsd_gpos<1e-3*kjmol/angstrom
            assert rmsd_vtens<0.5*kjmol
    system = get_system_quartz().supercell(4,4,4)
    scaling_factors0 = (1.0,1.0,1.0,1.0)
    scaling_factors1 = (1.0,1.0,1.0,1.0)
    compare_lammps_yaff_quartz(system, scaling_factors0, scaling_factors1)
    scaling_factors0 = (0.0,1.0,1.0,1.0)
    scaling_factors1 = (1.0,1.0,1.0,1.0)
    compare_lammps_yaff_quartz(system, scaling_factors0, scaling_factors1)
    scaling_factors0 = (0.0,1.0,1.0,1.0)
    scaling_factors1 = (1.0,0.2,1.0,1.0)
    compare_lammps_yaff_quartz(system, scaling_factors0, scaling_factors1)
    scaling_factors0 = (0.0,1.0,1.0,0.0)
    scaling_factors1 = (1.0,0.2,1.0,1.0)
    compare_lammps_yaff_quartz(system, scaling_factors0, scaling_factors1)
