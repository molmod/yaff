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
import os

from molmod.test.common import tmpdir
from molmod.units import kjmol, angstrom

from yaff import *

from yaff.test.common import get_system_water32
from yaff.pes.test.common import get_part_water32_9A_lj

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
    try:
        from lammps import lammps
    except:
        from nose.plugins.skip import SkipTest
        raise SkipTest('Could not import lammps')
    return
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

def test_liblammps_macos():
    try:
        from lammps import lammps
    except:
        from nose.plugins.skip import SkipTest
        raise SkipTest('Could not import lammps')
#    print(lammps.__file__)
    import sys
    from os.path import dirname,abspath,join
    from inspect import getsourcefile
    from ctypes import CDLL, RTLD_GLOBAL
    modpath = dirname(abspath(getsourcefile(lammps)))
    print(modpath)
    if sys.platform == 'darwin':
      lib_ext = ".dylib"
    else:
      lib_ext = ".so"
    print(lib_ext)
    print(join(modpath,"liblammps" + lib_ext))
    lib = CDLL(join(modpath,"liblammps" + lib_ext),RTLD_GLOBAL)
    print("Opened correctly")
    lib = CDLL("liblammps" + lib_ext,RTLD_GLOBAL)
    assert False
