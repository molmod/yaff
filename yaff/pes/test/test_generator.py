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

import pkg_resources
import numpy as np

from yaff import *

from yaff.test.common import get_system_water32, get_system_glycine, get_system_formaldehyde


def test_generator_water32_bondharm():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_bondharm.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    assert part_valence.dlist.ndelta == 64
    for i, j in system.bonds:
        row = part_valence.dlist.lookup.get((i, j))
        assert row is not None
    assert (part_valence.iclist.ictab['kind'] == 0).all()
    assert part_valence.iclist.nic == 64
    assert (part_valence.vlist.vtab['kind'] == 0).all()
    assert abs(part_valence.vlist.vtab['par0'] - 4.0088096730e+03*(kjmol/angstrom**2)).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'] - 1.0238240000e+00*angstrom).max() < 1e-10
    assert part_valence.vlist.nv == 64


def test_generator_water32_bondfues():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_bondfues.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    assert part_valence.dlist.ndelta == 64
    for i, j in system.bonds:
        row = part_valence.dlist.lookup.get((i, j))
        assert row is not None
    assert (part_valence.iclist.ictab['kind'] == 0).all()
    assert part_valence.iclist.nic == 64
    assert (part_valence.vlist.vtab['kind'] == 2).all()
    assert abs(part_valence.vlist.vtab['par0'] - 4.0088096730e+03*(kjmol/angstrom**2)).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'] - 1.0238240000e+00*angstrom).max() < 1e-10
    assert part_valence.vlist.nv == 64


def test_generator_water32_bendaharm():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_bendaharm.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    assert part_valence.dlist.ndelta == 64
    for i, j in system.bonds:
        row = part_valence.dlist.lookup.get((i, j))
        assert row is not None
    assert (part_valence.iclist.ictab['kind'] == 2).all()
    assert part_valence.iclist.nic == 32
    assert (part_valence.vlist.vtab['kind'] == 0).all()
    assert abs(part_valence.vlist.vtab['par0'] - 3.0230353700e+02*kjmol).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'] - 8.8401698835e+01*deg).max() < 1e-10
    assert part_valence.vlist.nv == 32


def test_generator_water32_bendcharm():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_bendcharm.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    assert part_valence.dlist.ndelta == 64
    for i, j in system.bonds:
        row = part_valence.dlist.lookup.get((i, j))
        assert row is not None
    assert (part_valence.iclist.ictab['kind'] == 1).all()
    assert part_valence.iclist.nic == 32
    assert (part_valence.vlist.vtab['kind'] == 0).all()
    assert abs(part_valence.vlist.vtab['par0'] - 3.0230353700e+02*kjmol).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'] - np.cos(8.8401698835e+01*deg)).max() < 1e-10
    assert part_valence.vlist.nv == 32


def test_generator_water32_ubharm():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_ubharm.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    assert part_valence.dlist.ndelta == 32
    for i, j in system.bonds:
        row = part_valence.dlist.lookup.get((i, j))
        assert row is None
    for i, n2s in system.neighs2.items():
        for j in n2s:
            row0 = part_valence.dlist.lookup.get((i, j))
            row1 = part_valence.dlist.lookup.get((j, i))
            assert row0 is not None or row1 is not None
    assert (part_valence.iclist.ictab['kind'] == 5).all()
    assert part_valence.iclist.nic == 32
    assert (part_valence.vlist.vtab['kind'] == 0).all()
    assert abs(part_valence.vlist.vtab['par0'] - 2.5465456475e+02*(kjmol/angstrom**2)).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'] - 2.6123213151e+00*angstrom).max() < 1e-10
    assert part_valence.vlist.nv == 32


def test_generator_water32_cross():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_cross.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    assert part_valence.dlist.ndelta == 64
    for i, j in system.bonds:
        row0 = part_valence.dlist.lookup.get((i, j))
        row1 = part_valence.dlist.lookup.get((j, i))
        assert row0 is not None or row1 is not None
    assert part_valence.iclist.nic == 96
    iclist = part_valence.vlist.iclist
    for row in part_valence.vlist.vtab:
        assert row['kind'] == 3
        ic0 = iclist.ictab[row['ic0']]
        ic1 = iclist.ictab[row['ic1']]
        if ic0['kind'] == 0 and ic1['kind'] == 0:
            assert row['par0'] - 2.0000000000e+01*(kjmol/angstrom**2) < 1e-10
            assert row['par1'] - 0.9470000000e+00*angstrom < 1e-10
            assert row['par2'] - 0.9470000000e+00*angstrom < 1e-10
        elif ic0['kind'] == 0 and ic1['kind'] == 2:
            assert row['par0'] - 1.0000000000e+01*(kjmol/angstrom*rad) < 1e-10
            assert row['par1'] - 0.9470000000e+00*angstrom < 1e-10
            assert row['par2'] - 1.0500000000e+02*deg < 1e-10
        else:
            raise AssertionError('ICs in Cross term should be Bond-Bond or Bond-BendAngle')
    assert part_valence.vlist.nv == 96


def test_generator_glycine_torsion():
    system = get_system_glycine()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_glycine_torsion.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_valence = ff.part_valence
    assert part_valence.vlist.nv == 11
    assert part_valence.dlist.ndelta == 9
    m_counts = {}
    for row in part_valence.vlist.vtab[:11]:
        if row['kind'] == 4:
            key = int(row['par0'])
        elif row['kind'] == 5:
            key = 1
        elif row['kind'] == 6:
            key = 2
        else:
            raise AssertionError
        m_counts[key] = m_counts.get(key, 0) + 1
    assert len(m_counts) == 3
    assert m_counts[1] == 5
    assert m_counts[2] == 2
    assert m_counts[3] == 4


def test_generator_fake_torsion1():
    system = get_system_glycine()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_fake_torsion1.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_valence = ff.part_valence
    assert part_valence.vlist.nv == 12
    m_counts = {}
    for row in part_valence.vlist.vtab[:12]:
        print(row['kind'])
        if row['kind'] == 5:
            key = 1
        elif row['kind'] == 6:
            key = 2
        elif row['kind'] == 7:
            key = 3
        elif row['kind'] == 8:
            key = 4
        elif row['kind'] == 9:
            key = 6
        else:
            raise AssertionError
        m_counts[key] = m_counts.get(key, 0) + 1
    assert len(m_counts) == 5
    assert m_counts[1] == 4
    assert m_counts[2] == 2
    assert m_counts[3] == 4
    assert m_counts[4] == 1
    assert m_counts[6] == 1


def test_generator_fake_torsion2():
    system = get_system_glycine()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_fake_torsion2.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_valence = ff.part_valence
    assert part_valence.vlist.nv == 12
    m_counts = {}
    for row in part_valence.vlist.vtab[:12]:
        print(row['kind'])
        if row['kind'] == 5:
            key = 1
        elif row['kind'] == 6:
            key = 2
        elif row['kind'] == 7:
            key = 3
        elif row['kind'] == 8:
            key = 4
        elif row['kind'] == 9:
            key = 6
        else:
            raise AssertionError
        m_counts[key] = m_counts.get(key, 0) + 1
    assert len(m_counts) == 5
    assert m_counts[1] == 1
    assert m_counts[2] == 1
    assert m_counts[3] == 4
    assert m_counts[4] == 2
    assert m_counts[6] == 4


#def test_generator_water32_bondcross():
#    system = get_system_water32()
#    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_bondcross.txt')
#    ff = ForceField.generate(system, fn_pars)
#    assert len(ff.parts) == 1
#    assert isinstance(ff.parts[0], ForcePartValence)
#    part_valence = ff.part_valence
#    assert part_valence.dlist.ndelta == 64
#    for i, j in system.bonds:
#        row0 = part_valence.dlist.lookup.get((i, j))
#        row1 = part_valence.dlist.lookup.get((j, i))
#        assert row0 is not None or row1 is not None
#    assert (part_valence.iclist.ictab['kind'] == 0).all()
#    assert part_valence.iclist.nic == 64
#    assert (part_valence.vlist.vtab['kind'] == 3).all()
#    assert abs(part_valence.vlist.vtab['par0'] - 1.1354652314e+01*(kjmol/angstrom**2)).max() < 1e-10
#    assert abs(part_valence.vlist.vtab['par1'] - 1.1247753211e+00*angstrom).max() < 1e-10
#    assert abs(part_valence.vlist.vtab['par2'] - 1.1247753211e+00*angstrom).max() < 1e-10
#    assert part_valence.vlist.nv == 32


def test_generator_water32_lj():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_lj.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_lj = ff.part_pair_lj
    # check parameters
    assert abs(part_pair_lj.pair_pot.sigmas[0] - 3.15*angstrom) < 1e-10
    assert abs(part_pair_lj.pair_pot.sigmas[1] - 0.4*angstrom) < 1e-10
    assert abs(part_pair_lj.pair_pot.epsilons[0] - 0.1521*kcalmol) < 1e-10
    assert abs(part_pair_lj.pair_pot.epsilons[1] - 0.046*kcalmol) < 1e-10


def test_generator_glycine_lj():
    system = get_system_glycine()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_fake_lj.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_lj = ff.part_pair_lj
    # check parameters
    assert part_pair_lj.pair_pot.sigmas.shape == (ff.system.natom,)
    assert part_pair_lj.pair_pot.epsilons.shape == (ff.system.natom,)
    assert abs(part_pair_lj.pair_pot.sigmas[0] - 1.7*angstrom) < 1e-10
    assert abs(part_pair_lj.pair_pot.sigmas[1] - 1.8*angstrom) < 1e-10
    assert abs(part_pair_lj.pair_pot.sigmas[3] - 1.6*angstrom) < 1e-10
    assert abs(part_pair_lj.pair_pot.sigmas[5] - 0.5*angstrom) < 1e-10
    assert abs(part_pair_lj.pair_pot.epsilons[0] - 0.18*kcalmol) < 1e-10
    assert abs(part_pair_lj.pair_pot.epsilons[1] - 0.22*kcalmol) < 1e-10
    assert abs(part_pair_lj.pair_pot.epsilons[3] - 0.12*kcalmol) < 1e-10
    assert abs(part_pair_lj.pair_pot.epsilons[5] - 0.05*kcalmol) < 1e-10


def test_generator_water32_mm3():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_mm3.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_mm3 = ff.part_pair_mm3
    # check parameters
    assert abs(part_pair_mm3.pair_pot.sigmas[0] - 1.7*angstrom) < 1e-10
    assert abs(part_pair_mm3.pair_pot.sigmas[1] - 0.2*angstrom) < 1e-10
    assert abs(part_pair_mm3.pair_pot.epsilons[0] - 0.12*kcalmol) < 1e-10
    assert abs(part_pair_mm3.pair_pot.epsilons[1] - 0.04*kcalmol) < 1e-10
    assert part_pair_mm3.pair_pot.onlypaulis[0] == 1
    assert part_pair_mm3.pair_pot.onlypaulis[1] == 0


def test_generator_glycine_mm3():
    system = get_system_glycine()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_fake_mm3.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_mm3 = ff.part_pair_mm3
    # check parameters
    assert part_pair_mm3.pair_pot.sigmas.shape == (ff.system.natom,)
    assert part_pair_mm3.pair_pot.epsilons.shape == (ff.system.natom,)
    assert part_pair_mm3.pair_pot.onlypaulis.shape == (ff.system.natom,)
    assert abs(part_pair_mm3.pair_pot.sigmas[0] - 1.7*angstrom) < 1e-10
    assert abs(part_pair_mm3.pair_pot.sigmas[1] - 1.8*angstrom) < 1e-10
    assert abs(part_pair_mm3.pair_pot.sigmas[3] - 1.6*angstrom) < 1e-10
    assert abs(part_pair_mm3.pair_pot.sigmas[5] - 0.5*angstrom) < 1e-10
    assert abs(part_pair_mm3.pair_pot.epsilons[0] - 0.18*kcalmol) < 1e-10
    assert abs(part_pair_mm3.pair_pot.epsilons[1] - 0.22*kcalmol) < 1e-10
    assert abs(part_pair_mm3.pair_pot.epsilons[3] - 0.12*kcalmol) < 1e-10
    assert abs(part_pair_mm3.pair_pot.epsilons[5] - 0.05*kcalmol) < 1e-10
    assert part_pair_mm3.pair_pot.onlypaulis[0] == 0
    assert part_pair_mm3.pair_pot.onlypaulis[1] == 1
    assert part_pair_mm3.pair_pot.onlypaulis[3] == 1
    assert part_pair_mm3.pair_pot.onlypaulis[5] == 0


def test_generator_water32_exprep1():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_exprep1.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_exprep = ff.part_pair_exprep
    # check parameters
    amp_cross = part_pair_exprep.pair_pot.amp_cross
    assert (amp_cross > 0).all()
    assert (amp_cross == amp_cross.T).all()
    assert abs(amp_cross[0,0] - 4.2117588157e+02) < 1e-10
    assert abs(amp_cross[1,1] - 2.3514195495e+00) < 1e-10
    b_cross = part_pair_exprep.pair_pot.b_cross
    assert (b_cross > 0).all()
    assert (b_cross == b_cross.T).all()
    assert abs(b_cross[0,0] - 4.4661933834e+00/angstrom) < 1e-10
    assert abs(b_cross[1,1] - 4.4107388814e+00/angstrom) < 1e-10


def test_generator_water32_exprep2():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_exprep2.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_exprep = ff.part_pair_exprep
    # check parameters
    amp_cross = part_pair_exprep.pair_pot.amp_cross
    assert (amp_cross > 0).all()
    assert (amp_cross == amp_cross.T).all()
    assert abs(amp_cross[0,0] - 4.2117588157e+02) < 1e-10
    assert abs(amp_cross[1,1] - 2.3514195495e+00) < 1e-10
    b_cross = part_pair_exprep.pair_pot.b_cross
    assert (b_cross > 0).all()
    assert (b_cross == b_cross.T).all()
    assert abs(b_cross[0,0] - 4.4661933834e+00/angstrom) < 1e-10
    assert abs(b_cross[1,1] - 4.4107388814e+00/angstrom) < 1e-10


def test_generator_water32_exprep3():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_exprep3.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_exprep = ff.part_pair_exprep
    # check parameters
    amp_cross = part_pair_exprep.pair_pot.amp_cross
    assert (amp_cross > 0).all()
    assert (amp_cross == amp_cross.T).all()
    assert abs(amp_cross[0,0] - 4.2117588157e+02) < 1e-10
    assert abs(amp_cross[0,1] - 1.4360351514e+01) < 1e-10
    assert abs(amp_cross[1,1] - 2.3514195495e+00) < 1e-10
    b_cross = part_pair_exprep.pair_pot.b_cross
    assert (b_cross > 0).all()
    assert (b_cross == b_cross.T).all()
    assert abs(b_cross[0,0] - 4.4661933834e+00/angstrom) < 1e-10
    assert abs(b_cross[0,1] - 4.0518324069e+00/angstrom) < 1e-10
    assert abs(b_cross[1,1] - 4.4107388814e+00/angstrom) < 1e-10


def test_generator_glycine_exprep1():
    system = get_system_glycine()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_fake_exprep1.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_exprep = ff.part_pair_exprep
    # check parameters
    amp_cross = part_pair_exprep.pair_pot.amp_cross
    assert amp_cross.shape == (4, 4)
    assert (amp_cross > 0).all()
    assert (amp_cross == amp_cross.T).all()
    assert abs(amp_cross[0,0] - 4.9873214987e+00) < 1e-10
    assert abs(amp_cross[1,1] - 4.3843216584e+02) < 1e-10
    assert abs(amp_cross[2,2] - 4.2117588157e+02) < 1e-10
    assert abs(amp_cross[3,3] - 2.9875648798e+00) < 1e-10
    assert abs(amp_cross[1,3] - np.sqrt(4.3843216584e+02*2.9875648798e+00)) < 1e-10
    b_cross = part_pair_exprep.pair_pot.b_cross
    assert b_cross.shape == (4, 4)
    assert (b_cross > 0).all()
    assert (b_cross == b_cross.T).all()
    assert abs(b_cross[0,0] - 4.4265465464e+00/angstrom) < 1e-10
    assert abs(b_cross[1,1] - 4.4132795167e+00/angstrom) < 1e-10
    assert abs(b_cross[2,2] - 4.4654231357e+00/angstrom) < 1e-10
    assert abs(b_cross[3,3] - 4.4371927495e+00/angstrom) < 1e-10
    assert abs(b_cross[2,0] - 0.5*(4.4265465464e+00+4.4654231357e+00)/angstrom) < 1e-10


def test_generator_water32_dampdisp1():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_dampdisp1.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_dampdisp = ff.part_pair_dampdisp
    # check parameters
    c6_cross = part_pair_dampdisp.pair_pot.cn_cross
    assert c6_cross.shape == (2,2)
    assert abs(c6_cross[0,0] - 1.9550248340e+01) < 1e-10
    assert abs(c6_cross[1,1] - 2.7982205915e+00) < 1e-10
    vratio = 3.13071058512e+00/5.13207980365e+00 # v[0]/v[1]
    tmp = 2*c6_cross[0,0]*c6_cross[1,1]/(c6_cross[0,0]/vratio + c6_cross[1,1]*vratio)
    assert abs(c6_cross[0,1] - tmp) < 1e-10
    assert (c6_cross == c6_cross.T).all()
    assert (c6_cross > 0).all()
    b_cross = part_pair_dampdisp.pair_pot.b_cross
    assert b_cross.shape == (2,2)
    assert abs(b_cross[0,0] - 3.2421589363e+00/angstrom) < 1e-10
    assert abs(b_cross[0,1] - 3.3501628381e+00/angstrom) < 1e-10
    assert abs(b_cross[1,1] - 3.4581667399e+00/angstrom) < 1e-10
    assert (b_cross == b_cross.T).all()
    assert (b_cross > 0).all()


def test_generator_water32_dampdisp2():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_dampdisp2.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_dampdisp = ff.part_pair_dampdisp
    # check parameters
    c6_cross = part_pair_dampdisp.pair_pot.cn_cross
    assert abs(c6_cross[0,0] - 1.9550248340e+01) < 1e-10
    assert abs(c6_cross[0,1] - 6.4847208208e+00) < 1e-10
    assert abs(c6_cross[1,1] - 2.7982205915e+00) < 1e-10
    assert (c6_cross == c6_cross.T).all()
    assert (c6_cross > 0).all()
    b_cross = part_pair_dampdisp.pair_pot.b_cross
    assert abs(b_cross[0,0] - 3.2421589363e+00/angstrom) < 1e-10
    assert abs(b_cross[0,1] - 3.3501628381e+00/angstrom) < 1e-10
    assert abs(b_cross[1,1] - 3.4581667399e+00/angstrom) < 1e-10
    assert (b_cross == b_cross.T).all()
    assert (b_cross > 0).all()


def test_generator_glycine_dampdisp1():
    system = get_system_glycine()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_fake_dampdisp1.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    part_pair_dampdisp = ff.part_pair_dampdisp
    # check parameters
    c6_cross = part_pair_dampdisp.pair_pot.cn_cross
    assert c6_cross.shape == (4, 4)
    assert abs(c6_cross[0,0] - 2.0121581791e+01) < 1e-10
    assert abs(c6_cross[1,1] - 2.5121581791e+01) < 1e-10
    assert abs(c6_cross[2,2] - 1.4633211522e+01) < 1e-10
    assert abs(c6_cross[3,3] - 2.4261074778e+00) < 1e-10
    vratio = 3.6001863542e+00/3.7957349423e+00 # v[0]/v[1]
    tmp = 2*c6_cross[0,0]*c6_cross[1,1]/(c6_cross[0,0]/vratio + c6_cross[1,1]*vratio)
    assert abs(c6_cross[0,1] - tmp) < 1e-10

    assert (c6_cross == c6_cross.T).all()
    assert (c6_cross > 0).all()
    b_cross = part_pair_dampdisp.pair_pot.b_cross
    assert b_cross.shape == (4, 4)
    assert abs(b_cross[0,0] - 5.13207980365e+00/angstrom) < 1e-10
    assert abs(b_cross[0,1] - 0.5*(5.13207980365e+00+5.01673173654e+00)/angstrom) < 1e-10
    assert abs(b_cross[1,1] - 5.01673173654e+00/angstrom) < 1e-10
    assert abs(b_cross[2,2] - 5.74321564987e+00/angstrom) < 1e-10
    assert abs(b_cross[3,3] - 3.13071058512e+00/angstrom) < 1e-10
    assert (b_cross == b_cross.T).all()
    assert (b_cross > 0).all()

def test_generator_water32_d3bj():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_fake_d3bj.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    d3bj = ff.part_pair_disp68bjdamp
    gps = d3bj.pair_pot.global_pars
    # check parameters
    c6HH = 1.4633211522e+01
    c6HO = 2.5121581791e+01
    c6OO = 2.4261074778e+00
    c8HH = 5.74321564987e+00
    c8HO = 5.01673173654e+00
    c8OO = 3.13071058512e+00

    c6_cross = d3bj.pair_pot.c6_cross
    assert c6_cross.shape == (2,2)
    assert abs(c6_cross[0,0] - c6HH) < 1e-10
    assert abs(c6_cross[0,1] - c6HO) < 1e-10
    assert abs(c6_cross[1,0] - c6HO) < 1e-10
    assert abs(c6_cross[1,1] - c6OO) < 1e-10

    c8_cross = d3bj.pair_pot.c8_cross
    assert c8_cross.shape == (2,2)
    assert abs(c8_cross[0,0] - c8HH) < 1e-10
    assert abs(c8_cross[0,1] - c8HO) < 1e-10
    assert abs(c8_cross[1,0] - c8HO) < 1e-10
    assert abs(c8_cross[1,1] - c8OO) < 1e-10

    gps = d3bj.pair_pot.global_pars
    assert len(gps) == 4
    assert abs(gps[0] - 1.0) < 1e-10
    assert abs(gps[1] - 2.0) < 1e-10
    assert abs(gps[2] - 3.0) < 1e-10
    assert abs(gps[3] - 4.0) < 1e-10

    R_cross = d3bj.pair_pot.R_cross
    assert R_cross.shape == (2,2)
    assert abs(R_cross[0,0] - np.sqrt(c8HH/c6HH)) < 1e-10
    assert abs(R_cross[0,1] - np.sqrt(c8HO/c6HO)) < 1e-10
    assert abs(R_cross[1,0] - np.sqrt(c8HO/c6HO)) < 1e-10
    assert abs(R_cross[1,1] - np.sqrt(c8OO/c6OO)) < 1e-10

def test_generator_water32_qmdffrep():
    system = get_system_water32()
    print(system.ffatypes)
    print(system.ffatype_ids)
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_fake_qmdffrep.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    qmdffrep = ff.part_pair_qmdffrep
    # check parameters
    A_cross = qmdffrep.pair_pot.amp_cross
    assert A_cross.shape == (2,2)
    print(A_cross)
    assert abs(A_cross[0,0] - 3.2490000000e+01) < 1e-10
    assert abs(A_cross[0,1] - 1.3395000000e+01) < 1e-10
    assert abs(A_cross[1,0] - 1.3395000000e+01) < 1e-10
    assert abs(A_cross[1,1] - 5.5225000000e+00) < 1e-10

    B_cross = qmdffrep.pair_pot.b_cross
    assert B_cross.shape == (2,2)
    assert abs(B_cross[0,0] - 4.08560961303e+00) < 1e-10
    assert abs(B_cross[0,1] - 5.00924416592e+00) < 1e-10
    assert abs(B_cross[1,0] - 5.00924416592e+00) < 1e-10
    assert abs(B_cross[1,1] - 6.34212100945e+00) < 1e-10

    # check scalings
    scalings = qmdffrep.scalings
    assert abs(scalings.scale1 - 0.0) < 1e-10
    assert abs(scalings.scale2 - 0.0) < 1e-10
    assert abs(scalings.scale3 - 0.5) < 1e-10
    assert abs(scalings.scale4 - 0.5) < 1e-10


def test_generator_water32_fixq():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_fixq.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 4
    part_pair_ei = ff.part_pair_ei
    part_ewald_reci = ff.part_ewald_reci
    part_ewald_cor = ff.part_ewald_cor
    part_ewald_neut = ff.part_ewald_neut
    # check part settings
    assert part_pair_ei.pair_pot.alpha > 0
    assert part_pair_ei.pair_pot.alpha == part_ewald_reci.alpha
    assert part_pair_ei.pair_pot.alpha == part_ewald_cor.alpha
    assert part_pair_ei.pair_pot.alpha == part_ewald_neut.alpha
    # check charges and atomic radii
    for i in range(system.natom):
        if system.numbers[i] == 1:
            assert abs(system.charges[i] - 0.417) < 1e-5
            assert abs(system.radii[i] - 1.2*angstrom) < 1e-5
        else:
            assert abs(system.charges[i] + 2*0.417) < 1e-5
            assert abs(system.radii[i] - 1.5*angstrom) < 1e-5

    system = get_system_water32()
    log.set_level(log.silent)
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_fixq.txt')
    ff2 = ForceField.generate(system, fn_pars)
    log.set_level(log.debug)
    # check charges
    for i in range(system.natom):
        if system.numbers[i] == 1:
            assert abs(system.charges[i] - 0.417) < 1e-5
            assert abs(system.radii[i] - 1.2*angstrom) < 1e-5
        else:
            assert abs(system.charges[i] + 2*0.417) < 1e-5
            assert abs(system.radii[i] - 1.5*angstrom) < 1e-5
    energy = ff.compute()
    energy2 = ff2.compute()
    assert abs(energy - energy2) < 1e-3


def test_generator_glycine_fixq():
    system = get_system_glycine()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_glycine_fixq.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1 #Non-periodic, so only one part
    part_pair_ei = ff.part_pair_ei
    # check part settings
    assert part_pair_ei.pair_pot.alpha == 0.0
    # check charges and atomic radii
    ac = {1:0.2, 6:0.5, 7:-1.0, 8:-0.5 } #Charges
    ar = {1:1.2*angstrom, 6: 1.7*angstrom, 7: 1.55*angstrom, 8: 1.50*angstrom} #Radii
    for i in range(system.natom):
        assert abs(system.charges[i] - ac[system.numbers[i]]) < 1e-5
        assert abs(system.radii[i] - ar[system.numbers[i]]) < 1e-5

    system = get_system_glycine()
    log.set_level(log.silent)
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_glycine_fixq.txt')
    ff2 = ForceField.generate(system, fn_pars)
    log.set_level(log.debug)
    # check charges and atomic radii
    ac = {1:0.2, 6:0.5, 7:-1.0, 8:-0.5 } #Charges
    ar = {1:1.2*angstrom, 6: 1.7*angstrom, 7: 1.55*angstrom, 8: 1.50*angstrom} #Radii
    for i in range(system.natom):
        assert abs(system.charges[i] - ac[system.numbers[i]]) < 1e-5
        assert abs(system.radii[i] - ar[system.numbers[i]]) < 1e-5
    energy = ff.compute()
    energy2 = ff2.compute()
    assert abs(energy - energy2) < 1e-3


def test_generator_water32_fixq_dielectric():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_fixq_dielectric.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 4
    part_pair_ei = ff.part_pair_ei
    part_ewald_reci = ff.part_ewald_reci
    part_ewald_cor = ff.part_ewald_cor
    part_ewald_neut = ff.part_ewald_neut
    # check part settings
    print(part_pair_ei.pair_pot.dielectric)
    print(part_ewald_reci.dielectric)
    print(part_ewald_cor.dielectric)
    print(part_ewald_neut.dielectric)
    assert part_pair_ei.pair_pot.dielectric == 1.44
    assert part_pair_ei.pair_pot.dielectric == part_ewald_reci.dielectric
    assert part_pair_ei.pair_pot.dielectric == part_ewald_cor.dielectric
    assert part_pair_ei.pair_pot.dielectric == part_ewald_neut.dielectric


def test_generator_water32():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water.txt')
    ff = ForceField.generate(system, fn_pars)
    # get all ff parts
    assert len(ff.parts) == 7
    part_valence = ff.part_valence
    part_pair_dampdisp = ff.part_pair_dampdisp
    part_pair_exprep = ff.part_pair_exprep
    part_pair_ei = ff.part_pair_ei
    part_ewald_reci = ff.part_ewald_reci
    part_ewald_cor = ff.part_ewald_cor
    part_ewald_neut = ff.part_ewald_neut
    # check dampdisp parameters
    c6_cross = part_pair_dampdisp.pair_pot.cn_cross
    assert abs(c6_cross[0,0] - 1.9550248340e+01) < 1e-10
    assert abs(c6_cross[1,1] - 2.7982205915e+00) < 1e-10
    assert (c6_cross == c6_cross.T).all()
    assert (c6_cross > 0).all()
    b_cross = part_pair_dampdisp.pair_pot.b_cross
    assert abs(b_cross[0,0] - 3.2421589363e+00/angstrom) < 1e-10
    assert abs(b_cross[0,1] - 3.3501628381e+00/angstrom) < 1e-10
    assert abs(b_cross[1,1] - 3.4581667399e+00/angstrom) < 1e-10
    assert (b_cross == b_cross.T).all()
    assert (b_cross > 0).all()
    # check exprep parameters
    amp_cross = part_pair_exprep.pair_pot.amp_cross
    assert (amp_cross > 0).all()
    assert (amp_cross == amp_cross.T).all()
    assert abs(amp_cross[0,0] - 4.2117588157e+02) < 1e-10
    assert abs(amp_cross[1,1] - 2.3514195495e+00) < 1e-10
    b_cross = part_pair_exprep.pair_pot.b_cross
    assert (b_cross > 0).all()
    assert (b_cross == b_cross.T).all()
    assert abs(b_cross[0,0] - 4.4661933834e+00/angstrom) < 1e-10
    assert abs(b_cross[1,1] - 4.4107388814e+00/angstrom) < 1e-10
    # check charges
    for i in range(system.natom):
        if system.numbers[i] == 1:
            assert abs(system.charges[i] - 3.6841957737e-01) < 1e-5
        else:
            assert abs(system.charges[i] + 2*3.6841957737e-01) < 1e-5
    # check valence
    assert part_valence.dlist.ndelta == 64
    assert part_valence.iclist.nic == 96
    assert (part_valence.iclist.ictab['kind'][:64] == 0).all()
    assert (part_valence.iclist.ictab['kind'][64:96] == 1).all()
    assert part_valence.vlist.nv == 96
    assert abs(part_valence.vlist.vtab['par0'][:64] - 4.0088096730e+03*(kjmol/angstrom**2)).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'][:64] - 1.0238240000e+00*angstrom).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par0'][64:96] - 3.0230353700e+02*kjmol).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'][64:96] - np.cos(8.8401698835e+01*deg)).max() < 1e-10


def test_add_part():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_bondharm.txt')
    ff = ForceField.generate(system, fn_pars)
    part_press = ForcePartPressure(system, 1e-3)
    ff.add_part(part_press)
    assert part_press in ff.parts
    assert ff.part_press is part_press
    assert ff.compute() == ff.part_valence.energy + ff.part_press.energy


def test_generator_formaldehyde_oopangle():
    system = get_system_formaldehyde()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_formaldehyde_inversion.txt')
    ff = ForceField.generate(system, fn_pars)
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    assert part_valence.dlist.ndelta == 3
    assert (part_valence.iclist.ictab['kind'][0:3] == 6).all()
    assert part_valence.iclist.nic == 3
    assert (part_valence.vlist.vtab['kind'][0:3] == 5).all()
    assert abs(part_valence.vlist.vtab['par0'] - 1.0*kjmol).all() < 1e-10
    assert part_valence.vlist.nv == 3
