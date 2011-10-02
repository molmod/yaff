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


import numpy as np

from yaff import kjmol, angstrom, deg
from yaff import ForceField, ForcePartValence

from common import get_system_water32


def test_ffgen_water32_bondharm():
    system = get_system_water32()
    from StringIO import StringIO
    f = StringIO()
    print >> f, 'BONDHARM:UNIT K kjmol/angstrom**2'
    print >> f, 'BONDHARM:UNIT R0 angstrom'
    print >> f, '# Comment'
    print >> f
    print >> f, 'BONDHARM:PARS O H  4.0088096730e+03  1.0238240000e+00'
    f.seek(0)
    ff = ForceField.generate(system, f)
    f.close()
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    assert part_valence.dlist.ndelta == 64
    for i, j in system.topology.bonds:
        row = part_valence.dlist.lookup.get((i, j))
        assert row is not None
    assert (part_valence.iclist.ictab['kind'] == 0).all()
    assert part_valence.iclist.nic == 64
    assert (part_valence.vlist.vtab['kind'] == 0).all()
    assert abs(part_valence.vlist.vtab['par0'] - 4.0088096730e+03*(kjmol/angstrom**2)).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'] - 1.0238240000e+00*angstrom).max() < 1e-10
    assert part_valence.vlist.nv == 64


def test_ffgen_water32_bondfues():
    system = get_system_water32()
    from StringIO import StringIO
    f = StringIO()
    print >> f, 'BONDFUES:UNIT K kjmol/angstrom**2'
    print >> f, 'BONDFUES:UNIT R0 angstrom'
    print >> f, '# Comment'
    print >> f
    print >> f, 'BONDFUES:PARS O H  4.0088096730e+03  1.0238240000e+00'
    f.seek(0)
    ff = ForceField.generate(system, f)
    f.close()
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    assert part_valence.dlist.ndelta == 64
    for i, j in system.topology.bonds:
        row = part_valence.dlist.lookup.get((i, j))
        assert row is not None
    assert (part_valence.iclist.ictab['kind'] == 0).all()
    assert part_valence.iclist.nic == 64
    assert (part_valence.vlist.vtab['kind'] == 2).all()
    assert abs(part_valence.vlist.vtab['par0'] - 4.0088096730e+03*(kjmol/angstrom**2)).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'] - 1.0238240000e+00*angstrom).max() < 1e-10
    assert part_valence.vlist.nv == 64


def test_ffgen_water32_bendaharm():
    system = get_system_water32()
    from StringIO import StringIO
    f = StringIO()
    print >> f, 'BENDAHARM:UNIT K kjmol/rad**2'
    print >> f, 'BENDAHARM:UNIT THETA0 deg'
    print >> f, '# Comment'
    print >> f
    print >> f, 'BENDAHARM:PARS H O H 4.0088096730e+02  1.0238240000e+02'
    f.seek(0)
    ff = ForceField.generate(system, f)
    f.close()
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    print part_valence.dlist.ndelta
    assert part_valence.dlist.ndelta == 64
    for i, j in system.topology.bonds:
        row = part_valence.dlist.lookup.get((i, j))
        assert row is not None
    assert (part_valence.iclist.ictab['kind'] == 2).all()
    assert part_valence.iclist.nic == 32
    assert (part_valence.vlist.vtab['kind'] == 0).all()
    assert abs(part_valence.vlist.vtab['par0'] - 4.0088096730e+02*kjmol).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'] - 1.0238240000e+02*deg).max() < 1e-10
    assert part_valence.vlist.nv == 32


def test_ffgen_water32_bendcharm():
    system = get_system_water32()
    from StringIO import StringIO
    f = StringIO()
    print >> f, 'BENDCHARM:UNIT K kjmol'
    print >> f, 'BENDCHARM:UNIT THETA0 deg'
    print >> f, '# Comment'
    print >> f
    print >> f, 'BENDCHARM:PARS H O H 4.0088096730e+02  1.0238240000e+02'
    f.seek(0)
    ff = ForceField.generate(system, f)
    f.close()
    assert len(ff.parts) == 1
    assert isinstance(ff.parts[0], ForcePartValence)
    part_valence = ff.parts[0]
    print part_valence.dlist.ndelta
    assert part_valence.dlist.ndelta == 64
    for i, j in system.topology.bonds:
        row = part_valence.dlist.lookup.get((i, j))
        assert row is not None
    assert (part_valence.iclist.ictab['kind'] == 1).all()
    assert part_valence.iclist.nic == 32
    assert (part_valence.vlist.vtab['kind'] == 0).all()
    assert abs(part_valence.vlist.vtab['par0'] - 4.0088096730e+02*kjmol).max() < 1e-10
    assert abs(part_valence.vlist.vtab['par1'] - np.cos(1.0238240000e+02*deg)).max() < 1e-10
    assert part_valence.vlist.nv == 32
