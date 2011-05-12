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

from yaff.ext import compute_ewald_reci, compute_ewald_corr
from yaff.dlist import DeltaList
from yaff.iclist import *
from yaff.vlist import *


__all__ = [
    'ForcePart', 'ForceField', 'PairPart', 'EwaldReciprocalPart',
    'EwaldCorrectionPart', 'EwaldNeutralizingPart', 'ValencePart',
    'add_bonds', 'add_bends', 'add_dihedrals',
]


class ForcePart(object):
    def __init__(self, system):
        # backup copies of last call to compute:
        self.energy = 0.0
        self.gpos = np.zeros((system.natom, 3), float)
        self.vtens = np.zeros((3, 3), float)
        self.clear()

    def clear(self):
        # Fill in bogus values that make things crash and burn
        self.energy = np.nan
        self.gpos[:] = np.nan
        self.vtens[:] = np.nan

    def update_rvecs(self, rvecs):
        self.clear()

    def update_pos(self, pos):
        self.clear()

    def compute(self, gpos=None, vtens=None):
        if gpos is None:
            my_gpos = None
        else:
            my_gpos = self.gpos
            my_gpos[:] = 0.0
        if vtens is None:
            my_vtens = None
        else:
            my_vtens = self.vtens
            my_vtens[:] = 0.0
        self.energy = self._internal_compute(my_gpos, my_vtens)
        if gpos is not None:
            gpos += my_gpos
        if vtens is not None:
            vtens += my_vtens
        return self.energy

    def _internal_compute(self, gpos, vtens):
        raise NotImplementedError


class ForceField(ForcePart):
    def __init__(self, system, parts, nlists=None):
        ForcePart.__init__(self, system)
        self.system = system
        self.parts = parts
        self.nlists = nlists
        self.needs_nlists_update = True

    def update_rvecs(self, rvecs):
        ForcePart.update_rvecs(self, rvecs)
        self.system.cell.update_rvecs(rvecs)
        self.needs_nlists_update = True

    def update_pos(self, pos):
        ForcePart.update_pos(self, pos)
        self.system.pos[:] = pos
        self.needs_nlists_update = True

    def _internal_compute(self, gpos, vtens):
        if self.needs_nlists_update:
            self.nlists.update()
            self.needs_nlists_update = False
        return sum([part.compute(gpos, vtens) for part in self.parts])


class PairPart(ForcePart):
    def __init__(self, system, nlists, scalings, pair_pot):
        ForcePart.__init__(self, system)
        self.nlists = nlists
        self.scalings = scalings
        self.pair_pot = pair_pot
        self.nlists.request_rcut(pair_pot.rcut)

    def _internal_compute(self, gpos, vtens):
        assert len(self.nlists) == len(self.scalings)
        result = 0.0
        for i in xrange(len(self.nlists)):
            result += self.pair_pot.compute(i, self.nlists[i], self.scalings[i], gpos, vtens)
        return result


class EwaldReciprocalPart(ForcePart):
    def __init__(self, system, charges, alpha, gcut=0.35):
        ForcePart.__init__(self, system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        self.system = system
        self.charges = charges
        self.alpha = alpha
        self.gcut = gcut
        self.update_gmax()
        self.work = np.empty(system.natom*2)
        self.needs_update_gmax = True

    def update_gmax(self):
        self.gmax = np.ceil(self.gcut/self.system.cell.gspacings-0.5).astype(int)

    def update_rvecs(self, rvecs):
        ForcePart.update_rvecs(self, rvecs)
        self.update_gmax()

    def _internal_compute(self, gpos, vtens):
        energy = compute_ewald_reci(
            self.system.pos, self.charges, self.system.cell, self.alpha,
            self.gmax, self.gcut, gpos, self.work, vtens
        )
        return energy


class EwaldCorrectionPart(ForcePart):
    def __init__(self, system, charges, alpha, scalings):
        ForcePart.__init__(self, system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        self.system = system
        self.charges = charges
        self.alpha = alpha
        self.scalings = scalings

    def _internal_compute(self, gpos, vtens):
        return sum([
            compute_ewald_corr(self.system.pos, i, self.charges,
                               self.system.cell, self.alpha, self.scalings[i],
                               gpos, vtens)
            for i in xrange(len(self.scalings))
        ])


class EwaldNeutralizingPart(ForcePart):
    def __init__(self, system, charges, alpha):
        ForcePart.__init__(self, system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        self.system = system
        self.charges = charges
        self.alpha = alpha

    def _internal_compute(self, gpos, vtens):
        fac = self.charges.sum()**2*np.pi/(2.0*self.system.cell.volume*self.alpha**2)
        if vtens is not None:
            vtens.ravel()[::4] -= fac
        return fac


class ValencePart(ForcePart):
    def __init__(self, system):
        ForcePart.__init__(self, system)
        self.dlist = DeltaList(system)
        self.iclist = InternalCoordinateList(self.dlist)
        self.vlist = ValenceList(self.iclist)

    def add_term(self, term):
        self.vlist.add_term(term)

    def _internal_compute(self, gpos, vtens):
        self.dlist.forward()
        self.iclist.forward()
        energy = self.vlist.forward()
        if not ((gpos is None) and (vtens is None)):
            self.vlist.back()
            self.iclist.back()
            self.dlist.back(gpos, vtens)
        return energy


def add_bonds(system, vpart, val_table):
    """
        Add bonds present in system to vpart (a ValencePart instance) with parameters form the val_table dictionairy
        The method returns 2 strings:
            
            warnings = list of warnings (missing terms)
            added    = list of added terms
    """
    warnings = ""
    added = ""
    for i, j in system.topology.bonds:
        key = system.ffatypes[i], system.ffatypes[j]
        if not key in val_table.keys():
            warnings += "    no term detected for atom pair %i,%i %s\n" %(i,j,key)
        else:
            terminfo = val_table[key]
            if terminfo[0]=="harm" and terminfo[1]=="dist":
                if terminfo[2][0][0]=="K" and terminfo[2][1][0]=="q0": 
                    fc = terminfo[2][0][1]
                    rv = terminfo[2][1][1]
                elif terminfo[2][1][0]=="K" and terminfo[2][0][0]=="q0": 
                    fc = terminfo[2][1][1]
                    rv = terminfo[2][0][1]
                else:
                    raise ValueError("Error reading parameters in bond term")
                vpart.add_term(Harmonic(fc, rv, Bond(i, j)))
                added += "    Bond    :  Harmonic(%s[%i] - %s[%i])\n" %(key[0], i, key[1], j)
            else:
                raise NotImplementedError("Bond term of kind %s(%s) not supported" %(terminfo[0], terminfo[1]) )
    return added, warnings


def add_bends(system, vpart, val_table):
    """
        Add bends present in system to vpart (a ValencePart instance) with parameters form the val_table dictionairy
        The method returns 2 strings:
            
            warnings = list of warnings (missing terms)
            added    = list of added terms
    """
    warnings = ""
    added = ""
    for i1 in xrange(system.natom):
        for i0 in system.topology.neighs1[i1]:
            for i2 in system.topology.neighs1[i1]:
                key = system.ffatypes[i0], system.ffatypes[i1], system.ffatypes[i2]
                if i0 > i2:
                    if not key in val_table.keys():
                        warnings += "    no term detected for atom triple %i,%i,%i %s\n" %(i0,i1,i2,key)
                    else:
                        terminfo = val_table[key]
                        if terminfo[0]=="harm" and terminfo[1]=="angle":
                            if terminfo[2][0][0]=="K" and terminfo[2][1][0]=="q0": 
                                fc = terminfo[2][0][1]
                                rv = terminfo[2][1][1]
                            elif terminfo[2][1][0]=="K" and terminfo[2][0][0]=="q0": 
                                fc = terminfo[2][1][1]
                                rv = terminfo[2][0][1]
                            else:
                                raise ValueError("Error reading parameters in bond term")
                            vpart.add_term(Harmonic(fc, rv, BendAngle(i0, i1, i2)))
                            added += "    Bend    :  Harmonic(%s[%i] - %s[%i] - %s[%i])\n" %(key[0], i0, key[1], i1, key[2], i2)
                        elif terminfo[0]=="sqbend" and terminfo[1]=="cangle":
                            if terminfo[2][0][0]=="K":
                                fc = terminfo[2][0][1]
                            else:
                                raise ValueError("Error reading parameters in bend term")
                            vpart.add_term(PolyFour([0, fc, fc, 0], BendCos(i0, i1, i2)))
                            added += "    Bend    :  PolyFour(%s[%i] - %s[%i] - %s[%i])\n" %(key[0], i0, key[1], i1, key[2], i2)
                        else:
                            raise NotImplementedError("Bend term of kind %s(%s) not supported" %(terminfo[0], terminfo[1]) )
    return added, warnings


def add_dihedrals(system, vpart, val_table, only_dihedral_number=None):
    """
        Add dihedrals present in system to vpart (a ValencePart instance) with parameters form the val_table dictionairy
        The method returns 2 strings:
            
            warnings = list of warnings (missing terms)
            added    = list of added terms
    """
    warnings = ""
    added = ""
    idih = -1
    for i1, i2 in system.topology.bonds:
        for i0 in system.topology.neighs1[i1]:
            if i0==i2: continue
            for i3 in system.topology.neighs1[i2]:
                if i3==i1: continue
                key = system.ffatypes[i0], system.ffatypes[i1], system.ffatypes[i2], system.ffatypes[i3]
                if not key in val_table.keys():
                    warnings += "    no term detected for atom quadruplet %i,%i,%i, %i %s\n" %(i0, i1, i2, i3, key)
                else:
                    idih += 1
                    if idih==only_dihedral_number or only_dihedral_number is None:
                        terminfo = val_table[key]
                        if terminfo[0]=="cos-m2-0" and terminfo[1]=="dihed":
                            if terminfo[2][0][0]=="K":
                                fc = terminfo[2][0][1]
                            else:
                                raise ValueError("Error reading parameters in dihed term")
                            vpart.add_term(PolyFour([0.0, -2*fc, 0.0, 0.0], DihedCos(i0, i1, i2, i3)))
                            added += "    Dihedral:  PolyFour(%s[%i] - %s[%i] - %s[%i] - %s[%i])\n" %(key[0], i0, key[1], i1, key[2], i2, key[3], i3)
                        elif terminfo[0]=="cos-m1-0" and terminfo[1]=="dihed":
                            if terminfo[2][0][0]=="K":
                                fc = terminfo[2][0][1]
                            else:
                                raise ValueError("Error reading parameters in dihed term")
                            vpart.add_term(PolyFour([-fc_table[key], 0.0, 0.0, 0.0], DihedCos(i0, i1, i2, i3)))
                            added += "    Dihedral:  PolyFour(%s-%s-%s-%s)\n" %(key[0], key[1], key[2], key[3])
                        else:
                            raise NotImplementedError("Dihedral term of kind %s(%s) not supported" %(terminfo[0], terminfo[1]) )
    return added, warnings
