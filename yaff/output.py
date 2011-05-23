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

from molmod.units import *
from molmod.periodic import periodic
import sys

from yaff.ff import *
from yaff.ext import *

header = """******************************************************************************************************************************************************
******************************************************************************************************************************************************
***********************************                                                                               ************************************
**********************************                          **    **   ****    ***  ***                            ***********************************
*********************************                            **  **   ** **   **   **                               **********************************
********************************                              ****   **  **   **   **                                *********************************
********************************                               **   *******  **** ****                               *********************************
********************************                              **   **    **   **   **                                *********************************
*********************************                            **   **     **   **   **                               **********************************
**********************************                          *************** ***  ***                               ***********************************
***********************************                                                                               ************************************
******************************************************************************************************************************************************
******************************************************************************************************************************************************

                                                   Welcome to YAFF - yet another force field code
                                                Developed at the Center for Molecular Modeling (CMM)
                                                           University of Ghent - Belgium
                                                                 (C) Copyright 2011


Execution date:             %s
Machine:                    %s
User:                       %s

Input xyz:                  %s
Input psf:                  %s
Input valence parameters:   %s
"""


tail = """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Terminated at:              %s

******************************************************************************************************************************************************
***********************************                    END OF FILE - THANK YOU, COME AGAIN ...                    ************************************
******************************************************************************************************************************************************"""

def print_block(title,block=None,out=sys.stdout):
    print >> out, "~"*150
    print >> out, " %s:" %title
    print >> out, " "+"-"*(len(title)+1)
    print >> out, ""
    if block is not None:
        print >> out, block
        print >> out, ""


def atom_attr_str(ff):
    """
        Write atom attributes (symbol, fftype, charge, vdW parameters) to string.
    """
    vpart, pair_part_lj, pair_part_ei, ewald_reci_part, ewald_corr_part = ff.parts
    charges = ewald_reci_part.charges
    sigmas = pair_part_lj.pair_pot.sigmas
    epsilons = pair_part_lj.pair_pot.epsilons
    atom_attr = ""
    atom_attr += "     number  atom   type     charge    sigma [A]    epsilon [kcalmol] \n"
    atom_attr += "    ------------------------------------------------------------------\n"
    for i in xrange(ff.system.natom):
        atom_attr += "     %3i      %2s    %4s    % 5.4f     %6.4f          %7.5f   \n" \
            %(i, periodic[ff.system.numbers[i]].symbol, ff.system.ffatypes[i], charges[i], sigmas[i]/angstrom, epsilons[i]/kcalmol)
    return atom_attr


def terms_str(ff):
    """
        Write force field terms to string.
    """
    types = ff.system.ffatypes
    vpart, pair_part_lj, pair_part_ei, ewald_reci_part, ewald_corr_part = ff.parts
    vlist = vpart.vlist.vtab
    iclist = vpart.vlist.iclist.ictab
    dlist = vpart.vlist.iclist.dlist.deltas

    vkinds = ['Harmonic', 'PolyFour', 'Cross']
    eunit = kjmol
    ickinds = ['Bond', 'BendCos', 'BendAngle', 'DihedCos', 'DihedAngle', "UreyBradley"]
    icunits = [angstrom, 1.0, deg, 1.0, deg, angstrom]
    v_descr = ""
    ic0_descr = ""
    ic1_descr = ""

    def get_ic_description(ic):
        indexes = ""
        d0 = dlist[ic['i0']]
        atoms = (d0['i'], d0['j'])
        if ic['kind'] in [1,2,3,4]:
            d1 = dlist[ic['i1']]
            if d1['i']==atoms[0]:
                atoms = (d1['j'], atoms[0], atoms[1])
            elif d1['i']==atoms[1]:
                atoms = (atoms[0], atoms[1], d1['j'])
            elif d1['j']==atoms[0]:
                atoms = (d1['i'], atoms[0], atoms[1])
            elif d1['j']==atoms[1]:
                atoms = (atoms[0], atoms[1], d1['i'])
            else:
                raise ValueError("Error in reading atom triple.")
        if ic['kind'] in [3,4]:
            d2 = dlist[ic['i2']]
            if d2['i']==atoms[0]:
                atoms = (d2['j'], atoms[0], atoms[1], atoms[2])
            elif d2['i']==atoms[2]:
                atoms = (atoms[0], atoms[1], atoms[2], d2['j'])
            elif d2['j']==atoms[0]:
                atoms = (d2['i'], atoms[0], atoms[1], atoms[2])
            elif d2['j']==atoms[2]:
                atoms = (atoms[0], atoms[1], atoms[2], d2['i'])
            else:
                raise ValueError("Error in reading atom quadruple.")
        for atom in atoms:
            indexes += "%4s[%2i], " %(types[atom], atom)
        indexes.rstrip(",")
        return ickinds[ic['kind']], indexes

    def get_par_description(term):
        ic0unit = icunits[iclist[term['ic0']]['kind']]
        if term['kind'] == 0:
            unit_rv = ic0unit
            if iclist[term['ic0']]['kind']==2:
                unit_fc = eunit
            else:
                unit_fc = eunit/ic0unit**2
            return "fc = %7.3f , rv = %7.3f" %(
                term['par0']/unit_fc,
                term['par1']/unit_rv
            )
        elif term['kind'] == 1:
            return "a0 = % 7.3f , a1 = % 7.3f , a2 = % 7.3f , a3 = % 7.3f" %(
                term['par0']/(eunit/ic0unit),
                term['par1']/(eunit/ic0unit**2),
                term['par2']/(eunit/ic0unit**3),
                term['par3']/(eunit/ic0unit**4)
            )
        else:
            raise NotImplementedError()

    terms  = ""
    terms += "       i  |   term   |     ic     |                   atoms                  |        pars [A, deg, kjmol/A^2, kjmol/rad^2, kjmol]              \n"
    terms += "    --------------------------------------------------------------------------------------------------------------------------------------------\n"
    for i in xrange(vpart.vlist.nv):
        term = vlist[i]
        ic0 = iclist[term['ic0']]
        ic0_kind, ic0_indexes = get_ic_description(ic0)
        if term['kind'] < 2:
            par_descr = get_par_description(term)
            terms += "     %3i  | %8s | %10s | %40s |  %s\n" %(i, vkinds[term['kind']], ic0_kind, ic0_indexes, par_descr)
        else:
            raise NotImplementedError()
    return terms


def ff_str(ff):
    """
        Write general force field info to string.
    """
    ff_info = ""
    do_valence = False
    do_ewald = False
    do_ei_real = False
    do_vdw_real = False
    for part in ff.parts:
        if isinstance(part,ValencePart):
            do_valence = True
            vpart = part
        if isinstance(part,EwaldReciprocalPart):
            do_ewald = True
            ewaldpart = part
        if isinstance(part,PairPart):
            if isinstance(part.pair_pot,PairPotEI):
                do_ei_real = True
                eipart = part
            if isinstance(part.pair_pot,PairPotLJ):
                do_vdw_real = True
                vdw_kind = "LJ"
                vdwpart = part
            if isinstance(part.pair_pot,PairPotMM3):
                do_vdw_real = True
                vdw_kind = "MM3"
                vdwpart = part
    if do_valence:
        ff_info += "    Valence interactions: ON\n"
    if do_ei_real:
        ff_info += "    Real electrostatics: ON\n"
        ff_info += "        cutoff      [A] = %9.6f\n" %(eipart.pair_pot.rcut/angstrom)
        ff_info += "        1-2 scaling [ ] = %4.3f\n" %eipart.scalings.scale1
        ff_info += "        1-3 scaling [ ] = %4.3f\n" %eipart.scalings.scale2
        ff_info += "        1-4 scaling [ ] = %4.3f\n" %eipart.scalings.scale3
    else:
        ff_info += "    Real electrostatics: OFF\n"
    if do_ewald:
        ff_info += "    Reciprocal electrostatics: EWALD\n"
        ff_info += "        alpha [1/A] = %9.6f\n" %(ewaldpart.alpha*angstrom)
        ff_info += "        gmax  [   ] = %2i,%2i,%2i\n" %(ewaldpart.gmax[0],ewaldpart.gmax[1],ewaldpart.gmax[2])
    else:
        ff_info += "    Reciprocal electrostatics: OFF\n"
    if do_vdw_real:
        ff_info += "    Real van der Waals: %s\n" %vdw_kind
        ff_info += "        cutoff      [A] = %9.6f\n" %(vdwpart.pair_pot.rcut/angstrom)
        ff_info += "        1-2 scaling [ ] = %4.3f\n" %vdwpart.scalings.scale1
        ff_info += "        1-3 scaling [ ] = %4.3f\n" %vdwpart.scalings.scale2
        ff_info += "        1-4 scaling [ ] = %4.3f\n" %vdwpart.scalings.scale3
    else:
        ff_info += "    Real lennard-jones: OFF\n"
    return ff_info
