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
'''liblammps

   This module provides an interface to LAMMPS used as a shared library.
   Its intended use is to outsource the calculation of the noncovalent part of
   the force field (which is not implemented very efficient in YAFF), to LAMMPS.
'''


import numpy as np
import os
import ctypes

from molmod.units import pascal, angstrom, kjmol, kcalmol
from molmod.constants import boltzmann

from yaff.log import log, timer
from yaff.external.lammpsio import *
from yaff.pes import *
from yaff.sampling.utils import cell_lower

__all__ = ['ForcePartLammps','swap_noncovalent_lammps']

class ForcePartLammps(ForcePart):
    '''Covalent energies obtained from Lammps.'''
    def __init__(self, ff, fn_system, fn_log="none", suffix='',
                    do_table=True, fn_table='lammps.table', scalings_table=[0.0,0.0,1.0],
                    do_ei=True, kspace='ewald', kspace_accuracy=1e-7, scalings_ei=[0.0,0.0,1.0],
                    triclinic=True, comm=None, move_central_cell=False):
        r'''Initalize LAMMPS ForcePart

           **Arguments:**

           system
                An instance of the ``System`` class.

           fn_system
                The file containing the system data in LAMMPS format, can be
                generated using external.lammpsio.write_lammps_system_data

           **Optional Arguments:**

           fn_log
                Filename where LAMMPS output is stored. This is probably only
                necessary for debugging. Default: None, which means no output
                is stored

           suffix
                The suffix of the liblammps_*.so library file

           do_table
                Boolean, compute a potentual using tabulated values

           fn_table
                Filename of file containing tabulated non-bonded potential
                without point-charge electrostatics.
                Can be written using the ```write_lammps_table``` method.
                Default: lammps.table

           scalings_vdw
                List containing vdW scaling factors for 1-2, 1-3 and 1-4
                neighbors

           do_ei
                Boolean, compute a point-charge electrostatic contribution

           kspace
                Method to treat long-range electrostatics, should be one of
                ewald or pppm

           kspace_accuracy
                Desired relative error in electrostatic forces
                Default: 1e-6

           scalings_ei
                List containing electrostatic scaling factors for 1-2, 1-3 and
                1-4 neighbors

           triclinic
                Boolean, specify whether a triclinic cell will be used during
                the simulation. If the cell is orthogonal, set it to False
                as LAMMPS should run slightly faster.
                Default: True

           comm
                MPI communicator, required if LAMMPS should run in parallel

           move_central_cell
                Boolean, if True, every atom is moved to the central cell
                (centered at the origin) before passing positions to LAMMPS.
                Change this if LAMMPS gives a Atoms Lost ERROR.
        '''
        self.system = ff.system
        # Try to load the lammps package, quit if not possible
        try:
            from lammps import lammps
        except:
            log("Could not import the lammps python package which is required to use LAMMPS as a library")
            raise ImportError
        # Some safety checks...
        if self.system.cell.nvec != 3:
            raise ValueError('The system must be 3D periodic for LAMMPS calculations.')
        if not os.path.isfile(fn_system):
            raise ValueError('Could not read file %s' % fn_system)
        if do_table:
            if not os.path.isfile(fn_table):
                raise ValueError('Could not read file %s' % fn_table)
            tables = read_lammps_table(fn_table)
            table_names = [table[0] for table in tables]
            npoints,_,rcut = tables[0][1]
        elif do_ei:
            rcut = 0
            for part in ff.parts:
                if part.name=='pair_ei':
                    rcut = part.pair_pot.rcut
            if rcut==0:
                log("ERROR, do_ei set to True, but pair_ei was not found in the ff")
        else: raise NotImplementedError
        if not kspace in ['ewald','pppm']:
            raise ValueError('kspace should be one of ewald or pppm')
        if self.system.natom>2000 and kspace=='ewald' and log.do_warning:
            log.warn("You are simulating a more or less large system."
                     "It might be more efficient to use kspace=pppm")
        if self.system.natom<1000 and kspace=='pppm' and log.do_warning:
            log.warn("You are simulating a more or less small system."
                     "It might be better to use kspace=ewald")

        # Initialize a class instance and some attributes
        ForcePart.__init__(self, 'lammps', self.system)
        self.comm = comm
        self.triclinic = triclinic
        self.move_central_cell = move_central_cell
        ffatypes, ffatype_ids = get_lammps_ffatypes(ff)
        nffa = len(ffatypes)

        # Pass all commands that would normally appear in the LAMMPS input file
        # to our instance of LAMMPS.
        self.lammps = lammps(name=suffix, comm=self.comm, cmdargs=["-screen",fn_log,"-log","none"])
        self.lammps.command("units electron")
        self.lammps.command("atom_style full")
        self.lammps.command("atom_modify map array")
        self.lammps.command("box tilt large")
        self.lammps.command("read_data %s"%fn_system)
        self.lammps.command("mass * 1.0")
        self.lammps.command("bond_style none")

        # Hybrid style combining electrostatics and table
        if do_ei and do_table:
            self.lammps.command("pair_style hybrid/overlay coul/long %13.8f table spline %d"%(rcut,npoints))
            self.lammps.command("pair_coeff * * coul/long")
            self.lammps.command("kspace_style %s %10.5e" % (kspace,kspace_accuracy))
        # Only electrostatics
        elif do_ei:
            self.lammps.command("pair_style coul/long %13.8f"%rcut)
            self.lammps.command("pair_coeff * *")
            self.lammps.command("kspace_style %s %10.5e" % (kspace,kspace_accuracy))
        # Only table
        elif do_table:
            self.lammps.command("pair_style table spline %d"%(npoints))
        else:
            raise NotImplementedError
        for i in range(nffa):
            ffai = ffatypes[i]
            for j in range(i,nffa):
                ffaj = ffatypes[j]
                if ffai>ffaj:
                    name = '%s---%s' % (str(ffai),str(ffaj))
                else:
                    name = '%s---%s' % (str(ffaj),str(ffai))
                if do_ei and do_table:
                    self.lammps.command("pair_coeff %d %d table %s %s" % (i+1,j+1,fn_table,name))
                elif do_table:
                    self.lammps.command("pair_coeff %d %d %s %s" % (i+1,j+1,fn_table,name))
        if do_ei is not None:
            self.lammps.command("special_bonds lj %f %f %f coul %f %f %f" %
                 (scalings_table[0],scalings_table[1],scalings_table[2],scalings_ei[0],scalings_ei[1],scalings_ei[2]))
        else:
            self.lammps.command("special_bonds lj %f %f %f" %
                 (scalings_table[0],scalings_table[1],scalings_table[2]))
        self.lammps.command("neighbor 0.0 bin")
        self.lammps.command("neigh_modify delay 0 every 1 check no")
        self.lammps.command("compute virial all pressure NULL virial")
        self.lammps.command("variable eng equal pe")
        self.lammps.command("variable Wxx equal c_virial[1]")
        self.lammps.command("variable Wyy equal c_virial[2]")
        self.lammps.command("variable Wzz equal c_virial[3]")
        self.lammps.command("variable Wxy equal c_virial[4]")
        self.lammps.command("variable Wxz equal c_virial[5]")
        self.lammps.command("variable Wyz equal c_virial[6]")
        thermo_style = "step time etotal evdwl ecoul elong etail "
        thermo_style += " ".join(["c_virial[%d]"%i for i in range(1,7)])
        self.lammps.command("thermo_style custom %s"%thermo_style)
        self.lammps.command("fix 1 all nve")
        # LAMMPS needs cell vectors (ax,0,0), (bx,by,0) and (cx,cy,cz)
        # This means we need to perform a rotation to switch between Yaff and
        # LAMMPS coordinates. All information about this rotation is stored
        # in the variables defined below
        self.rvecs = np.eye(3)
        self.cell = Cell(self.rvecs)
        self.rot = np.zeros((3,3))
        self.vtens_lammps = np.zeros((6,))

    def update_pos(self, pos):
        '''
        Update the LAMMPS positions based on the coordinates from Yaff
        '''
        # Perform the rotation
        pos[:] = np.einsum('ij,kj', pos, self.rot)
        if self.move_central_cell:
            for i in xrange(self.system.natom):
                self.cell.mic(pos[i])
        self.lammps.scatter_atoms("x",1,3,ctypes.c_void_p(pos.ctypes.data))

    def update_rvecs(self, rvecs):
        # Find cell vectors in LAMMPS format
        self.rvecs[:], self.rot[:] = cell_lower(rvecs)
        self.cell.update_rvecs(self.rvecs)
        if self.triclinic:
            self.lammps.command("change_box all x final %f %30.20f y final %f %30.20f z final %f %30.20f xy final %30.20f xz final %30.20f yz final %30.20f\n" %
                (0.0,self.rvecs[0,0],0.0,self.rvecs[1,1], 0.0, self.rvecs[2,2], self.rvecs[1,0],self.rvecs[2,0],self.rvecs[2,1]))
        else:
            self.lammps.command("change_box all x final %f %30.20f y final %f %30.20f z final %f %30.20f\n" %
                (0.0,self.rvecs[0,0],0.0,self.rvecs[1,1], 0.0, self.rvecs[2,2]))

    def _internal_compute(self, gpos, vtens):
        with timer.section("LAMMPS overhead"):
            self.update_rvecs(self.system.cell.rvecs)
            self.update_pos(self.system.pos.copy())
        with timer.section("LAMMPS"):
            self.lammps.command("run 0 post no")
        with timer.section("LAMMPS overhead"):
            energy = self.lammps.extract_variable("eng",None,0)
            if gpos is not None:
                f = self.lammps.gather_atoms("f",1,3)
                gpos[:] = np.ctypeslib.as_array(f).reshape((-1,3))
                gpos[:] = -np.einsum('ij,kj', gpos, self.rot.transpose())
            if vtens is not None:
                self.vtens_lammps[0] = self.lammps.extract_variable("Wxx",None,0)
                self.vtens_lammps[1] = self.lammps.extract_variable("Wyy",None,0)
                self.vtens_lammps[2] = self.lammps.extract_variable("Wzz",None,0)
                self.vtens_lammps[3] = self.lammps.extract_variable("Wxy",None,0)
                self.vtens_lammps[4] = self.lammps.extract_variable("Wxz",None,0)
                self.vtens_lammps[5] = self.lammps.extract_variable("Wyz",None,0)
                # Lammps gives the virial per volume in pascal, so we have to
                # multiply with some prefactors
                self.vtens_lammps[:] *= -pascal*self.system.cell.volume
                # The [6x1] vector has to be cast to a symmetric [3x3] tensor
                # Lammps orders the values as [xx,yy,zz,xy,xz,yz]
                vtens[np.triu_indices(3)] = self.vtens_lammps[[0,3,4,1,5,2]]
                vtens[np.tril_indices(3)] = self.vtens_lammps[[0,3,1,4,5,2]]
                # Finally we have to compute the effect of the rotation on the
                # the virial tensor to get the values in Yaff coordinates
                vtens[:] = np.dot(self.rot.transpose(),np.dot(vtens[:],self.rot))
        return energy


def swap_noncovalent_lammps(ff, **kwargs):
    r'''Take a YAFF ForceField instance and replace noncovalent interactions with
    a ForcePartLammps instance.

           **Arguments:**

           ff
                a YAFF ForceField instance

           **Optional arguments:**

           fn_system
                The filename to which the system data in LAMMPS format will be
                written.

           overwrite_table
                whether or not fn_table should be updated if it already exists.
                Default is ``False``

           nrows
                The number of rows to use for tabulating noncovalent
                interactions. Default is 5000

           keep_forceparts
                classes of ForceParts present in this list will be retained
                in the YAFF ForceField and will not be included in the
                tabulated interactions. Typically this will be covalent
                interactions, but also for instance analytical tail
                corrections. Default:[ForcePartTailCorrection,ForcePartGrid,
                ForcePartPressure,ForcePartValence]

           Any optional argument in the constructor of the
                :class:`yaff.external.liblammps.ForcePartLammps` class,
                can be passed here as well.
    '''
    # Some keyword arguments that should not be passed to the constructor of
    # ForcePartLammps
    overwrite_table = kwargs.pop("overwrite_table", False)
    nrows = kwargs.pop("nrows", 5000)
    keep_forceparts = kwargs.pop("keep_forceparts", [ForcePartTailCorrection,
        ForcePartGrid, ForcePartPressure, ForcePartValence])
    fn_system = kwargs.pop("fn_system", "system.dat")
    # Some other keywords that should be passed to ForcePartLammps
    comm = kwargs.get("comm", None)
    fn_table = kwargs.get("fn_table", "lammps.table")
    triclinic = kwargs.get("triclinic", True)
    # Find out which parts need to be retained, and which ones need to be tabulated
    parts, parts_tabulated = [],[]
    scaling_rules = [1.0,1.0,1.0,1.0]
    correct_15_rule = False
    do_table, do_ei = False, False
    for part in ff.parts:
        if part.__class__ in keep_forceparts:
            parts.append(part)
        else:
            parts_tabulated.append(part)
        # LAMMPS will use the ``strongest'' scaling rules, i.e. following the
        # part that sets most nearest neighbors interactions to zero.
        # Corrections for this are applied afterwards
        if part.__class__==ForcePartPair:
            if part.scalings.scale1!=1.0: scaling_rules[0]=0.0
            if part.scalings.scale2!=1.0: scaling_rules[1]=0.0
            if part.scalings.scale3!=1.0: scaling_rules[2]=0.0
            if part.scalings.scale4!=1.0:
                correct_15_rule = True
            # Check whether electrostatics and/or tables need to be included by LAMMPS
            if part.name=='pair_ei':
                do_ei = True
                # Corrections to electrostatic interactions for Gaussian
                # are included in the table
                if np.any(part.pair_pot.radii!=0.0): do_table=True
            else: do_table = True
    # Generate tables (if necessary) for force field containing relevant parts
    if not os.path.isfile(fn_table) or overwrite_table:
        # Make sure that at most one process actually writes the table
        if comm is None or comm.Get_rank()==0:
            ff_tabulate = ForceField(ff.system, parts_tabulated, nlist=ff.nlist)
            write_lammps_table(ff_tabulate,fn=fn_table, nrows=nrows)
        # Let all processes wait untill the table is completely written
        if comm is not None: comm.Barrier()
    # Write system data
    # Make sure that at most one process writes the data file
    if comm is None or comm.Get_rank()==0:
        write_lammps_system_data(ff.system, ff=ff, fn=fn_system, triclinic=triclinic)
    # Let all processes wait untill the data file is completely written
    if comm is not None: comm.Barrier()
    # Adapt optional arguments
    kwargs["scalings_ei"] = np.array(scaling_rules)
    kwargs["do_ei"] = do_ei
    kwargs["scalings_table"] = np.array(scaling_rules)
    kwargs["do_table"] = do_table
    # Get the ForcePartLammps, which will handle noncovalent interactions
    part_lammps = ForcePartLammps(ff, fn_system, **kwargs)
    parts.append(part_lammps)
    # Potentially add additional parts which correct scaling rules
    scaling_nlist = BondedNeighborList(ff.system,selected=[],add15=correct_15_rule)
    nlist = None
    for part in ff.parts:
        if part.__class__==ForcePartPair:
            part_scalings = np.array([part.scalings.scale1,part.scalings.scale2,part.scalings.scale3,part.scalings.scale4])
            if np.any(part_scalings!=scaling_rules):
                scale4 = part_scalings[3]-1.0 if correct_15_rule else 1.0
                correction_scalings = Scalings(ff.system,scale1=part_scalings[0]-scaling_rules[0],
                    scale2=part_scalings[1]-scaling_rules[1],scale3=part_scalings[2]-scaling_rules[2],scale4=scale4)
                # Electrostatics: special case, because now alpha should be zero
                if part.name=='pair_ei':
                    pair_correction = PairPotEI(ff.system.charges, 0.0, rcut=part.pair_pot.rcut,
                        tr=part.pair_pot.get_truncation(), dielectric=part.pair_pot.dielectric,
                        radii=part.pair_pot.radii.copy())
                    part_correction = ForcePartPair(ff.system, scaling_nlist, correction_scalings, pair_correction)
                else:
                    part_correction = ForcePartPair(ff.system, scaling_nlist, correction_scalings, part.pair_pot)
                parts.append(part_correction)
                nlist = scaling_nlist
    # Construct the new force field
    ff_lammps = ForceField(ff.system, parts, nlist=nlist)
    return ff_lammps
