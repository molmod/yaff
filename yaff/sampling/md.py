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

from molmod.units import *
from molmod.constants import boltzmann
from molmod.io.xyz import XYZWriter
from molmod.periodic import periodic

from yaff.pes.ff import ForcePartPair
from yaff.pes.ext import PairPotEI

__all__ = [
    'MolecularDynamics', 'NVE',
]

class MolecularDynamics(object):
    def __init__(self, ff, timestep=1.0*femtosecond, nsteps=100,
                 out=None, xyz_writer=None, dipole_writer=None, time_writer=None,
                 energy_writer=None, potential_writer=None, kinetic_writer=None, temperature_writer=None):
        self.ff = ff
        self.timestep = timestep
        self.nsteps = nsteps
        self.natom  = ff.system.natom
        self.masses  = np.array([ periodic[i].mass for i in self.ff.system.numbers ]).reshape((-1,1))
        self.pos     = np.zeros([self.natom, 3], float)
        self.prevpos = np.zeros([self.natom, 3], float)
        self.vel     = np.zeros([self.natom, 3], float)
        self.gpos    = np.zeros([self.natom, 3], float)
        self.energy  = None

        if out is not None :
            if not isinstance(out, file):
                raise TypeError("out should be of type file")
        else:
            raise TypeError("No output file defined")
        self.out = out

        if xyz_writer is not None and not isinstance(xyz_writer, XYZWriter):
            raise TypeError("xyz_writer should be of type XYZWriter")
        self.xyz_writer = xyz_writer

        if dipole_writer is not None and not isinstance(dipole_writer, file):
            raise TypeError("dipole_writer should be of type file")
        self.dipole_writer = dipole_writer

        if time_writer is not None and not isinstance(time_writer, file):
            raise TypeError("time_writer should be of type file")
        self.time_writer = time_writer

        if energy_writer is not None and not isinstance(energy_writer, file):
            raise TypeError("enery_writer should be of type file")
        self.energy_writer = energy_writer

        if kinetic_writer is not None and not isinstance(kinetic_writer, file):
            raise TypeError("kinetic_writer should be of type file")
        self.kinetic_writer = kinetic_writer

        if potential_writer is not None and not isinstance(potential_writer, file):
            raise TypeError("potential_writer should be of type file")
        self.potential_writer = potential_writer

        if temperature_writer is not None and not isinstance(temperature_writer, file):
            raise TypeError("temperature_writer should be of type file")
        self.temperature_writer = temperature_writer

        print >> self.out, "~"*150
        print >> self.out, ""
        print >> self.out, "Molecular Dynamics simulation:"
        print >> self.out, "------------------------------"
        print >> self.out, ""
        print >> self.out, "   Timestep |        Energy [kcalmol]          |     Temperature [K]  "
        print >> self.out, "  --------------------------------------------------------------------"


    def get_inst_temp(self):
        return (self.masses*self.vel**2).sum()/(3*self.natom*boltzmann)


    def get_kinetic_energy(self):
        return  (0.5*self.masses*self.vel**2).sum()


    def get_potential_energy(self):
        return self.energy


    def get_total_energy(self):
        return self.energy + self.get_kinetic_energy()


    def get_dipole(self):
        for part in self.ff.parts:
            if isinstance(part, ForcePartPair):
                if isinstance(part.pair_pot, PairPotEI):
                    charges = (part.pair_pot.charges).reshape((-1,1))
        dipole       = sum(charges*self.pos)
        dipole_deriv = sum(charges*self.vel)
        return dipole, dipole_deriv


    def run(self):
        raise NotImplementedError


    def write_output(self, i):
        if i==-1:
            print >> self.out, "      init  |  %30.10f  |  %15.10f" %(self.energy/kcalmol, self.get_inst_temp()/kelvin)
        else:
            print >> self.out, "   %7i  |  %30.10f  |  %15.10f" %(i, self.get_total_energy()/kcalmol, self.get_inst_temp()/kelvin)
            if self.energy_writer is not None:
                self.energy_writer.write("%15.10f\n" %(self.get_total_energy()))

            if self.time_writer is not None:
                self.time_writer.write("%15.10f\n" %(i*self.timestep))

            if self.kinetic_writer is not None:
                self.kinetic_writer.write("%15.10f\n" %(self.get_kinetic_energy()))

            if self.potential_writer is not None:
                self.potential_writer.write("%15.10f\n" %(self.get_potential_energy()))

            if self.temperature_writer is not None:
                self.temperature_writer.write("%15.10f\n" %(self.get_inst_temp()))

            if self.xyz_writer is not None:
                self.xyz_writer.dump(" i = %7i , Energy = %15.10f" %(i, self.get_total_energy()), self.pos)

            if self.dipole_writer is not None:
                dipole, dipole_deriv = self.get_dipole()
                self.dipole_writer.write("%15.10f %15.10f %15.10f %15.10f %15.10f %15.10f\n" %(
                    dipole[0], dipole[1], dipole[2],
                    dipole_deriv[0], dipole_deriv[1], dipole_deriv[2],
                ))




class NVE(MolecularDynamics):
    def __init__(self, ff, temperature=300*kelvin, temptol=1e-2, timestep=1.0*femtosecond, nsteps=100,
                 out=None, xyz_writer=None, dipole_writer=None, time_writer=None,
                 energy_writer=None, potential_writer=None, kinetic_writer=None, temperature_writer=None):
        MolecularDynamics.__init__(self, ff, timestep=timestep, nsteps=nsteps,
                 out=out, xyz_writer=xyz_writer, dipole_writer=dipole_writer, time_writer=time_writer,
                 energy_writer=energy_writer, potential_writer=potential_writer, kinetic_writer=kinetic_writer, temperature_writer=temperature_writer)
        self.temperature = temperature
        self.temptol=temptol
        self.verlet_initialize()


    def verlet_initialize(self):
        self.pos = self.ff.system.pos
        self.gpos = np.zeros(self.pos.shape, float)
        self.energy = self.ff.compute(self.gpos)
        temp=10*self.temperature #high enough value to ensure while loop runs initially
        print "~ Initialising verlet ..."
        while abs(temp/self.temperature-1.0)>self.temptol:
            self.vel = np.random.normal(0, 1, (self.natom, 3))*np.sqrt(boltzmann*2.0*self.temperature/self.masses)
            temp=self.get_inst_temp()/2.0
            print "    Half temperature of initial velocities = %.1f" %temp
        print "    Succes!"
        self.prevpos = self.pos - self.vel*self.timestep - self.gpos/(2.0*self.masses)*(self.timestep)**2
        self.write_output(-1)


    def verlet_integrate(self):
        tmp = self.pos.copy()
        self.pos = 2.0*self.pos - self.prevpos - self.gpos/self.masses*(self.timestep)**2
        self.vel = (self.pos - self.prevpos)/(2.0*self.timestep)
        self.prevpos = tmp.copy()


    def run(self):
        print "~ Running verlet integration ..."
        for i in xrange(self.nsteps):
            self.ff.update_pos(self.pos)
            self.gpos = np.zeros(self.pos.shape, float)
            self.energy = self.ff.compute(self.gpos)
            self.verlet_integrate()
            self.write_output(i)
