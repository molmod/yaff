# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
#--
'''Barostats'''


import numpy as np

from molmod import boltzmann, femtosecond, kjmol

from yaff.log import log, timer
from yaff.sampling.utils import get_random_vel
from yaff.sampling.verlet import VerletHook


__all__ = [
    'AndersenMcDonaldBarostat', 'MartynaTobiasKleinBarostat'
]


class AndersenMcDonaldBarostat(VerletHook):
    def __init__(self, temp, press, start=0, step=1, amp=1e-3):
        """
           Warning: this code is not fully tested yet!

           **Arguments:**

           temp
                The average temperature of the NpT ensemble

           press
                The external pressure of the NpT ensemble

           **Optional arguments:**

           start
                The first iteration at which this hook is called

           step
                The number of iterations between two subsequent calls to this
                hook.

           amp
                The amplitude of the changes in the logarithm of the volume.
        """
        self.temp = temp
        self.press = press
        self.amp = amp
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        pass

    def pre(self, iterative):
        pass

    def post(self, iterative):
        def compute(pos, rvecs):
            iterative.pos[:] = pos
            iterative.gpos[:] = 0.0
            iterative.ff.update_rvecs(rvecs)
            iterative.ff.update_pos(pos)
            iterative.epot = iterative.ff.compute(iterative.gpos)
            iterative.acc = -iterative.gpos/iterative.masses.reshape(-1,1)

        natom = iterative.ff.system.natom
        with timer.section('AMB'):
            # A) Change the logarithm of the volume isotropically.
            scale = np.exp(np.random.uniform(-self.amp, self.amp))
            # A.0) Keep track of old state
            vol0 = iterative.ff.system.cell.volume
            epot0 = iterative.epot
            rvecs0 = iterative.ff.system.cell.rvecs.copy()
            pos0 = iterative.pos.copy()
            # A.1) scale the system and recompute the energy
            compute(pos0*scale, rvecs0*scale)
            epot1 = iterative.epot
            vol1 = iterative.ff.system.cell.volume
            # A.2) compute the acceptance ratio
            beta = 1/(boltzmann*self.temp)
            arg = (epot1 - epot0 + self.press*(vol1 - vol0) - (natom+1)/beta*np.log(vol1/vol0))
            accepted = arg < 0 or np.random.uniform(0, 1) < np.exp(-beta*arg)
            if accepted:
                # add a correction to the conserved quantity
                self.econs_correction += epot0 - epot1
            else:
                # revert the cell and the positions in the original state
                compute(pos0, rvecs0)
            # B) Change the velocities
            ekin0 = iterative._compute_ekin()
            iterative.vel[:] = get_random_vel(self.temp, False, iterative.masses)
            # C) Update the kinetic energy and the reference for the conserved quantity
            ekin1 = iterative._compute_ekin()
            self.econs_correction += ekin0 - ekin1
            if log.do_medium:
                with log.section('AMB'):
                    s = {True: 'accepted', False: 'rejected'}[accepted]
                    log('BARO   volscale %10.7f      arg %s      %s' % (scale, log.energy(arg), s))
                    if accepted:
                        log('BARO   energy change %s      (new vol)**(1/3) %s' % (
                            log.energy(epot1 - epot0), log.length(vol1**(1.0/3.0))
                        ))
                    log('THERMO energy change %s' % log.energy(ekin0 - ekin1))


class MartynaTobiasKleinBarostat(VerletHook):
    def __init__(self, ff, temp, press, start=0, timecon=1000*femtosecond):
        """
            This hook implements the combination of the Nosé-Hoover chain thermostat
            and the Martyna-Tobias-Klein barostat. The equations are derived in:

                Martyna, G. J.; Tobias, D. J.: Klein, M. L. J. Chem. Phys. 1994,
                101, 4177-4189.

            The implementation (used here) of a symplectic integrator of this thermostat
            and barostat is discussed in

                Martyna, G. J.;  Tuckerman, M. E.;  Tobias, D. J.;  Klein,
                M. L. Mol. Phys. 1996, 87, 1117-1157.

            **Arguments:**

            ff
                A ForceField instance.

            temp
                The temperature of thermostat.

            press
                The applied pressure for the barostat.

            **Optional arguments:**

            start
                The step at which the thermostat becomes active.

            timecon
                The time constant of the Martyna-Tobias-Klein barostat.
        """
        self.temp = temp
        self.press = press
        self.timecon_press = timecon
        self.cell = ff.system.cell.rvecs.copy()
        self.dim = ff.system.cell.nvec
        # number of degrees of freedom is not yet known
        self.dof = 0
        self.mass_press = 0
        self.vel_press = np.zeros((3,3),float)
        # symmetrize the cell tensor
        self.cell_symmetrize(ff)
        VerletHook.__init__(self, start, 1)

    def set_ndof(self, ndof):
        # allocate degrees of freedom
        angfreq = 2*np.pi/self.timecon_press
        self.mass_press = (ndof+self.dim**2)*boltzmann*self.temp/angfreq**2
        # define initial barostat velocity
        self.vel_press = self.get_random_vel_press()

    def cell_symmetrize(self, ff):
        # SVD decomposition of cell tensor
        U, s, Vt = np.linalg.svd(self.cell)
        # definition of the rotation matrix to symmetrize cell tensor
        rot_mat = np.dot(Vt.T, U.T)
        # symmetrize cell tensor and update cell
        self.cell = np.dot(self.cell, rot_mat)
        ff.update_rvecs(self.cell)
        # also update the new atomic positions
        pos_new = np.dot(ff.system.pos, rot_mat)
        ff.update_pos(pos_new)

    def get_random_vel_press(self):
        # generates symmetric tensor of barostat velocities
        shape = 3, 3
        # generate random 3x3 tensor
        rand = np.random.normal(0, np.sqrt(self.mass_press*boltzmann*self.temp), shape)/self.mass_press
        vel_press = np.zeros(shape)
        # create initial symmetric pressure velocity tensor
        for i in xrange(3):
            for j in xrange(3):
                if i >= j:
                    vel_press[i,j] = rand[i,j]
                else:
                    vel_press[i,j] = rand[j,i]
        return vel_press

    def init(self, iterative):
        self.timestep_press = iterative.timestep
        # compute gpos and vtens, since they differ
        # after symmetrising the cell tensor
        iterative.gpos = np.zeros(iterative.pos.shape, float)
        iterative.vtens = np.zeros((3,3),float)
        energy = iterative.ff.compute(iterative.gpos,iterative.vtens)

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def propagate_press(self, chain_vel, ndof, ekin, vel, masses, volume, iterative):
        # iL vxi_1 h/8
        self.vel_press *= np.exp(-chain_vel*self.timestep_press/8)
        # definition of P_intV and G
        ptens_vol = np.dot(vel.T*masses, vel) - iterative.vtens
        ptens_vol = 0.5*(ptens_vol.T + ptens_vol)
        G = (ptens_vol+(2.0*ekin/ndof-self.press*volume)*np.eye(3))/self.mass_press
        # iL G_g h/4
        self.vel_press += G*self.timestep_press/4
        # iL vxi_1 h/8
        self.vel_press *= np.exp(-chain_vel*self.timestep_press/8)

    def propagate_vel(self, chain_vel, ndof, vel, masses):
        # diagonalize propagator matrix and define D'_g
        Dg, Eg = np.linalg.eigh(self.vel_press+(np.trace(self.vel_press)/ndof+chain_vel)*np.eye(3))
        Daccg = np.diagflat(np.exp(-Dg*self.timestep_press/2))
        # iL (vg + Tr(vg)/ndof + vxi_1) h/2
        vel[:] = np.dot(np.dot(np.dot(vel, Eg), Daccg), Eg.T)
        # change energy
        ekin = 0.5*(vel**2*masses.reshape(-1,1)).sum()
        return vel, ekin

    def add_press_cont(self):
        kt = self.temp*boltzmann
        # pressure contribution to g1: kinetic cell tensor energy
        # and extra degrees of freedom due to cell tensor
        return self.mass_press*np.trace(np.dot(self.vel_press.T,self.vel_press)) - self.dim**2*kt

    def get_econs_correction(self, chain_pos, volume):
        kt = boltzmann*self.temp
        # add correction due to combination barostat and thermostat
        return self.dim**2*kt*chain_pos + 0.5*self.mass_press*np.trace(np.dot(self.vel_press.T,self.vel_press)) + self.press*volume
