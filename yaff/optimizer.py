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

import sys, numpy as np
from molmod.minimizer import *
from molmod.units import *
from molmod.io.xyz import XYZWriter
from molmod.periodic import periodic as periodictable

from yaff.ext import *
from yaff.ff import *

__all__ = [
    'Optimizer', 'CellOptimizer', 'ThetaOptimizer',
]

class Optimizer(object):
    """
        An abstract Optimizer class.
    """
    def __init__(self, ff, rvecs_scale = 1e-2,
            search_direction = ConjugateGradient(), line_search = NewtonLineSearch(), convergence = ConvergenceCondition(grad_rms=1e-4, grad_max=3.3e-4),
            stop_loss = StopLossCondition(max_iter=1000), xyz_writer=None, out=None):
        self.ff = ff
        self.rvecs_scale = rvecs_scale
        self.periodic = ff.system.cell.rvecs.shape == (3,3)
        self.search_direction = search_direction
        self.line_search = line_search
        self.convergence = convergence
        self.stop_loss = stop_loss

        if out is None:
            out = file("yaff.out", 'w')
        else:
            if isinstance(out, file):
                self.out = out
            elif isinstance(out, str):
                self.out = file(out, 'w')
            else:
                raise TypeError("argument out should be of type string or file")

        if xyz_writer is None:
            self.xyz_writer = XYZWriter(
                file("yaff-traj.xyz", 'w'),
                [periodictable[i].symbol for i in ff.system.numbers]
            )
        else:
            if isinstance(xyz_writer, file):
                self.xyz_writer =  XYZWriter(
                    xyz_writer,
                    [periodictable[i].symbol for i in ff.system.numbers]
                )
            elif isinstance(xyz_writer, str):
                self.xyz_writer = XYZWriter(
                    file(xyz_writer, 'w'),
                    [periodictable[i].symbol for i in ff.system.numbers]
                )
            elif isinstance(xyz_writer, XYZWriter):
                self.xyz_writer = xyz_writer
            else:
                raise TypeError("argument xyz_writer should be of type file, string or XYZWriter")


    def get_xinit(self, *args):
        raise NotImplementedError

    def calc_energy(self, x, do_gradient=False):
        raise NotImplementedError

    def callback(self, m):
        raise NotImplementedError

    def write_results(self, *args):
        raise NotImplementedError

    def run(self):
        self.minimizer = Minimizer(
            self.get_xinit(), self.calc_energy, self.search_direction, self.line_search,
            self.convergence, self.stop_loss, anagrad=True, verbose=True, callback=self.callback
        )


class CellOptimizer(Optimizer):
    """
        An optimizer for performing geometrical optimization of orthorhombic cells in which all cell vector lengths are allowed to change.
    """
    def __init__(self, ff, rvecs_scale = 1e-2, search_direction=ConjugateGradient(), line_search=NewtonLineSearch(),
            convergence=ConvergenceCondition(grad_rms=1e-4, grad_max=3.3e-4), stop_loss=StopLossCondition(max_iter=1000),
            xyz_writer=None, out=None):
        Optimizer.__init__(self, ff, rvecs_scale=rvecs_scale, search_direction=search_direction, line_search=line_search,
            convergence=convergence, stop_loss=stop_loss, xyz_writer=xyz_writer, out=out)
        print >> self.out, ""
        print >> self.out, "  Geometry Optimization:"
        print >> self.out, "  ----------------------"
        print >> self.out, ""
        print >> self.out, "      i  |  Energy [kcalmol]  |   a [A]   b [A]    c [A]"
        print >> self.out, "    ----------------------------------------------------"

    def get_xinit(self):
        if self.periodic:
            x_init = np.concatenate([
                np.diag(self.ff.system.cell.rvecs)*self.rvecs_scale,
                np.dot(self.ff.system.pos, np.linalg.inv(self.ff.system.cell.rvecs)).ravel(),
            ])
        else:
            x_init = self.ff.system.pos.ravel()
        return x_init

    def calc_energy(self, x, do_gradient=False):
        if self.periodic:
            rvecs = np.diag(x[:3])/self.rvecs_scale
            scaled = x[3:].reshape(-1,3)
            pos = np.dot(scaled, rvecs)
            self.ff.update_rvecs(rvecs)
        else:
            pos = x.reshape(-1,3)
        self.ff.update_pos(pos)
        if do_gradient:
            gpos = np.zeros(pos.shape)
            vtens = np.zeros((3, 3), float)
            energy = self.ff.compute(gpos, vtens)
            if self.periodic:
                grad_rvecs = np.dot(self.ff.system.cell.gvecs, vtens)
                grad_scaled = np.dot(gpos, rvecs)
                gradient = np.concatenate([
                    np.diag(grad_rvecs)/self.rvecs_scale,
                    grad_scaled.ravel()
                ])
            else:
                gradient = gpos.ravel()
            return energy, gradient
        else:
            energy = self.ff.compute()
            return energy

    def callback(self, m):
        if self.periodic:
            rvecs = np.diag(m.x[:3])/self.rvecs_scale
            scaled = m.x[3:].reshape(-1,3)
            pos = np.dot(scaled, rvecs)
            self.xyz_writer.dump(
                'Energy = %.10f , Cell = %f,%f,%f' % (m.f, rvecs[0,0]/angstrom, rvecs[1,1]/angstrom, rvecs[2,2]/angstrom),
                pos
            )
            print >> self.out, "     %3i |    %10.6f    | % 5.3f  % 5.3f  % 5.3f " %(
                m.counter,
                m.f/kcalmol,
                self.ff.system.cell.rvecs[0,0]/angstrom,
                self.ff.system.cell.rvecs[1,1]/angstrom,
                self.ff.system.cell.rvecs[2,2]/angstrom,
            )
        else:
            pos = m.x.reshape(-1,3)
            self.xyz_writer.dump('Energy = %.10f' % m.f, pos)

    def write_results(self, xyz_writer=None, ener_writer=None, vdw_writer=None):
        # write energy to energy file
        energy = self.ff.compute()
        if ener_writer is not None:
            ener_writer.write("%15.10f\n" % energy)

        # write vdw contribution to vdw file
        if vdw_writer is not None:
            vdw_part = None
            for part in self.ff.parts:
                if isinstance(part, PairPart):
                    if isinstance(part.pair_pot, PairPotLJ) or isinstance(part.pair_pot, PairPotMM3):
                        vdw_part = part
            vdw = vdw_part._internal_compute(None, None)
            vdw_writer.write("%15.10f\n" % vdw)

        # write xyz to trajectory file
        if xyz_writer is not None:
            xyz_writer.dump('Energy  = %.10f' % energy, self.ff.system.pos)

        # write to general output
        print >> self.out, ""
        if self.minimizer.success:
            print >> self.out, " Final Geometry (CONVERGED)"
        else:
            print >> self.out, " Final Geometry (NOT CONVERGED)"

        print >> self.out, " ------------------------------------------------"
        print >> self.out, ""
        print >> self.out, "    Energy [kcalmol]= %15.9f" %(self.ff.compute()/kcalmol)
        print >> self.out, "    Cell dimensions:"
        print >> self.out, "       a [A] = %9.6f %9.6f %9.6f" %(
            self.ff.system.cell.rvecs[0,0]/angstrom, self.ff.system.cell.rvecs[0,1]/angstrom, self.ff.system.cell.rvecs[0,2]/angstrom
        )
        print >> self.out, "       b [A] = %9.6f %9.6f %9.6f" %(
            self.ff.system.cell.rvecs[1,0]/angstrom, self.ff.system.cell.rvecs[1,1]/angstrom, self.ff.system.cell.rvecs[1,2]/angstrom
        )
        print >> self.out, "       c [A] = %9.6f %9.6f %9.6f" %(
            self.ff.system.cell.rvecs[2,0]/angstrom, self.ff.system.cell.rvecs[2,1]/angstrom, self.ff.system.cell.rvecs[2,2]/angstrom
        )
        print >> self.out, "       cell volume          [A^3] = %10.6f" %(self.ff.system.cell.volume/angstrom**3)
        print >> self.out, "       cell diagonal length   [A] = %9.6f" %(np.sqrt(self.ff.system.cell.rvecs[0,0]**2 + self.ff.system.cell.rvecs[2,2]**2)/angstrom)
        print >> self.out, "       cell diagonal angle  [deg] = %9.6f" %(2.0*np.arctan(self.ff.system.cell.rvecs[2,2]/self.ff.system.cell.rvecs[0,0])/deg)
        print >> self.out, ""


class ThetaOptimizer(Optimizer):
    """
        An optimizer for performing geometrical optimization at a fixed angle of the diagonal between a and c cell vectors.
        The length of the diagonal and the length of the b cell vector is allowed to change during optimization.
    """
    def __init__(self, ff, theta, rvecs_scale = 1e-2, search_direction=ConjugateGradient(), line_search=NewtonLineSearch(),
            convergence=ConvergenceCondition(grad_rms=1e-4, grad_max=3.3e-4), stop_loss=StopLossCondition(max_iter=1000),
            xyz_writer=None, out=None):
        Optimizer.__init__(self, ff, rvecs_scale=rvecs_scale, search_direction=search_direction, line_search=line_search,
            convergence=convergence, stop_loss=stop_loss, xyz_writer=xyz_writer, out=out)
        self.theta = theta
        print >> self.out, "~"*150
        print >> self.out, " Geometry Optimization at theta = %6.3f deg:" %(self.theta/deg)
        print >> self.out, " ---------------------------------------------"
        print >> self.out, ""
        print >> self.out, "     i  |  Energy [kcalmol]  |   d [A]   b [A]"
        print >> self.out, "   -------------------------------------------"

    def get_xinit(self):
        if self.periodic:
            x_init = np.concatenate([
                np.array([
                    np.sqrt(self.ff.system.cell.rvecs[0,0]**2 + self.ff.system.cell.rvecs[2,2]**2),
                    self.ff.system.cell.rvecs[1,1]
                ])*self.rvecs_scale,
                np.dot(
                    self.ff.system.pos,
                    np.linalg.inv(self.ff.system.cell.rvecs)
                ).ravel(),
            ])
        else:
            x_init = self.ff.system.pos.ravel()
        return x_init

    def calc_energy(self, x, do_gradient=False):
        if self.periodic:
            rvecs = np.diag([
                x[0]*np.cos(self.theta/2.0),
                x[1],
                x[0]*np.sin(self.theta/2.0),
            ])/self.rvecs_scale
            scaled = x[2:].reshape(-1,3)
            pos = np.dot(scaled, rvecs)
            self.ff.update_rvecs(rvecs)
        else:
            pos = x.reshape(-1,3)
        self.ff.update_pos(pos)

        if do_gradient:
            gpos = np.zeros(self.ff.system.pos.shape)
            vtens = np.zeros((3, 3), float)
            energy = self.ff.compute(gpos, vtens)
            if self.periodic:
                grad_rvecs = np.dot(self.ff.system.cell.gvecs, vtens)
                grad_scaled = np.dot(gpos, rvecs)
                gradient = np.concatenate([
                    np.array([
                        grad_rvecs[0,0]*np.cos(self.theta/2.0) + grad_rvecs[2,2]*np.sin(self.theta/2.0),
                        grad_rvecs[1,1],
                    ])/self.rvecs_scale,
                    grad_scaled.ravel()
                ])
            else:
                gradient = gpos.ravel()
            return energy, gradient
        else:
            energy = self.ff.compute()
            return energy

    def callback(self, m):
        if self.periodic:
            d, b = m.x[:2]/self.rvecs_scale
            rvecs = np.diag(np.array([
                d*np.cos(self.theta/2.0),
                b,
                d*np.sin(self.theta/2.0),
            ]))
            scaled = m.x[2:].reshape(-1,3)
            pos = np.dot(scaled, rvecs)
            self.xyz_writer.dump(
                'Energy = %.10f , d = %6.3f , b = %6.3f' % (m.f, d, b),
                pos
            )
            print >> self.out, "    %3i |    %10.6f    | % 5.3f  % 5.3f  " %(
                m.counter,
                m.f/kcalmol,
                d/angstrom,
                b/angstrom,
            )
        else:
            pos = m.x.reshape(-1,3)
            self.xyzwriter.dump('Energy = %.10f' % m.f, pos)

    def write_results(self, index="", xyz_writer=None, ener_writer=None, vdw_writer=None):
        # write energy to energy file
        energy = self.ff.compute()
        if ener_writer is not None:
            ener_writer.write("%15.10f %15.10f\n" %(self.theta, energy))

        # write vdw contribution to vdw file
        if vdw_writer is not None:
            vdw_part = None
            for part in self.ff.parts:
                if isinstance(part, PairPart):
                    if isinstance(part.pair_pot,PairPotLJ) or isinstance(part.pair_pot,PairPotMM3) or isinstance(part.pair_pot,PairPotGrimme):
                        vdw_part = part
            if vdw_part is None:
                raise TypeError("No vdw part found in ff")
            vdw = vdw_part._internal_compute(None, None)
            vdw_writer.write("%15.10f %15.10f\n" %(self.theta, vdw))

        # write xyz to trajectory file
        if xyz_writer is not None:
            xyz_writer.dump(
                'Theta[%s]  = %.10f, d = %.4f, b = %.4f , Energy  = %.10f' % (
                    str(index), self.theta,
                    np.sqrt(self.ff.system.cell.rvecs[0,0]**2 + self.ff.system.cell.rvecs[2,2]**2)/angstrom,
                    self.ff.system.cell.rvecs[1,1]/angstrom,
                    energy
                ),
                self.ff.system.pos
            )

        # write to general output
        print >> self.out, ""
        if self.minimizer.success:
            print >> self.out, " Geometry at theta[%s] = %6.3f deg (CONVERGED)" %(str(index), self.theta/deg)
        else:
            print >> self.out, " Geometry at theta[%s] = %6.3f deg (NOT CONVERGED)" %(str(index), self.theta/deg)

        print >> self.out, " ----------------------------------"
        print >> self.out, ""
        print >> self.out, "    Energy [kcalmol]= %15.9f" %(self.ff.compute()/kcalmol)
        print >> self.out, "    Cell dimensions:"
        print >> self.out, "       a [A] = %9.6f %9.6f %9.6f" %(
            self.ff.system.cell.rvecs[0,0]/angstrom, self.ff.system.cell.rvecs[0,1]/angstrom, self.ff.system.cell.rvecs[0,2]/angstrom
        )
        print >> self.out, "       b [A] = %9.6f %9.6f %9.6f" %(
            self.ff.system.cell.rvecs[1,0]/angstrom, self.ff.system.cell.rvecs[1,1]/angstrom, self.ff.system.cell.rvecs[1,2]/angstrom
        )
        print >> self.out, "       c [A] = %9.6f %9.6f %9.6f" %(
            self.ff.system.cell.rvecs[2,0]/angstrom, self.ff.system.cell.rvecs[2,1]/angstrom, self.ff.system.cell.rvecs[2,2]/angstrom
        )
        print >> self.out, "       cell volume          [A^3] = %10.6f" %(self.ff.system.cell.volume/angstrom**3)
        print >> self.out, "       cell diagonal length   [A] = %9.6f" %(np.sqrt(self.ff.system.cell.rvecs[0,0]**2 + self.ff.system.cell.rvecs[2,2]**2)/angstrom)
        print >> self.out, "       cell diagonal angle  [deg] = %9.6f" %(2.0*np.arctan(self.ff.system.cell.rvecs[2,2]/self.ff.system.cell.rvecs[0,0])/deg)
        print >> self.out, ""
