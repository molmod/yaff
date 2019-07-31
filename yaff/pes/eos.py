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
'''Equations of state'''


from __future__ import division

import numpy as np
from scipy.optimize import newton as newton_opt # Avoid clash with newton from molmod.units
import pkg_resources

from molmod import boltzmann, planck, amu, pascal, kelvin

import yaff
from yaff.log import log


__all__ = [
    'PREOS',
]


class PREOS(object):
    """The Peng-Robinson equation of state"""
    def __init__(self, Tc, Pc, omega, mass=0.0):
        """
           The Peng-Robinson EOS gives a relation between pressure, volume, and
           temperature with parameters based on the critical pressure, critical
           temperature and acentric factor.

           **Arguments:**

           Tc
                The critical temperature of the species

           Pc
                The critical pressure of the species

           omega
                The acentric factor of the species

           **Optional arguments:**

           mass
                The mass of one molecule of the species. Some properties can be
                computed without this, so it is an optional argument
        """
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.mass = mass
        # Some parameters derived from the input parameters
        self.a = 0.457235 * self.Tc**2 / self.Pc
        self.b = 0.0777961 * self.Tc / self.Pc
        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2

    @classmethod
    def from_name(cls, compound):
        """
           Initialize a Peng-Robinson EOS based on the name of the compound.
           Only works if the given compound name is included in
           'yaff/data/critical_acentric.csv'
        """
        # Read the data file containing parameters for a number of selected compounds
        fn = pkg_resources.resource_filename(yaff.__name__, 'data/critical_acentric.csv')
        dtype=[('compound','S20'),('mass','f8'),('Tc','f8'),('Pc','f8'),('omega','f8'),]
        data = np.genfromtxt(fn, dtype=dtype, delimiter=',')
        # Select requested compound
        if not compound.encode('utf-8') in data['compound']:
            raise ValueError("Could not find data for %s in file %s"%(compound,fn))
        index = np.where( compound.encode('utf-8') == data['compound'] )[0]
        assert index.shape[0]==1
        mass = data['mass'][index[0]]*amu
        Tc = data['Tc'][index[0]]*kelvin
        Pc = data['Pc'][index[0]]*1e6*pascal
        omega = data['omega'][index[0]]
        return cls(Tc, Pc, omega, mass=mass)

    def set_conditions(self, T, P):
        """
           Set the parameters that depend on T and P

           **Arguments:**

           T
                Temperature

           P
                Pressure
        """
        self.Tr = T / self.Tc  # reduced temperature
        self.alpha = (1 + self.kappa * (1 - np.sqrt(self.Tr)))**2
        self.A = self.a * self.alpha * P / T**2
        self.B = self.b * P / T

    def polynomial(self, Z):
        """
           Evaluate the polynomial form of the Peng-Robinson equation of state
           If returns zero, the point lies on the PR EOS curve

           **Arguments:**

           Z
                Compressibility factor
        """
        return Z**3 - (1 - self.B) * Z**2 + (self.A - 2*self.B - 3*self.B**2) * Z - (
                self.A * self.B - self.B**2 - self.B**3)

    def calculate_mu_ex(self, T, P):
        """
           Evaluate the excess chemical potential at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Returns:**

           mu
                The excess chemical potential

           Pref

                The pressure at which the reference chemical potential was calculated
        """
        # Find a reference pressure at the given temperature for which the fluidum
        # is nearly ideal
        Pref = P
        for i in range(100):
            self.set_conditions(T, Pref)
            Zref = newton_opt(self.polynomial, 1.0)
            # Z close to 1.0 means ideal gas behavior
            if np.abs(Zref-1.0)>1e-5:
                Pref /= 2.0
            else: break
        if np.abs(Zref-1.0)>1e-5:
            raise ValueError("Failed to find pressure where the fluidum is ideal-gas like, check input parameters")
        # Find zero of polynomial expression to get the compressibility factor
        self.set_conditions(T, P)
        Z = newton_opt(self.polynomial, 1.0)
        # Add contributions to chemical potential at requested pressure
        mu = Z - 1 - np.log(Z - self.B) - self.A / np.sqrt(8) / self.B * np.log(
                    (Z + (1 + np.sqrt(2)) * self.B) / (Z + (1 - np.sqrt(2)) * self.B))
        mu += np.log(P/Pref)
        mu *= T*boltzmann
        return mu, Pref

    def calculate_fugacity(self, T, P):
        """
           Evaluate the excess chemical potential at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Returns:**

           f
                The fugacity
        """
        mu, Pref = self.calculate_mu_ex(T, P)
        fugacity = np.exp( mu/(boltzmann*T) )*Pref
        return fugacity

    def calculate_mu(self, T, P):
        """
           Evaluate the chemical potential at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Returns:**

           mu
                The chemical potential
        """
        # Excess part
        mu, Pref = self.calculate_mu_ex(T,P)
        # Ideal gas contribution to chemical potential
        assert self.mass!=0.0
        lambd = 2.0*np.pi*self.mass*boltzmann*T/planck**2
        mu0 = -boltzmann*T*np.log( boltzmann*T/Pref*lambd**1.5)
        return mu0+mu
