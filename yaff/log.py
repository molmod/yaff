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


import sys, atexit, os, datetime

from molmod.units import kjmol, kcalmol, electronvolt, angstrom, nanometer, \
    femtosecond, picosecond, amu, deg


__all__ = ['ScreenLog', 'log']


head_banner = r"""
_____/\\\________/\\\___/\\\\\\\\\______/\\\\\\\\\\\\\\\__/\\\\\\\\\\\\\\\______
_____\///\\\____/\\\/__/\\\\\\\\\\\\\___\ \\\///////////__\ \\\///////////______
________\///\\\/\\\/___/\\\/////////\\\__\ \\\_____________\ \\\________________
___________\///\\\/____\ \\\_______\ \\\__\ \\\\\\\\\\\_____\ \\\\\\\\\\\_______
______________\ \\\_____\ \\\\\\\\\\\\\\\__\ \\\///////______\ \\\///////_______
_______________\ \\\_____\ \\\/////////\\\__\ \\\_____________\ \\\_____________
________________\ \\\_____\ \\\_______\ \\\__\ \\\_____________\ \\\____________
_________________\ \\\_____\ \\\_______\ \\\__\ \\\_____________\ \\\___________
__________________\///______\///________\///___\///______________\///___________


                 Welcome to YAFF - yet another force field code

                                   Written by
                  Toon Verstraelen(1)* and Louis Vanduyfhuys(1)

(1) Center for Molecular Modeling, Ghent University Belgium.
* mailto: Toon.Vesrtraelen@UGent.be

In a not-too-distant future, this program will be renamed to NJAFF, which stands
for 'not just another force field code'. Please, bear with us."""


foot_banner = r"""
__/\\\__________________________________________________________________/\\\____
  \ \\\                                                                 \ \\\
   \ \\\      End of file. Thanks for using YAFF! Come back soon!!       \ \\\
____\///__________________________________________________________________\///__
"""


class UnitSystem(object):
    def __init__(self, energy, length, time, mass, charge, force, forceconst, velocity, acceleration, angle):
        self.energy = energy
        self.length = length
        self.time = time
        self.mass = mass
        self.charge = charge
        self.force = force
        self.forceconst = forceconst
        self.velocity = velocity
        self.acceleration = acceleration
        self.angle = angle

    def log_info(self):
        if log.do_low:
            log.set_prefix('UNITS')
            log('The following units will be used below:')
            log.hline()
            log('Type          Conversion             Notation')
            log.hline()
            log('Energy        %21.15e  %s' % self.energy)
            log('Length        %21.15e  %s' % self.length)
            log('Time          %21.15e  %s' % self.time)
            log('Mass          %21.15e  %s' % self.mass)
            log('Charge        %21.15e  %s' % self.charge)
            log('Force         %21.15e  %s' % self.force)
            log('Force Const.  %21.15e  %s' % self.forceconst)
            log('Veolicty      %21.15e  %s' % self.velocity)
            log('Acceleration  %21.15e  %s' % self.acceleration)
            log('Angle         %21.15e  %s' % self.angle)
            log.hline()
            log('The internal data is divided by the corresponding conversion factor before it gets printed on screen.')

    def apply(self, some):
        some.energy = self.energy[0]
        some.length = self.length[0]
        some.time = self.time[0]
        some.mass = self.mass[0]
        some.force = self.force[0]
        some.forceconst = self.forceconst[0]
        some.velocity = self.velocity[0]
        some.angle = self.angle[0]


class ScreenLog(object):
    # log levels
    silent = 0
    warning = 1
    low = 2
    medium = 3
    high = 4
    debug = 5

    # screen parameters
    margin = 8
    width = 71

    # unit systems
    joule = UnitSystem(
        energy=(kjmol, 'kJ/mol'),
        length=(angstrom, 'A'),
        time=(femtosecond, 'fs'),
        mass=(amu, 'amu'),
        charge=(1, 'e'),
        force=(kjmol/angstrom, 'kJ/mol/A'),
        forceconst=(kjmol/angstrom**2, 'kJ/mol/A**2'),
        velocity=(angstrom/femtosecond, 'A/fs'),
        acceleration=(angstrom/femtosecond**2, 'A/fs**2'),
        angle=(deg, 'deg'),
    )
    cal = UnitSystem(
        energy=(kcalmol, 'kcal/mol'),
        length=(angstrom, 'A'),
        time=(femtosecond, 'fs'),
        mass=(amu, 'amu'),
        charge=(1, 'e'),
        force=(kcalmol/angstrom, 'kcal/mol/A'),
        forceconst=(kjmol/angstrom**2, 'kcal/mol/A**2'),
        velocity=(angstrom/femtosecond, 'A/fs'),
        acceleration=(angstrom/femtosecond**2, 'A/fs**2'),
        angle=(deg, 'deg'),
    )
    solid = UnitSystem(
        energy=(kcalmol, 'eV/mol'),
        length=(angstrom, 'A'),
        time=(femtosecond, 'fs'),
        mass=(amu, 'amu'),
        charge=(1, 'e'),
        force=(kcalmol/angstrom, 'eV/mol/A'),
        forceconst=(kjmol/angstrom**2, 'eV/mol/A**2'),
        velocity=(angstrom/femtosecond, 'A/fs'),
        acceleration=(angstrom/femtosecond**2, 'A/fs**2'),
        angle=(deg, 'deg'),
    )
    bio = UnitSystem(
        energy=(kcalmol, 'kcal/mol'),
        length=(nanometer, 'nm'),
        time=(picosecond, 'ps'),
        mass=(amu, 'amu'),
        charge=(1, 'e'),
        force=(kcalmol/nanometer, 'kcal/mol/nm'),
        forceconst=(kjmol/nanometer**2, 'kcal/mol/nm**2'),
        velocity=(nanometer/picosecond, 'A/ps'),
        acceleration=(nanometer/picosecond**2, 'A/ps**2'),
        angle=(deg, 'deg'),
    )
    atomic = UnitSystem(
        energy=(1, 'E_h'),
        length=(1, 'a_0'),
        time=(1, 'a.u.t'),
        mass=(1, 'amu'),
        charge=(1, 'e'),
        force=(1, 'E_h/a_0'),
        forceconst=(1, 'E_h/a_0**2'),
        velocity=(1, 'a_0/a.u.t'),
        acceleration=(1, 'a_0/a.u.t**2'),
        angle=(1, 'rad'),
    )


    def __init__(self, f=None):
        self._active = False
        self._level = self.medium
        self.unitsys = self.joule
        self.unitsys.apply(self)
        self.prefix = ' '*(self.margin-1)
        if f is None:
            self._file = sys.stdout
        else:
            self._file = f

    do_warning = property(lambda self: self._level >= self.warning)
    do_low = property(lambda self: self._level >= self.low)
    do_medium = property(lambda self: self._level >= self.medium)
    do_high = property(lambda self: self._level >= self.high)
    do_debug = property(lambda self: self._level >= self.debug)

    def set_level(self, level):
        if level < self.silent or level > self.debug:
            raise ValueError('The level must be one of the ScreenLog attributes.')
        self._level = level

    def __call__(self, *words):
        s = ' '.join(words)
        if not self.do_low:
            raise RuntimeError('The runlevel should be at least low when logging.')
        if not self._active:
            prefix = self.prefix
            self.print_header()
            self.prefix = prefix
            print >> self._file
        # Check for alignment code '&'
        pos = s.find('&')
        if pos == -1:
            lead = ''
            rest = s
        else:
            lead = s[:pos] + ' '
            rest = s[pos+1:]
        width = self.width - len(lead)
        if width < self.width/2:
            raise ValueError('The lead may not exceed half the width of the terminal.')
        # break and print the line
        first = True
        while len(rest) > 0:
            if len(rest) > width:
                pos = rest.rfind(' ', 0, width)
                if pos == -1:
                    current = rest[:width]
                    rest = rest[width:]
                else:
                    current = rest[:pos]
                    rest = rest[pos:].lstrip()
            else:
                current = rest
                rest = ''
            print >> self._file, '%s %s%s' % (self.prefix, lead, current)
            if first:
                lead = ' '*len(lead)
                first = False

    def hline(self, char='~'):
        self(char*self.width)

    def set_prefix(self, prefix):
        if len(prefix) > self.margin-1:
            raise ValueError('The prefix must be at most %s characters wide.' % (self.margin-1))
        self.prefix = prefix.upper().rjust(self.margin-1, ' ')
        if self._active:
            print >> self._file

    def set_unitsys(self, unitsys):
        self.unitsys = unitsys
        self.unitsys.apply(self)
        if self._active:
            self.unitsys.log_info()

    def print_header(self):
        if self.do_low and not self._active:
            self._active = True
            print >> self._file, head_banner
            self._print_basic_info()
            self.unitsys.log_info()

    def print_footer(self):
        # Do not show footer when program crashes.
        if self.do_low and sys.exc_info()[0] is None and self._active:
            self._print_basic_info()
            print >> self._file, foot_banner

    def _print_basic_info(self):
            import yaff
            log.set_prefix('ENV')
            log('User:          &' + os.getlogin())
            log('Machine info:  &' + ' '.join(os.uname()))
            log('Time:          &' + datetime.datetime.now().isoformat())
            log('Python version:&' + sys.version.replace('\n', ''))
            log('YAFF version:  &' + yaff.__version__)


log = ScreenLog()
atexit.register(log.print_footer)
