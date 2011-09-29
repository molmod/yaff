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


import sys


__all__ = ['ScreenLog', 'log']


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
    content = 71

    def __init__(self, f=None):
        self._level = self.medium
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

    def __call__(self, key, s):
        key = key.upper().rjust(self.margin, '_')
        while len(s) > 0:
            if len(s) > self.content:
                pos = s.rfind(' ', 0, self.content)
                if pos == -1:
                    current = s[:self.content]
                    s = s[self.content:]
                else:
                    current = s[:pos]
                    s = s[pos:].lstrip()
            else:
                current = s
                s = ''
            print >> self._file, key, current


log = ScreenLog()
