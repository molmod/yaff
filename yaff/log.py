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
"""Screen logger

   This module holds the main screen loging object of Yaff. The ``log`` object
   is an instance off the ``ScreenLog`` class in the module ``molmod.log``.
   The logger also comes with a timer infrastructure, which is also implemented
   in the ``molmod.log`` module.
"""


import atexit

from molmod.log import ScreenLog, TimerGroup

import yaff


__all__ = ['log', 'timer']


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

                  Welcome to Yaff {} - Yet another force field

                                   Written by
      Toon Verstraelen(1)*, Louis Vanduyfhuys(1) and Steven Vandenbrande(1)

(1) Center for Molecular Modeling, Ghent University Belgium.
* mailto: Toon.Verstraelen@UGent.be

In a not-too-distant future, this program will be renamed to NINJAFF, which
stands for 'NINJAFF is not just another force field code'. Please, bear with us.
""".format(yaff.__version__)


foot_banner = r"""
__/\\\__________________________________________________________________/\\\____
  \ \\\                                                                 \ \\\
   \ \\\      End of file. Thanks for using Yaff! Come back soon!!       \ \\\
____\///__________________________________________________________________\///__
"""

timer = TimerGroup()
log = ScreenLog('YAFF', yaff.__version__, head_banner, foot_banner, timer)
atexit.register(log.print_footer)
