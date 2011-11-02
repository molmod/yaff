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


                 Welcome to YAFF - yet another force field code

                                   Written by
                  Toon Verstraelen(1)* and Louis Vanduyfhuys(1)

(1) Center for Molecular Modeling, Ghent University Belgium.
* mailto: Toon.Vesrtraelen@UGent.be

In a not-too-distant future, this program will be renamed to NJAFF, which stands
for 'not just another force field code'. Please, bear with us.
"""


foot_banner = r"""
__/\\\__________________________________________________________________/\\\____
  \ \\\                                                                 \ \\\
   \ \\\      End of file. Thanks for using YAFF! Come back soon!!       \ \\\
____\///__________________________________________________________________\///__
"""

timer = TimerGroup()
log = ScreenLog('YAFF', yaff.__version__, head_banner, foot_banner, timer)
atexit.register(log.print_footer)
