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


__all__ = ['find_first', 'lex_split']


def find_first(s, subs, start=0, end=None):
    best_pos = -1
    best_sub = None
    for sub in subs:
        pos = s.find(sub, start, end)
        if pos >= 0:
            if best_pos == -1 or best_pos > pos:
                best_pos = pos
                best_sub = sub
                end = best_pos
    return best_pos, best_sub


def lex_split(s, splitter):
    """Split the string at the given character, ignoring characters in brackets

       This routine also checks if the number of opening and closing brackets
       match, and that there is no character in the string the is preceeded by
       more closing than opening brackets.
    """
    depth = 0
    start = 0
    last = 0
    result = []
    while start < len(s):
        if depth == 0:
            pos, char = find_first(s, splitter + '()', start)
            if pos == -1:
                break
            elif char == splitter:
                result.append(s[last:pos])
                last = pos+1
            elif char == '(':
                depth += 1
            else:
                raise ValueError('Closing more brackets than opening at char %i.' % pos)
            start = pos+1
        else:
            pos, char = find_first(s, '()', start)
            if pos == -1:
                break
            elif char == '(':
                depth += 1
            else:
                depth -= 1
            start = pos+1
    if depth != 0:
        raise ValueError('Too few closing brackets.')
    result.append(s[last:])
    return result
