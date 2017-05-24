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
"""The ATSELECT language

   The ATSELECT language is a single-line language that can be used to define
   atom types in Yaff, or to construct a list of atoms in a system that match an
   ATSELECT rule.
"""


from collections import namedtuple


__all__ = [
    'check_name', 'find_first', 'lex_find', 'lex_split', 'atsel_compile', 'iter_matches',
]


def check_name(name):
    """Raise a ``ValueError`` if the given ffatype or scope name is invalid.

       **Arugments:**

       name
            The name to be checked.

       A name is invalid ...

       * if it has length zero,
       * if it contains any of ``:%=<>@()&|!``, or
       * if it starts with a digit [0-9].

       These rules make sure that the ffatype and scope names do not conflict
       with the ATSELECT language.
    """
    if len(name) == 0:
        raise ValueError('A name may not be empty.')
    for symbol in ':%=<>@()&|!':
        if symbol in name:
            raise ValueError('A scope or atom type name should not contain \'%s\'. Invalid name: \'%s\'' % (symbol, name))
    if symbol[0].isdigit():
        raise ValueError('A scope or atom type name should not start with a digit.')


def find_first(s, subs, start=0, end=None):
    """Find the first occurence of a set of substrings in ``s``

       **Arguments:**

       s
            The string that must be searched for the patterns in the ``subs``
            list.

       subs
            A list of patterns.

       **Optional arguments:**

       start
            Only start searching at this index in the string ``s``.

       end
            When given, no subs occuring after end are mathcing.
    """
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


def lex_find(s, sub, start=0, end=None):
    """find sub in s that is not enclosed in brackets"""
    assert '(' not in sub
    assert ')' not in sub
    if s.count('(') != s.count(')'):
        raise ValueError('The number of opening and closing brackets must be equal')
    depth = 0
    while start < len(s):
        if depth == 0:
            pos, match = find_first(s, (sub, '(', ')'), start, end)
            if pos == -1:
                return -1
            elif match == sub:
                return pos
            elif match == '(':
                depth += 1
            else:
                raise ValueError('Closing more brackets than opening at char %i.' % pos)
            start = pos+1
        else:
            pos, match = find_first(s, '()', start)
            if pos == -1:
                break
            elif match == '(':
                depth += 1
            else:
                depth -= 1
            start = pos+1
    return -1


def lex_split(s, splitter):
    """Split the string at the given character, ignoring characters in brackets

       This routine also checks if the number of opening and closing brackets
       match, and that there is no character in the string that is preceeded by
       more closing than opening brackets.
    """
    assert len(splitter) == 1
    start = 0
    result = []
    while start < len(s):
        pos = lex_find(s, splitter, start)
        if pos == -1:
            break
        result.append(s[start:pos])
        start = pos+1
    result.append(s[start:])
    return result


class Rule(object):
    precedence = 0

    def get_string(self, precedence=1000):
        result = self._get_string_low()
        if self.precedence > precedence:
            result = '(%s)' % result
        return result


class All(Rule):
    precedence = 100

    @staticmethod
    def _compile(s):
        words = lex_split(s, '&')
        if len(words) > 1:
            return All(*[_compile_low(word) for word in words])

    def __init__(self, *fns):
        self.fns = fns

    def __call__(self, system, i):
        for fn in self.fns:
            if not fn(system, i):
                return False
        return True

    def _get_string_low(self):
        return '&'.join(fn.get_string(self.precedence) for fn in self.fns)


class Any(Rule):
    precedence = 90

    @staticmethod
    def _compile(s):
        words = lex_split(s, '|')
        if len(words) > 1:
            return Any(*[_compile_low(word) for word in words])

    def __init__(self, *fns):
        self.fns = fns

    def __call__(self, system, i):
        for fn in self.fns:
            if fn(system, i):
                return True
        return False

    def _get_string_low(self):
        return '|'.join(fn.get_string(self.precedence) for fn in self.fns)


class Not(Rule):
    precedence = 80

    @staticmethod
    def _compile(s):
        if s.startswith('!'):
            return Not(_compile_low(s[1:]))

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, system, i):
        return not self.fn(system, i)

    def _get_string_low(self):
        return '!' + self.fn.get_string(self.precedence)


class BaseNeighs(Rule):
    precedence = 80
    first = None

    @classmethod
    def _compile(cls, s):
        if s.startswith(cls.first):
            pos = s.find('%')
            if pos >= 0:
                num = int(s[1:pos])
                fn = _compile_low(s[pos+1:])
            else:
                num = int(s[1:])
                fn = None
            return cls(num, fn)

    def __init__(self, num, fn):
        self.num = num
        self.fn = fn

    def __call__(self, system, i):
        if system.bonds is None:
            raise ValueError('The system does not have bond data.')
        num = 0
        for j in system.neighs1[i]:
            if self.fn is None or self.fn(system, j):
                num += 1
        return num

    def _get_string_low(self):
        if self.fn is None:
            return '%s%i' % (self.first, self.num)
        else:
            return '%s%i%%%s' % (self.first, self.num, self.fn.get_string(self.precedence))


class CountNeighs(BaseNeighs):
    precedence = 80
    first = '='

    def __call__(self, system, i):
        return BaseNeighs.__call__(self, system, i) == self.num


class LessNeighs(BaseNeighs):
    precedence = 80
    first = '<'

    def __call__(self, system, i):
        return BaseNeighs.__call__(self, system, i) < self.num


class MoreNeighs(BaseNeighs):
    precedence = 80
    first = '>'

    def __call__(self, system, i):
        return BaseNeighs.__call__(self, system, i) > self.num


class Name(Rule):
    precedence = 70

    @staticmethod
    def _compile(s):
        pos = s.find(':')
        if pos == -1:
            scope = None
            ffatype = s
        else:
            scope = s[:pos]
            ffatype = s[pos+1:]
        if ffatype.isdigit():
            number = int(ffatype)
            ffatype = None
        else:
            number = None
        return Name(scope, ffatype, number)

    def __init__(self, scope, ffatype, number):
        if scope is not None:
            check_name(scope)
        if ffatype is not None:
            check_name(ffatype)
        self.scope = scope
        self.ffatype = ffatype
        self.number = number

    def __call__(self, system, i):
        if self.scope is not None:
            if system.scopes is None:
                raise ValueError('The system does not have scopes.')
            if system.get_scope(i) != self.scope:
                return False
        if self.ffatype != '*':
            if self.ffatype is not None:
                if system.ffatypes is None:
                    raise ValueError('The system does not have ffatypes.')
                if system.get_ffatype(i) != self.ffatype:
                    return False
            if self.number is not None:
                if system.numbers[i] != self.number:
                    return False
        return True

    def _get_string_low(self):
        if self.ffatype is not None:
            result = self.ffatype
        elif self.number is not None:
            result = str(self.number)
        else:
            raise RuntimeError('This should not happen.')
        if self.scope is not None:
            result = '%s:%s' % (self.scope, result)
        return result


rules = [All, Any, Not, CountNeighs, LessNeighs, MoreNeighs, Name]


def atsel_compile(s):
    """Compiles an ATSELECT line into a boolean function

       **Argument:**

       s
            a string with one ATSELECT line. Before this line is processed, all
            whitespace (including newline) charachters are removed.

       **Returns:**

       rule
            A function that takes two arguments: ``system`` and ``i``, where
            system is a ``System`` instance and ``i`` is an atom index. The
            function returns ``True`` if the atom matches the ATSELECT line
            in the string ``s``.
    """
    # first get rid of the whitespace
    s = s.replace(' ', '')
    s = s.replace('\t', '')
    s = s.replace('\n', '')
    return _compile_low(s)


def _compile_low(s):
    while len(s) >= 2 and s[0] == '(' and s[-1] == ')':
        s = s[1:-1]
    for rule in rules:
        result = rule._compile(s)
        if result is not None:
            return result
    raise ValueError('Do not know how to compile: %s' % s)


Slice = namedtuple('Slice', ['match', 'error_sq', 'next_index1', 'next_allowed_index0'])


class Stack(object):
    """Stack object used by iter_matches to build up partial solutions."""

    def __init__(self, dm0, dm1, allowed, threshold_sq, error_sq_fn):
        """Initialize Stack instance.

        See iter_matches for the meaning of the parameters.
        """
        self._dm0 = dm0
        self._dm1 = dm1
        self._allowed = allowed
        self._threshold_sq = threshold_sq
        if error_sq_fn is None:
            self._error_sq_fn = lambda x, y: (x - y)**2
        else:
            self._error_sq_fn = error_sq_fn
        # Each element in _state is a tuple with the following elements:
        # - partial solution so far, internally stored as pairs (index1, index0)
        # - error_sq so far
        # - next index1
        # - list of allowed index0 values to be paired with index1
        self._state = [Slice((), 0.0, 0, list(allowed[0]))]

    def grow(self):
        """Extend the current partial solution with a new element, taken from allowed."""
        # Sanity check
        assert not self.is_complete()
        # Take the latest partial solution and prepare next element.
        prev_match, prev_error_sq, new_index1, new_allowed_index0 = self._state[-1]
        assert len(new_allowed_index0) > 0
        new_index0 = new_allowed_index0.pop(0)
        # Compute the updated error_sq
        new_error_sq = prev_error_sq
        for prev_index1, prev_index0 in prev_match:
            new_error_sq += self._error_sq_fn(
                self._dm0[prev_index0, new_index0],
                self._dm1[prev_index1, new_index1]
            )
        new_match = prev_match + ((new_index1, new_index0),)
        # Prepare for next grow call: new list of allowed atoms
        if len(self._state) == len(self._allowed) or new_error_sq >= self._threshold_sq:
            # Don't prepare for the next iteration if all atoms of the subsystem have been
            # assigned or when the error has gotten too large.
            next_index1 = None
            next_allowed_index0_sorted = None
        else:
            # Take as next index1 the one that is closest to the new_index1.
            used_index1 = set([index1 for index1, index0 in new_match])
            next_allowed_index1 = set(range(len(self._allowed))) - used_index1
            next_index1 = self._get_sorted_indexes(self._dm1, new_index1, next_allowed_index1)[0]
            # The new indexes to try is the given set of indexes, minus the ones already
            # used.
            used_index0 = set([index0 for index1, index0 in new_match])
            #   - unused index0 should be considered for matching
            next_allowed_index0 = list(set(self._allowed[next_index1]) - set(used_index0))
            #   - nearby first
            next_allowed_index0_sorted = self._get_sorted_indexes(self._dm0, new_index0, next_allowed_index0)
        self._state.append(Slice(new_match, new_error_sq, next_index1, next_allowed_index0_sorted))

    @staticmethod
    def _get_sorted_indexes(dm, new_index, next_allowed_index):
        distances_sq = [((dm[new_index, next_index]**2).sum(), next_index)
                        for next_index in next_allowed_index]
        return [index for d, index in sorted(distances_sq)]


    def shrink(self):
        """Discard the current partial solution and prepare stack for adding new element."""
        self._state.pop(-1)
        while len(self._state) > 0 and len(self._state[-1].next_allowed_index0) == 0:
            self._state.pop(-1)

    def drop(self):
        """Remove the current match from the allowed atoms and drop partial solution."""
        match = self.get_match()
        # Remove previously found atoms from the allowed lists.
        self._allowed = [[i for i in a if i not in match] for a in self._allowed]
        # Drop the current stack.
        if any(len(a) == 0 for a in self._allowed):
            self._state = []
        else:
            self._state = [Slice((), 0.0, 0, list(self._allowed[0]))]

    def is_acceptable(self):
        """Return True if the partial solution is acceptable."""
        return self._state[-1].error_sq < self._threshold_sq

    def is_complete(self):
        """Return True if the partial solution is actually a complete one."""
        return self._state[-1].next_index1 is None

    def is_done(self):
        """Return True when no more reorderings are available for testing."""
        return len(self._state) == 0

    def get_match(self):
        """Return the current (partial) match."""
        return tuple([index0 for index1, index0 in sorted(self._state[-1].match)])


def iter_matches(dm0, dm1, allowed, threshold=1e-3, error_sq_fn=None, overlapping=True):
    """Iterate over all possible renumberings of system 1 that are present in system 0.

    Parameters
    ----------
    dm0 : np.ndarray, shape=(n0, n0)
        A reference distance matrix (or analogeous object).
    dm1 : np.ndarray, shape=(n1, n1), n1 <= n0
        A distance matrix of the system to be reordered.
    allowed : list (length n1) of lists of integer indexes
        For each element in system 1, the allowed corresponding indexes in system 0. This
        can be based on corresponding chemical elements, atom types, etc. The more
        specific, the more efficient this function becomes.
    threshold : float
        An allowed deviation (L2-norm) between the distance matrix of the
        reference and the reordered system 1.
    error_sq_fn : function taking two arguments
        When not given, the squared difference is computed as error measure when comparing
        two distance matrix elements (dm0 versus dm1). This can be replaced by any
        user-provided squared error function.
    overlapping : bool
        When set to False, the algorithm excludes matches that are permutations of
        previous matches or that have a partial overlap with any previous match.

    Yields
    ------
    match : tuple
        All possible renumberings of elements in system 1 that match with (a subset of)
        system 0. All elements of the tuple are integers.
    """

    # Convert allowed to lists of lists, if needed.
    allowed = [list(a) for a in allowed]
    # There is no hope of finding a solution if one of the allowed lists is empty.
    if any(len(a) == 0 for a in allowed):
        return
    # The length of the allowed list should be correct for a solution to exist.
    if dm1.shape[0] != len(allowed):
        return
    # The allowed indexes must be present in the reference.
    if not set(sum(allowed, [])).issubset(range(dm0.shape[0])):
        return

    stack = Stack(dm0, dm1, allowed, threshold**2, error_sq_fn)
    while not stack.is_done():
        # Try to add something to stack
        stack.grow()
        if not stack.is_acceptable():
            # Discard new partial solution if the error becomes too big.
            stack.shrink()
        elif stack.is_complete():
            match = stack.get_match()
            if overlapping:
                # Discard current solution because it is already generated.
                stack.shrink()
            else:
                # Discard the current stack and reduce the allowed lists.
                stack.drop()
            yield match
