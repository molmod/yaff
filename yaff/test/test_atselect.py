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



import numpy as np

from yaff import *

from yaff.test.common import get_system_water32, get_system_caffeine


def test_find_first():
    assert find_first('(foo)', '()', 1) == (4, ')')
    assert find_first('(foo)', '()', 0) == (0, '(')
    s = 'This is a test'
    assert find_first(s, ('a', 'is')) == (2, 'is')
    assert find_first(s, ('a', 'te', 're')) == (8, 'a')
    assert find_first(s, ('qwefasd', 'fsa', 're')) == (-1, None)
    assert find_first(s, ('a',)) == (8, 'a')


def test_lex_find():
    assert lex_find('foo&bar&spam&egg', '&') == 3
    assert lex_find('(foo&bar)&spam&egg', '&') == 9
    assert lex_find('(foo&bar&spam&egg)', '&') == -1
    assert lex_find('(foo&(bar))&spam&egg', '&') == 11
    assert lex_find('foo&(bar&spam)&egg', '&') == 3
    assert lex_find('foo&bar&(spam&egg)', '&') == 3
    assert lex_find('(foo&bar)&(spam&egg)', '&') == 9


def test_lex_split():
    assert lex_split('foo&bar&spam&egg', '&') == 'foo&bar&spam&egg'.split('&')
    assert lex_split('(foo&bar)&spam&egg', '&') == ['(foo&bar)', 'spam', 'egg']
    assert lex_split('(foo&bar&spam&egg)', '&') == ['(foo&bar&spam&egg)']
    assert lex_split('(foo&(bar))&spam&egg', '&') == ['(foo&(bar))', 'spam', 'egg']
    assert lex_split('foo&(bar&spam)&egg', '&') == ['foo', '(bar&spam)', 'egg']
    assert lex_split('foo&bar&(spam&egg)', '&') == ['foo', 'bar', '(spam&egg)']
    assert lex_split('(foo&bar)&(spam&egg)', '&') == ['(foo&bar)', '(spam&egg)']


def test_compile():
    assert atsel_compile('!0').get_string() == '!0'
    assert atsel_compile('C').get_string() == 'C'
    assert atsel_compile('(C)').get_string() == 'C'
    assert atsel_compile('((C))').get_string() == 'C'
    assert atsel_compile('C&N').get_string() == 'C&N'
    assert atsel_compile('C&N|O').get_string() == 'C&N|O'
    assert atsel_compile('(C&N)|O').get_string() == '(C&N)|O'
    assert atsel_compile('(!C)&O').get_string() == '!C&O'
    assert atsel_compile('C&=3').get_string() == 'C&=3'
    assert atsel_compile('C&>3').get_string() == 'C&>3'
    assert atsel_compile('C&<3').get_string() == 'C&<3'
    assert atsel_compile('C&=3%1').get_string() == 'C&=3%1'
    assert atsel_compile('C&=3%(1|O_W)').get_string() == 'C&=3%(1|O_W)'
    assert atsel_compile('!(C&=3%1)').get_string() == '!(C&=3%1)'
    assert atsel_compile('ALKANE:C&=3%1').get_string() == 'ALKANE:C&=3%1'
    assert atsel_compile('ALKANE:*&=3%1').get_string() == 'ALKANE:*&=3%1'
    assert atsel_compile('ALKANE:6&=3%1').get_string() == 'ALKANE:6&=3%1'
    assert atsel_compile('C &\t<\n3').get_string() == 'C&<3'
    assert atsel_compile('C _a &\t<\n3').get_string() == 'C_a&<3'


def test_compile_failures():
    ss = ['((C)', '())(', '=x', '!!', '&', '=2%()']
    for s in ss:
        try:
            fn = atsel_compile(s)
            assert False, 'The following should raise a ValueError when compiling: %s' % s
        except ValueError:
            pass


def test_atselect_water32():
    system = get_system_water32()
    o_indexes = (system.numbers == 8).nonzero()[0]
    for s in 'O', '8', '=2', '>1', '=2%H':
        fn = atsel_compile(s)
        assert fn(system, 0)
        assert not fn(system, 1)
        assert not fn(system, 2)
        assert fn(system, 3)
        assert not fn(system, 4)
        assert not fn(system, 5)
        assert fn(system, 6)
        assert not fn(system, 7)
        assert not fn(system, 8)
        assert (system.get_indexes(s)==o_indexes).all()
        assert (system.get_indexes(fn)==o_indexes).all()
    h_indexes = (system.numbers == 1).nonzero()[0]
    for s in 'H', '1', '=1', '<2', '=1%O':
        fn = atsel_compile(s)
        assert not fn(system, 0)
        assert fn(system, 1)
        assert fn(system, 2)
        assert not fn(system, 3)
        assert fn(system, 4)
        assert fn(system, 5)
        assert not fn(system, 6)
        assert fn(system, 7)
        assert fn(system, 8)
        assert (system.get_indexes(s)==h_indexes).all()
        assert (system.get_indexes(fn)==h_indexes).all()


def test_atselect_caffeine():
    system = get_system_caffeine()
    assert (system.get_indexes('C&=3%H')==np.array([11, 12, 13])).all()
    assert (system.get_indexes('C&=2%N')==np.array([7, 9, 10])).all()
    assert (system.get_indexes('O')==np.array([0, 1])).all()
    assert (system.get_indexes('O&=1')==np.array([0, 1])).all()
    assert (system.get_indexes('O&=1%C')==np.array([0, 1])).all()
    assert (system.get_indexes('O&=1%(C&=3)')==np.array([0, 1])).all()
    assert (system.get_indexes('O&=1%(C&=2%N)')==np.array([1])).all()
    assert (system.get_indexes('O&=1%(C&=1%C)')==np.array([0])).all()
    assert (system.get_indexes('C&=3')==np.array([6, 7, 8, 9, 10])).all()
    assert (system.get_indexes('C&=1%C')==np.array([7, 8])).all()
    assert (system.get_indexes('C&>1%C')==np.array([6])).all()
    assert (system.get_indexes('C&<2%C')==np.array([7, 8, 9, 10, 11, 12, 13])).all()
    assert (system.get_indexes('N&=2%C&=2')==np.array([5])).all()
    assert (system.get_indexes('N&!=2')==np.array([2, 3, 4])).all()
    assert (system.get_indexes('N|O')==np.array([0, 1, 2, 3, 4, 5])).all()
    assert (system.get_indexes('N|8')==np.array([0, 1, 2, 3, 4, 5])).all()
    assert (system.get_indexes('!0')==np.arange(system.natom)).all()


def test_atselect_scope():
    system = System(
        numbers=np.array([8, 1, 1, 6, 1, 1, 1, 8, 1]),
        pos=np.zeros((9, 3), float),
        scopes=['WAT', 'WAT', 'WAT', 'METH', 'METH', 'METH', 'METH', 'METH', 'METH'],
        ffatypes=['O', 'H', 'H', 'C', 'H_C', 'H_C', 'H_C', 'O', 'H_O'],
    )
    assert (system.get_indexes('WAT:*')==np.array([0, 1, 2])).all()
    assert (system.get_indexes('METH:*')==np.array([3, 4, 5, 6, 7, 8])).all()
    assert (system.get_indexes('WAT:H')==np.array([1, 2])).all()
    assert (system.get_indexes('WAT:1')==np.array([1, 2])).all()
    assert (system.get_indexes('WAT:1|WAT:8')==np.array([0, 1, 2])).all()
    assert (system.get_indexes('O')==np.array([0, 7])).all()
    assert (system.get_indexes('8')==np.array([0, 7])).all()


def test_iter_matches_water_water():
    # Water molecule with oxygen in center
    dm0 = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0],
    ])
    # Water molecule with oxygen first
    dm1 = np.array([
        [0, 1, 1],
        [1, 0, 2],
        [1, 2, 0],
    ])
    # Allowed new indexes
    allowed = [[1], [0, 2], [0, 2]]
    # Get all solutions
    solutions = np.array(sorted(iter_matches(dm0, dm1, allowed)))
    np.testing.assert_equal(solutions, [[1, 0, 2], [1, 2, 0]])


def test_iter_matches_water_hydroxyl():
    # Water molecule with oxygen in center
    dm0 = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0],
    ])
    # Hydroxyl group
    dm1 = np.array([
        [0, 1],
        [1, 0],
    ])
    # Allowed new indexes
    allowed = [[1], [0, 2]]
    # Get all solutions
    solutions = np.array(sorted(iter_matches(dm0, dm1, allowed)))
    np.testing.assert_equal(solutions, [[1, 0], [1, 2]])
