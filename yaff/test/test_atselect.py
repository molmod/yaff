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


from yaff import *


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
    assert atsel_compile('C').get_string() == 'C'
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
