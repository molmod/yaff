..
    : YAFF is yet another force-field code.
    : Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
    : Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
    : (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
    : stated.
    :
    : This file is part of YAFF.
    :
    : YAFF is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : YAFF is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>
    :
    : --

.. _ug_sec_atselect:

The ATSELECT language
#####################

Several functions (or methods) in Yaff have a selection atoms as one of the
function arguments. This selection must be provided in the form of a list or
array of selected atom indexes. The ATSELECT language allows one to define
rules that generate such lists of atom indexes that match a certain
specification.

ATSELECT is similar to `SMARTS
<http://en.wikipedia.org/wiki/Smiles_arbitrary_target_specification>`_.
The SMARTS system has the advantage of being very compact, but it has a few
disadvantages that make it poorly applicable in the context of Yaff: e.g. it
assumes that the hybridization state of first-row atoms and bond orders are
known. The only real `knowns` for ATSELECT are: ``numbers`` and
optionally ``ffatypes``, ``scopes`` and ``bonds``.

The syntax of the ATSELECT language is defined as follows. An ATSELECT
expression consists of a single line and is case-sensitive. White-space is
completely ignored. An ATSELECT expression can be any of the following:

``[scope:]number``
    Matches an atom with the given number, optionally part of the given scope.

``[scope:]ffatype``
    Matches an atom with the given atop type, optionally part of the given scope.

``scope:*``
    Matches any atom in the given scope.

``expr1 & expr2 [& ...]``
    Matches an atom that satisfies all the given expressions.

``expr1 | expr2 [| ...]``
    Matches an atom that satisfies any of the given expressions.

``!expr``
    Matches an atom that does not satisfy the given expression.

``=N[%expr]``
    Matches an atom that has exactly N neighbors, that optionally match the
    given expression.

``>N[%expr]``
    Matches an atom that has more than N neighbors, that optionally match the
    given expression.

``<N[%expr]``
    Matches an atom that has less than N neighbors, that optionally match the
    given expression.

``@N``
    Matches an atom that is part of a strong ring with N atoms. **Not
    implemented yet.**

``(expr)``
    Round brackets are part of the syntax, used to override operator precedence.
    The precedence of the operators corresponds to the order of this list.

In the list above, ``expr`` can be any valid ATSELECT expression. Atom types and
scope names should not contain the following symbols: ``:``, ``%``, ``=``,
``<``, ``>``, ``@``, ``(``, ``)``, ``&``, ``|``, ``!``, and should
not start with a digit. Some examples of atom selectors:

 * ``6`` -- any carbon atom.
 * ``TPA:6`` -- a carbon atom in the TPA scope.
 * ``C3`` -- any atom with type C3.
 * ``TPA:C3`` -- an atom with type C3 in the TPA scope.
 * ``!1`` -- anything that is not a hydrogen.
 * ``C2|C3`` -- an atom of type C2 or C3.
 * ``6|7&=1%1`` or ``(6|7)&=1%1`` -- a carbon or nitrogen bonded to exactly one
   hydrogen.
 * ``>0%(6|=4)`` -- an atom bonded to at least one carbon atom or bonded to at
   least one atom with four bonds.
 * ``6&@6`` -- a Carbon atom that is part of a six-membered ring.

There are currently three ways to use the ATSELECT strings in Yaff:

1. Compile the string into a function and use it directly::

    from yaff import *
    fn = atsel_compile('C&=4')
    system = System.from_file('test.chk')
    if fn(systen, 0):
        pass
        # Do something if the first atom is a carbon with four neighbors.
        # ...

2. Get all atom indexes in a system that match a certain ATSELECT string::

    from yaff import *
    system = System.from_file('test.chk')
    indexes = system.get_indexes('C&=4')
    # The array indexes contains all the indexes of the carbon atoms with
    # four neighbors.

3. Define FF atom types in a system based on ATSELECT strings. For this purpose,
   one can normally not rely on the presence of FF atom types in the system
   object. ::

    from yaff import *
    system = System.from_file('test.chk')
    system.detect_ffatypes([
        ('H', '1'),
        ('O_water', '8&=2%1'),
    ])


Whenever one uses a compiled expression on a system that does not have
sufficient attributes, a ``ValueError`` is raised. For example, a ValueError
would be raised when one would refer to atom types in the third use case.
