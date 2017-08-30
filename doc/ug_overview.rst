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

Overview of Yaff
################

Yaff is a Python library that can be used to implement all sorts of
force-field simulations. A useful simulation typically consists of four steps:

1. Specification of the molecular system that will be simulated.
2. Specification of the force field model used to compute energies and forces.
3. An (iterative) simulation protocol, such as a Molecular Dynamics or a Monte
   Carlo simulation, a geometry optimization, or yet something else.
4. Analysis of the output to extract meaningful numbers from the simulation.

Each step will be discussed in more detail in the following sections.

In Yaff, the conventional input file is replaced by an input script. This means
that one must write one or more small `main` programs that specify what type of
simulation is carried out. A minimalistic example, which covers all four steps,
is given in the file ``data/examples/000_overview/simulation.py``:

.. literalinclude:: ../yaff/examples/000_overview/simulation.py
    :lines: 26-

Yaff internally works with atomic units, although other unit systems can be used
in input and (some) output files. The units used for the screen output are
controlled with the ``log.set_unitsys`` method. Output written in HDF5
files will always be in atomic units. When output is written to a format from
other projects/programs, the units of that program/project will be used.

Numpy, Cython and h5py are used extensively in Yaff for numerical efficiency.
The examples below often use Numpy too, assuming the following import
statement::

    import numpy as np, h5py as h5
