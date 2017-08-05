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

Yaff Documentation
##################

Yaff stands for "Yet another force field". It is a pythonic force-field (FF) code
used at the `Center for Molecular Modeling <http://molmod.ugent.be/>`_ (CMM) to
test-drive new FF models. The original motivation to develop Yaff was to provide
a good reference implementation of the force fields developed at the CMM. In its
current version, Yaff is general and flexible enough to handle a large variety
of force field models.

User Guide
==========

This guide is a gentle introduction to Yaff with many examples.

.. toctree::
   :maxdepth: 2
   :numbered:

   ug_features.rst
   ug_install.rst
   ug_troubles.rst
   ug_cite.rst
   ug_overview.rst
   ug_system.rst
   ug_atselect.rst
   ug_forcefield.rst
   ug_sampling.rst
   ug_analysis.rst
   ug_tune.rst


Tutorials
=========

.. toctree::
   :maxdepth: 2
   :numbered:

   tu_silica.rst


Reference Guide
===============

This guide is mostly generated automatically based on the doc strings in the
source code.

.. toctree::
   :maxdepth: 2
   :numbered:

   rg_yaff.rst
   rg_yaff_pes.rst
   rg_yaff_sampling.rst
   rg_yaff_analysis.rst
   rg_yaff_conversion.rst
   rg_yaff_tune.rst


Development information
=======================

The git repository is hosted at `Github <https://github.com/molmod/yaff>`_.

.. toctree::
   :maxdepth: 1
   :numbered:

   dg_todo.rst
   dg_backprop.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
