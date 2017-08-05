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

.. _ug_sec_install:

Installation
############


Disclaimer
==========

Yaff is developed and tested on modern Linux environments. The installation instructions
below are given for a Linux system only. If you want to use Yaff on other operating
systems such as Windows or OSX, you should have a minimal computer geek status to get it
working. We are always interested in hearing from your installation adventures.


External dependencies
=====================

Some software packages should be installed before Yaff can be installed or
used. It is recommended to use the software package management of your Linux
distribution to install these dependencies.

The following software must be installed:

* Python >=2.7, <3.0 (including the development files): http://www.python.org/
* A C++ compiler e.g. gcc: http://gcc.gnu.org/
* Numpy >=1.0: http://www.numpy.org/
* Scipy >=0.17.1: http://www.scipy.org/
* Cython >=0.24.1 : http://www.cython.org/
* h5py >=2.0.0: http://code.google.com/p/h5py/
* Nosetests >=0.11: http://somethingaboutorange.com/mrl/projects/nose/0.11.2/
* matplotlib >=1.0.0: http://matplotlib.sourceforge.net/

Most Linux distributions can install this software with just a single terminal
command.

* **Ubuntu**

  .. code:: bash

      sudo apt-get install gcc g++ python-devel python-numpy cython python-h5py python-matplotlib python-nose python-scipy

* **Fedora pre 22**

  .. code:: bash

      sudo yum install gcc gcc-c++ redhat-rpm-config python-devel numpy cython h5py python-matplotlib python-nose sphinx scipy

* **Fedora 22 and later**

  .. code:: bash

      sudo dnf install gcc gcc-c++ redhat-rpm-config python-devel numpy cython h5py python-matplotlib python-nose sphinx scipy


Installation
============

You can install Yaff with pip (in your home directory):

.. code:: bash

    pip install numpy Cython --user
    pip install yaff --user

or in system wide or in a virtual environment:

.. code:: bash

    pip install numpy Cython
    pip install yaff

Alternatively, you can also install yaff with conda. See https://www.continuum.io/downloads

.. code:: bash

    conda install -c tovrstra yaff


Test your installation
======================

Execute the following commands to run the tests:

.. code:: bash

    nosetests -v yaff

If some tests fail, you can post on issue on https://github.com/molmod/yaff/issues
