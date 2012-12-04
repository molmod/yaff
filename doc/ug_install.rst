.. _ug_sec_install:

Installation
############


Disclaimer
==========

Yaff is developed and tested on modern Linux environments. The
installation instructions below are given for a Linux system only. If you want
to use Yaff on other operating systems such as Windows or OSX, you should
have a minimal computer geek status to get it working. We are always interested
in hearing from your installation adventures.


MolMod dependency
=================

`MolMod <http://molmod.github.com/molmod/>`_ is a Python library used by most
Python programs developed at the CMM. It must be installed before Yaff can
be used or installed. Installation and download instructions can be found in the
`molmod documentation <http://molmod.github.com/molmod/tutorial/install.html>`_.
The instructions below only work if the MolMod package is installed.


External dependencies
=====================

Some software packages should be installed before Yaff can be installed or
used. It is recommended to use the software package management of your Linux
distribution to install these dependencies.

The following software must be installed:

* Python 2.5, 2.6 or 2.7 (including the development files): http://www.python.org/
* Numpy >= 1.0: http://numpy.scipy.org/
* Cython >= 0.15.1 : http://www.cython.org/
* h5py >= 2.0.0: http://code.google.com/p/h5py/
* matplotlib >= 1.0.0: http://matplotlib.sourceforge.net/

Most Linux distributions can install this software with just a single terminal
command.

* Ubuntu 12.4::

    sudo apt-get install python-numpy cython python-h5py python-matplotlib

* Fedora 17::

    sudo yum install numpy cython h5py python-matplotlib


Installing the latest version of Yaff
=====================================

The following series of commands will download the latest version of Yaff,
and will then install it into your home directory. ::

    cd ~/build/
    git clone git://github.com/molmod/yaff.git
    (cd yaff; ./setup.py install --home=~)

You are now ready to start using Yaff!


Upgrading to the latest version of MolMod and Yaff
==================================================

In case you want to upgrade Yaff to the latest development version after
a previous install, then execute the following commands::

    cd ~/build/
    (cd molmod; git pull; rm -r ~/lib*/python/molmod*; ./setup.py install --home=~)
    (cd yaff; git pull; rm -r ~/lib*/python/yaff*; ./setup.py install --home=~)


Testing your installation
=========================

For development and testing one needs to install additional packages:

* Nosetests >= 0.11: http://somethingaboutorange.com/mrl/projects/nose/0.11.2/
* Sphinx >= 1.0: http://sphinx.pocoo.org/

Most Linux distributions can install this software with just a single terminal command:

* Ubuntu 12.4::

    sudo apt-get install python-nose python-sphinx

* Fedora 17::

    sudo yum install python-nose sphinx

Once these dependencies are installed, execute the following commands to run the
tests::

    cd ~/build/yaff
    ./cleanfiles.sh
    ./setup.py build_ext -i
    nosetests -v

If some tests fail, post the output of the tests on the `Yaff
mailing list <https://groups.google.com/forum/#!forum/ninjaff>`_.
