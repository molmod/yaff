Installation
############


Disclaimer
==========

YAFF is developed and tested in modern Linux environments. The
installation instructions below are given for a Linux system only. If you want
to use YAFF on other operating systems such as Windows or OSX, you should
have a minimal computer geek status to get it working. We are always interested
in hearing from your installation adventures.


MolMod dependency
=================

`MolMod <http://molmod.github.com/molmod/>`_ is a Python library used by most
Python programs developed at the CMM. It must be installed before YAFF can
be used or installed. Installation and download instructions can be found in the
`molmod documentation <http://molmod.github.com/molmod/tutorial/install.html>`_.
The instructions below only work of the MolMod package is installed.


External dependencies
=====================

Some software packages should be installed before YAFF can be installed or
used. It is recommended to use the software package management of your Linux
distribution to install these dependencies.

The following software must be installed:

* Python 2.5, 2.6 or 2.7 (including the development files): http://www.python.org/
* Numpy >= 1.0: http://numpy.scipy.org/
* Cython >= 0.15.1 : http://www.cython.org/

Most Linux distributions can install this software with just a single terminal
command.

* Ubuntu 12.4::

    sudo apt-get install python-numpy cython

* Fedora 17::

    sudo yum install numpy cython


Installing the latest version of YAFF
=====================================

The following series of commands will download the latest version of YAFF,
and will then install it into your home directory. ::

    cd ~/build/
    git clone git://github.com/molmod/yaff.git
    (cd yaff; ./setup.py install --home=~)

You are now ready to start using YAFF!


Upgrading to the latest version of MolMod and YAFF
==================================================

In case you want to upgrade YAFF to the latests development version after
a previous install, then execute the following commands (in the same directory
that was originally used to install YAFF)::

    cd ~/build/
    (cd molmod; git pull; rm -r ~/lib*/python/molmod*; ./setup.py install --home=~)
    (cd yaff; git pull; rm -r ~/lib*/python/yaff*; ./setup.py install --home=~)


Testing your installation
=========================

For the development and testing one needs to install additional packages:

* Nosetests >= 0.11: http://somethingaboutorange.com/mrl/projects/nose/0.11.2/
* Sphinx >= 1.0: http://sphinx.pocoo.org/

Most Linux distributions can install this software with just a single terminal command:

* Ubuntu 12.4::

    sudo apt-get install python-nose python-sphinx

* Debian 5::

    su -
    apt-get install python-nose python-sphinx
    exit

* Fedora 17::

    sudo yum install python-nose sphinx

* Suse 11.2::

    sudo zypper install python-nose sphinx

Once these dependencies are installed, execute the following commands to run the
tests::

    cd ~/build/yaff
    ./cleanfiles.sh
    ./setup.py build_ext -i
    nosetests -v

If some tests fail, post the output of the tests on the `YAFF
mailing list <https://groups.google.com/forum/#!forum/yaff>`_.
