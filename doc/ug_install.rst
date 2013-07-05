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

* Python >= 2.7 (including the development files): http://www.python.org/
* Numpy >= 1.0: http://numpy.scipy.org/
* Cython >= 0.15.1 : http://www.cython.org/
* h5py >= 2.0.0: http://code.google.com/p/h5py/
* matplotlib >= 1.0.0: http://matplotlib.sourceforge.net/

Most Linux distributions can install this software with just a single terminal
command.

* Ubuntu 12.4 and later::

    sudo apt-get install python-numpy cython python-h5py python-matplotlib python-nose python-sphinx

* Fedora 17 and later::

    sudo yum install numpy cython h5py python-matplotlib python-nose sphinx


Download Yaff
=============

Stable release
--------------

The latest stable release of Yaff can be downloaded here:

    http://users.ugent.be/~tovrstra/yaff/yaff-1.0.tar.gz.

Choose a suitable directory, e.g. ``~/build``, download and unpack the archive::

    mkdir -p ~/build
    cd ~/build
    wget http://users.ugent.be/~tovrstra/yaff/yaff-1.0.tar.gz
    tar -xvzf yaff-1.0.tar.gz
    cd yaff-1.0

Latest development code (experts only)
--------------------------------------

In order to get the latest development version of the source code, and to upload
your own changes, you need to work with git. Git is a version control system
that makes life easy when a group of people are working on a common source code.
All information about git (including downloads and tutorials) can be found here:
http://git-scm.com/. The official git URL of Yaff is:
git://github.com/yaff/yaff.git. In order to `clone` the public Yaff
repository, run this command::

    git clone git://github.com/molmod/yaff.git
    cd yaff

The version history can be updated with the latest patches with the following
command::

    git pull

There is also a web interface to Yaff's git repository:
https://github.com/molmod/yaff


Install Yaff
============

The following command installs Yaff into your home directory. ::

    ./setup.py install --user

You are now ready to start using Yaff!


Test your installation
======================

Execute the following commands to run the tests::

    cd
    nosetests -v yaff

If some tests fail, post the output of the tests on the `Yaff
mailing list <https://groups.google.com/forum/#!forum/ninjaff>`_.
