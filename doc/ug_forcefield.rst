Force field models
##################


Introduction
============

Once the system is defined (see :ref:`ug_system`), one can continue with the
specification of the force field model. The simplest way to create a force-field
is as follows::

    ff = ForceField.generate(system, 'parameters.txt')

where the file ``parameters.txt`` contains all force field parameters. See :ref:`ug_sec_ff_par_format`
for a detailed description of the file format. Additional `technical`
parameters that determine the behavior of the force field, such as the
real-space cutoff, the verlet skin, and so on, may be specified as keyword
arguments in the ``generate`` method. See
:meth:`yaff.pes.ff.ForceField.generate` for a detailed description of the
``generate`` method.

Once an ``ff`` object is created, it can be used to evaluate the energy (and
optionally the forces) for a given set of Cartesian coordinates and/or cell
parameters::

    # change the atomic positions and cell parameters
    ff.update_pos(new_pos)
    ff.update_rvecs(new_rvecs)
    # compute the energy
    new_energy = ff.compute()

One may also allocate arrays to store the derivative of the energy towards
the atomic positions and uniform deformations of the system::

    # allocate arrays for the Cartesian gradient of the energy and the virial
    # tensor.
    gpos = np.zeros(system.pos.shape, float)
    vtens = np.zeros((3,3), float)
    # change the atomic positions and cell parameters
    ff.update_pos(new_pos)
    ff.update_rvecs(new_rvecs)
    # compute the energy
    new_energy = ff.compute(gpos, vtens)

This will take a little more CPU time because the presence of the optional
arguments implies that a lot of partial derivatives must be computed.

After the ``compute`` method is called, one can obtain a lot of intermediate
results by accessing attributes of the ``ff`` object. Some examples::

    print ff.part_pair_ei.energy/kjmol
    print ff.part_valence.gpos
    print ff.part_ewald_cor.vtens

Depending on the system and the contents of the file ``parameters.txt`` some
``part_*`` attributes may not be present. All parts are also accessible through
the list ``ff.parts``.

Instead of using the ``ForceField.generate`` method, one may also construct all
the parts of the force field manually. However, this can become very tedious.
This is a simple example of a Lennard-Jones force field::

    system = System(
        numbers=np.array([18]*10),
        pos=np.random.uniform(0, 10*angstrom, (10,3)),
        ffatypes=['Ar']*10,
        bonds=None,
        rvecs=np.identity(3)*10*angstrom,
    )
    sigmas = np.array([3.98e-4]*10),
    epsilons = np.array([6.32]*10),
    pair_pot_lj = PairPotLJ(sigmas, epsilons, rcut=15*angstrom, smooth=True)
    nlists = NeighborLists(system)
    scalings = Scalings(system.topology)
    part_pair_lj = ForcePartPair(system, nlists, scalings, pair_pot_lj)
    ff = ForceField(system, [part_pair_lj], nlists)


.. _ug_sec_ff_par_format:


Force field parameter file format
=================================

The force field parameter file has a case-insensitive line-based format. The
order of the lines is not relevant for processing by Yaff. Nevertheless, for the sake of
human readability, it is best to follow some logical ordering.
Comments start with a pound sign (``#``) and reach until the end of the line.
These comments and empty lines are ignored when processing the parameter file.

Each line has the following structure::

    PREFIX:COMMAND DATA

where ``PREFIX`` and ``COMMAND`` do not contain white spaces and ``DATA`` may
consist of multiple words and/or numbers. Each prefix corresponds to a certain
type of energy term. Each command for a given prefix configures certain
properties and parameters for that type of energy terms.

When a PREFIX is present in a parameter file, all known commands for that prefix
must be included. Some commands may repeat. This imposes a complete definition
of a type of energy term without implicit default settings. This requirement
guarantees that the parameter file is self-explaining.

All possible prefixes and corresponding commands are documented in the following
subsections. Some commands have comparable behavior for different prefixes and
are therefore document in a separate subsection,
:ref:`sub_sub_sec_general_commands`.


Prefix -- BONDHARM
------------------

**Energy term:**

.. math:: E_\text{BONDHARM} = \sum_{i=1}^{N_b} \frac{1}{2} K_i (r_i - R_{0,i})^2

**Parameters:**

* :math:`K_i` (``K``): the force constant parameter of bond :math:`i`.
* :math:`R_{0,i}` (``R0``): the rest value parameter of bond :math:`i`.

**Constants:**

* :math:`N_b`: the number of bonds.

**Geometry dependent variables:**

* :math:`r_i`: the length of bond :math:`i`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``K`` and ``R0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  (Bonds are defined in the System instance.) Four
  data fields must be given: ``ffatype0``, ``ffatype1``, ``K`` and ``R0``.

**Example**:

.. literalinclude:: ../input/parameters_water_bondharm.txt


Prefix -- BONDFUES
------------------

**Energy term:**

.. math:: E_\text{BONDFUES} = \sum_{i=1}^{N_b} \frac{1}{2} K_i R^2_{0,i} \left(1 - \frac{R_{0,i}}{r_i}\right)^2

**Parameters:**

* :math:`K_i` (``K``): the force constant parameter of bond :math:`i`.
* :math:`R_{0,i}` (``R0``): the rest value parameter of bond :math:`i`.

**Constants:**

* :math:`N_b`: the number of bonds.

**Geometry dependent variables:**

* :math:`r_i`: the length of bond :math:`i`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``K`` and ``R0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  (Bonds are defined in the System instance.) Four
  data fields must be given: ``ffatype0``, ``ffatype1``, ``K`` and ``R0``.

**Example**:

.. literalinclude:: ../input/parameters_water_bondfues.txt



Prefix -- BENDAHARM
-------------------

**Energy term:**

.. math:: E_\text{BENDAHARM} = \sum_{i=1}^{N_a} \frac{1}{2} K_i (\theta_i - \Theta_{0,i})^2

**Parameters:**

* :math:`K_i` (``K``): the force constant parameter of bend :math:`i`.
* :math:`\Theta_{0,i}` (``THETA0``): the angle rest value parameter of bend :math:`i`.

**Constants:**

* :math:`N_a`: the number of bending angles.

**Geometry dependent variables:**

* :math:`\theta_i`: the angle of bend :math:`i`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``K`` and ``THETA0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Five data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``K`` and ``THETA0``.

**Example**:

.. literalinclude:: ../input/parameters_water_bendaharm.txt



Prefix -- BENDCHARM
-------------------

**Energy term:**

.. math:: E_\text{BENDAHARM} = \sum_{i=1}^{N_a} \frac{1}{2} K_i (\cos(\theta_i) - C_{0,i})^2

**Parameters:**

* :math:`K_i` (``K``): the force constant parameter of bend :math:`i`.
* :math:`C_{0,i}` (``COS0``): the cosine rest value parameter of bend :math:`i`.

**Constants:**

* :math:`N_a`: the number of bending angles.

**Geometry dependent variables:**

* :math:`\cos(\theta_i)`: the cosine of the angle of bend :math:`i`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``K`` and ``COS0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Five data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``K`` and ``COS0``.

**Example**:

.. literalinclude:: ../input/parameters_water_bendcharm.txt


Prefix -- UBHARM
----------------

**Energy term:**

.. math:: E_\text{UBHARM} = \sum_{i=1}^{N_a} \frac{1}{2} K_i (r_i - R_{0,i})^2

**Parameters:**

* :math:`K_i` (``K``): the force constant parameter of the Urey-Bradley term :math:`i`.
* :math:`R_{0,i}` (``R0``): the rest value parameter of the Urey-Bradley term :math:`i`.

**Constants:**

* :math:`N_a`: the number of bending angles.

**Geometry dependent variables:**

* :math:`r_i`: the distance between the two outermost atoms in bending angle :math:`i`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``K`` and ``R0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Five data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``K`` and ``R0``.

**Example**:

.. literalinclude:: ../input/parameters_water_ubharm.txt


Prefix -- TORSION
-----------------

**Energy term:**

.. math:: E_\text{TORSION} = \sum_{i=1}^{N_t} \frac{1}{2} A_i (1 - \cos(M_i (\phi_i-\Phi_{0,i})))

**Parameters:**

* :math:`M_i` (``M``): The multiplicity of the torsional potential.
* :math:`A_i` (``A``): The amplitude of torsional barrier :math:`i`.
* :math:`\Phi_{0,i}` (``PHI0``): The location of the (or a) minimum in the torsional potential :math:`i`.

**Constants:**

* :math:`N_t`: the number of torsional terms.

**Geometry dependent variables:**

* :math:`\phi_i`: the dihedral angle of torsional term :math:`i`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``K``, ``M`` and ``PHI0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Seven data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``ffatype3``, ``M``, ``A`` and ``PHI0``.

**Example**:

.. literalinclude:: ../input/parameters_glycine_torsion.txt


Prefix -- BONDCROSS
-------------------

**Energy term:**

.. math:: E_\text{BONDCROSS} = \sum_{i=1}^{N_a} \frac{1}{2} K_i (r_{0,i} - R_{0,i})*(r_{1,i} - R_{1,i})

**Parameters:**

* :math:`K_i` (``K``): the off-diagonal force constant of cross term :math:`i`.
* :math:`R_{0,i}` (``R0``): the rest value parameter for the first bond in angle :math:`i`.
* :math:`R_{0,i}` (``R1``): the rest value parameter for the second bond in angle :math:`i`.

**Constants:**

* :math:`N_a`: the number of bending angles.

**Geometry dependent variables:**

* :math:`r_{0,i}`: the first bond length in angle :math:`i`.
* :math:`r_{1,i}`: the second bond length in angle :math:`i`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``K``, ``R0`` and ``R1``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Six data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``K``, ``R0`` and ``R1``.

**Example**:

.. literalinclude:: ../input/parameters_water_bondcross.txt

**Note**:

In the case of symmetric angles, i.e. with the same ffatypes for the outermost
angles, R0 has to be equal to R1. When the outermost ffatypes are different,
``R0`` corresponds to the bond between ``ffatype0`` and ``ffatype1`` and ``R1``
corresponds to the bond between ``ffatype1`` and ``ffatype2``.


Prefix -- LJ
------------

**Description:** the traditional Lennard-Jones potential.

**Energy term:**

.. math:: E_\text{LJ} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} 4 s_{ij} \epsilon_{ij} \left[
          \left(\frac{\sigma_{ij}}{d_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{d_{ij}}\right)^6
          \right]

with

.. math:: \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}

.. math:: \sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}

**Parameters:**

* :math:`\epsilon_i` (``EPSILON``): the depth of the energy minimum (for a pair of atoms of the same type).
* :math:`\sigma_i` (``SIGMA``): the (finite) distance at which the energy becomes zero (for a pair of atoms of the same type).

**Constants:**

* :math:`N`: the number of atoms.
* :math:`s_{ij}`: the scaling of the interaction between atoms :math:`i` and :math:`j`.

**Geometry dependent variables:**

* :math:`d_{ij}`: the distance between atoms :math:`i` and :math:`j`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``EPSILON`` and ``SIGMA``. See
  :ref:`sub_sub_sec_general_commands`.
* ``SCALE`` (may repeat): Specify the scaling of short-ranged interactions. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given atom type.
  Four data fields must be given: ``ffatype0``,
  ``ffatype1``, ``EPSILON`` and ``SIGMA``.

**Example**:

.. literalinclude:: ../input/parameters_water_lj.txt


Prefix -- MM3
-------------

**Description:** the MM3 variant of the Lennard-Jones potential.

**Energy:**

.. math:: E_\text{MM3} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} \epsilon_{ij} \left[
          1.84\times10^{5} \exp\left(\frac{\sigma_{ij}}{d_{ij}}\right) - 2.25\left(\frac{\sigma_{ij}}{d_{ij}}\right)^6
          \right]

with

.. math:: \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}

.. math:: \sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}

**Parameters:**

* :math:`\epsilon_i` (``EPSILON``): the depth of the energy minimum (for a pair of atoms of the same type).
* :math:`\sigma_i` (``SIGMA``): the (finite) distance at which the energy becomes zero (for a pair of atoms of the same type).

**Constants:**

* :math:`N`: the number of atoms.
* :math:`s_{ij}`: the scaling of the interaction between atoms :math:`i` and :math:`j`.

**Geometry dependent variables:**

* :math:`d_{ij}`: the distance between atoms :math:`i` and :math:`j`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``EPSILON`` and ``SIGMA``. See
  :ref:`sub_sub_sec_general_commands`.
* ``SCALE`` (may repeat): Specify the scaling of short-ranged interactions. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given atom type.
  Four data fields must be given: ``ffatype0``,
  ``ffatype1``, ``EPSILON`` and ``SIGMA``.

**Example**:

.. literalinclude:: ../input/parameters_water_mm3.txt


Prefix -- EXPREP
----------------

**Description:** an exponential repulsion term.


**Energy:**

.. math:: E_\text{EXPREP} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} A_{ij} \exp(-B_{ij} r_{ij})

The pair parameters can be provided explicitly, or can be derived from atomic
parameters using two possible mixing rules for each parameter:

* ``GEOMETRIC`` mixing for :math:`A_{ij}`: :math:`A_{ij} = \sqrt{A_i A_j}`

* ``GEOMETRIC_COR`` mixing for :math:`A_{ij}`: :math:`\ln A_{ij} = (\ln A_i + \ln A_j)\frac{1-x\vert\ln(A_i/A_j)\vert}{2}` where :math:`x` is a configurable parameter

* ``ARITHMETIC`` mixing for :math:`B_{ij}`: :math:`B_{ij} = \frac{B_i + B_j}{2}`

* ``ARITHMETIC_COR`` mixing for :math:`B_{ij}`: :math:`B_{ij} = (B_i + B_j)\frac{1-x\vert\ln(A_i/A_j)\vert}{2}` where :math:`x` is a configurable parameter

**Parameters:**

* :math:`A_i` or :math:`A_ij` (``A``): the amplitude of the exponential repulsion.
* :math:`B_i` or :math:`B_ij`(``B``): the decay of the exponential repulsion.

**Constants:**

* :math:`N`: the number of atoms.
* :math:`s_{ij}`: the scaling of the interaction between atoms :math:`i` and :math:`j`.

**Geometry dependent variables:**

* :math:`d_{ij}`: the distance between atoms :math:`i` and :math:`j`.

**Commands:**

* ``UNITS`` (may repeat): Specify the units of the parameters ``A`` and ``B``. See
  :ref:`sub_sub_sec_general_commands`.
* ``SCALE`` (may repeat): Specify the scaling of short-ranged interactions. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given atom type.
  Three data fields must be given: ``ffatype0``, ``A`` and ``B``.
* ``CPARS`` (may repeat): Specify parameters for a given combination of atom types. This
  overrides parameters derived from mixing rules.
  Four data fields must be given: ``ffatype0``, ``ffatype1``, ``A`` and ``B``.

**Example**:

.. literalinclude:: ../input/parameters_fake_exprep1.txt


Prefix -- DAMPDISP
------------------

**Description:** a dispersion term with optional Tang-Toenies damping.

**Example**:

.. literalinclude:: ../input/parameters_fake_dampdisp1.txt


Prefix -- FIXQ
--------------

**Description:** Electrostatic interactions with fixed atomic partial point charges.

**Example**:

.. literalinclude:: ../input/parameters_water_fixq.txt


.. _sub_sub_sec_general_commands:

General commands
----------------





Example force field file
========================

The following is an example for a reasonable non-polarizable water FF. The
parameters were generated with an old beta version of our in-house parameter
calibration software. Don't expect it to be a great water model!

.. literalinclude:: ../input/parameters_water.txt


Beyond force field parameter files
==================================
