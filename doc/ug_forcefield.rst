.. _ug_sec_forcefield:

Force-field models
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
optionally the forces and/or the virial tensor) for a given set of Cartesian
coordinates and/or cell parameters::

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
arguments implies that a large number of partial derivatives must be computed.

After the ``compute`` method is called, one can obtain many intermediate
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
order of the lines is not relevant when Yaff processes the file. Nevertheless,
for the sake of human readability, it is best to follow some logical ordering.
Comments start with a pound sign (``#``) and reach until the end of the line.
These comments and empty lines are ignored when processing the parameter file.

Each (non-empty) line has the following format::

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

* ``UNIT`` (may repeat): Specify the units of the parameters ``K`` and ``R0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  (Bonds are defined in the System instance.) Four
  data fields must be given: ``ffatype0``, ``ffatype1``, ``K`` and ``R0``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_water_bondharm.txt


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

* ``UNIT`` (may repeat): Specify the units of the parameters ``K`` and ``R0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  (Bonds are defined in the System instance.) Four
  data fields must be given: ``ffatype0``, ``ffatype1``, ``K`` and ``R0``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_water_bondfues.txt



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

* ``UNIT`` (may repeat): Specify the units of the parameters ``K`` and ``THETA0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Five data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``K`` and ``THETA0``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_water_bendaharm.txt



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

* ``UNIT`` (may repeat): Specify the units of the parameters ``K`` and ``COS0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Five data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``K`` and ``COS0``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_water_bendcharm.txt


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

* ``UNIT`` (may repeat): Specify the units of the parameters ``K`` and ``R0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Five data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``K`` and ``R0``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_water_ubharm.txt


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

* ``UNIT`` (may repeat): Specify the units of the parameters ``K``, ``M`` and ``PHI0``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Seven data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``ffatype3``, ``M``, ``A`` and ``PHI0``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_glycine_torsion.txt


Prefix -- INVERSION
-------------------

**Energy term:**

.. math:: E_\text{INVERSION} = \sum_{i=1}^{N_t} \frac{1}{2} A_i (1 - \cos( \chi_i ))

**Parameters:**

* :math:`A_i` (``A``): The amplitude of inversion barrier :math:`i`.

**Constants:**

* :math:`N_t`: the number of inversion terms.

**Geometry dependent variables:**

* :math:`\chi_i`: the out-of-plane angle between the plane spanned by atoms 1,2,4 and the bond between atoms 3 and 4.

**Commands:**

* ``UNIT`` (may repeat): Specify the units of the parameter ``A``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Five data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``ffatype3`` and ``A``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_formaldehyde_inversion.txt


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

* ``UNIT`` (may repeat): Specify the units of the parameters ``K``, ``R0`` and ``R1``. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given combination of atom types.
  Six data fields must be given: ``ffatype0``,
  ``ffatype1``, ``ffatype2``, ``K``, ``R0`` and ``R1``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_water_bondcross.txt

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

.. math:: \sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}

.. math:: \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}


**Parameters:**

* :math:`\sigma_i` (``SIGMA``): the (finite) distance at which the energy becomes zero (for a pair of atoms of the same type).
* :math:`\epsilon_i` (``EPSILON``): the depth of the energy minimum (for a pair of atoms of the same type).

**Constants:**

* :math:`N`: the number of atoms.
* :math:`s_{ij}`: the scaling of the interaction between atoms :math:`i` and :math:`j`.

**Geometry dependent variables:**

* :math:`d_{ij}`: the distance between atoms :math:`i` and :math:`j`.

**Commands:**

* ``UNIT`` (may repeat): Specify the units of the parameters ``SIGMA`` and ``EPSILON``. See
  :ref:`sub_sub_sec_general_commands`.
* ``SCALE`` (may repeat): Specify the scaling of short-ranged interactions. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given atom type.
  Three data fields must be given: ``ffatype``, ``SIGMA`` and ``EPSILON``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_water_lj.txt


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

* :math:`\sigma_i` (``SIGMA``): the (finite) distance at which the energy becomes zero (for a pair of atoms of the same type).
* :math:`\epsilon_i` (``EPSILON``): the depth of the energy minimum (for a pair of atoms of the same type).

**Constants:**

* :math:`N`: the number of atoms.
* :math:`s_{ij}`: the scaling of the interaction between atoms :math:`i` and :math:`j`.

**Geometry dependent variables:**

* :math:`d_{ij}`: the distance between atoms :math:`i` and :math:`j`.

**Commands:**

* ``UNIT`` (may repeat): Specify the units of the parameters ``SIGMA`` and ``EPSILON``. See
  :ref:`sub_sub_sec_general_commands`.
* ``SCALE`` (may repeat): Specify the scaling of short-ranged interactions. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given atom type.
  Four data fields must be given: ``ffatype``, ``SIGMA``, ``EPSILON`` and ``ONLYPAULI``.
  The last data field corresponds to an undocumented feature. Set it to ``0`` to
  get the original MM3 form.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_water_mm3.txt


Prefix -- EXPREP
----------------

**Description:** an exponential repulsion term.

**Energy:**

.. math:: E_\text{EXPREP} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} A_{ij} \exp(-B_{ij} d_{ij})

The pair parameters can be provided explicitly, or can be derived from atomic
parameters using two possible mixing rules for each parameter:

* ``GEOMETRIC`` mixing for :math:`A_{ij}`: :math:`A_{ij} = \sqrt{A_i A_j}`

* ``GEOMETRIC_COR`` mixing for :math:`A_{ij}`: :math:`\ln A_{ij} = (\ln A_i + \ln A_j)\frac{1-x\vert\ln(A_i/A_j)\vert}{2}` where :math:`x` is a configurable parameter

* ``ARITHMETIC`` mixing for :math:`B_{ij}`: :math:`B_{ij} = \frac{B_i + B_j}{2}`

* ``ARITHMETIC_COR`` mixing for :math:`B_{ij}`: :math:`B_{ij} = (B_i + B_j)\frac{1-x\vert\ln(A_i/A_j)\vert}{2}` where :math:`x` is a configurable parameter

**Parameters:**

* :math:`A_i` or :math:`A_{ij}` (``A``): the amplitude of the exponential repulsion.
* :math:`B_i` or :math:`B_{ij}`(``B``): the decay of the exponential repulsion.

**Constants:**

* :math:`N`: the number of atoms.
* :math:`s_{ij}`: the scaling of the interaction between atoms :math:`i` and :math:`j`.

**Geometry dependent variables:**

* :math:`d_{ij}`: the distance between atoms :math:`i` and :math:`j`.

**Commands:**

* ``UNIT`` (may repeat): Specify the units of the parameters ``A`` and ``B``. See
  :ref:`sub_sub_sec_general_commands`.
* ``SCALE`` (may repeat): Specify the scaling of short-ranged interactions. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given atom type.
  Three data fields must be given: ``ffatype``, ``A`` and ``B``.
* ``CPARS`` (may repeat): Specify parameters for a given combination of atom types. This
  overrides parameters derived from mixing rules.
  Four data fields must be given: ``ffatype0``, ``ffatype1``, ``A`` and ``B``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_fake_exprep1.txt


Prefix -- DAMPDISP
------------------

**Description:** a dispersion term with optional Tang-Toennies damping.

**Energy:**

.. math:: E_\text{DAMPDISP} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} C_{6,ij} f_\text{damp,6}(d_{ij}) d_{ij}^{-6}

where the damping factor :math:`f_\text{damp}(d_{ij})` is optional. When used
it has the Tang-Toennies form:

.. math:: f_\text{damp,n}(d_{ij}) = 1 - \exp(-B_{ij}r)\sum_{k=0}^n\frac{(B_{ij}r)^k}{k!}

The pair parameters :math:`C_{6,ij}` and :math:`B_{ij}` are derived from atomic
parameters using mixing rules, unless they are provided explicitly for a given
pair of atom types. These are the mixing rules:

.. math:: C_{6,ij} = \frac{2 C_{6,i} C_{6,j}}{\left(\frac{V_j}{V_i}\right)^2 C_{6,i} + \left(\frac{V_i}{V_j}\right)^2 C_{6,j}}

.. math:: B_{ij} = \frac{B_i+B_j}{2}

**Parameters:**

* :math:`C_i` or :math:`C_{ij}` (``C``): the strength of the dispersion interaction.
* :math:`B_i` or :math:`B_{ij}` (``B``): the decay of the damping function. When
  this parameter is zero, the damping is not applied.
* :math:`V_i` (``VOL``): the atomic volume parameter used in the mixing rule for the
  :math:`C_{ij}`.

**Constants:**

* :math:`N`: the number of atoms.
* :math:`s_{ij}`: the scaling of the interaction between atoms :math:`i` and :math:`j`.

**Geometry dependent variables:**

* :math:`d_{ij}`: the distance between atoms :math:`i` and :math:`j`.

**Commands:**

* ``UNIT`` (may repeat): Specify the units of the parameters ``C6``, ``B`` and ``VOL``. See
  :ref:`sub_sub_sec_general_commands`.
* ``SCALE`` (may repeat): Specify the scaling of short-ranged interactions. See
  :ref:`sub_sub_sec_general_commands`.
* ``PARS`` (may repeat): Specify parameters for a given atom type.
  Four data fields must be given: ``ffatype``, ``C6``, ``B`` and ``VOL``.
* ``CPARS`` (may repeat): Specify parameters for a given combination of atom types. This
  overrides parameters derived from mixing rules.
  Four data fields must be given: ``ffatype0``, ``ffatype1``, ``C6`` and ``B``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_fake_dampdisp1.txt


Prefix -- FIXQ
--------------

**Description:** Electrostatic interactions with constant atomic point charges.

**Energy:**

.. math:: E_\text{FIXQ} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} \frac{q_i q_j}{d_{ij}} \textrm{erf}\left(\frac{d_{ij}}{R_{ij}}\right)

with :math:`R_{ij}^2 = R_i^2 + R_j^2`. When :math:`R_{ij}=0`, this simplifies
to the familiar expression for point charges:

.. math:: E_\text{FIXQ} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} \frac{q_i q_j}{d_{ij}}

The charges are derived from so-called `pre-charges` (:math:`q_{0,i}`) and
`bond charge increments` (:math:`p_{i,j}`) as follows:

.. math:: q_i = q_{0,i} + \sum_{j \text{ bonded to } i}p_{ij}

where the summation is limited to atoms :math:`j` that are bonded to atom
:math:`i`. The parameter :math:`p_{i,j}` represents the amount of charge
transfered from atom :math:`i` to :math:`j`. Hence
:math:`p_{i,j}=-p_{j,i}`. The pre-charge is the charge on an atom when
it is not bonded to any other atom. From a physical perspective, the pre-charge
should always be integer, which would also impose integer charges on molecules.
However, one is free to follow other conventions for the sake of convenience.
The charge density :math:`\rho_i(\mathbf{r})` for a charge with radius
:math:`R_i` is given by the Gaussian distribution:

.. math:: \rho_i(\mathbf{r}) = \frac{q_i}{\left(\sqrt{\pi}R_i\right)^3}\exp{-\frac{|\mathbf{r}-\mathbf{r}_i|^2}{R_i^2}}

When :math:`R_i=0`, the distribution becomes a point charge:

.. math::  \rho_i(\mathbf{r}) = q_i\delta\left( \mathbf{r}-\mathbf{r}_i \right)

**Parameters:**

* :math:`q_{0,i}` (``Q0``): the pre-charge
* :math:`p_{ij}` (``P``): the bond charge increment
* :math:`R_{i}` (``R``): the charge radius

**Constants:**

* :math:`N`: the number of atoms.
* :math:`s_{ij}`: the scaling of the interaction between atoms :math:`i` and :math:`j`.

**Geometry dependent variables:**

* :math:`d_{ij}`: the distance between atoms :math:`i` and :math:`j`.

**Commands:**

* ``UNIT`` (may repeat): Specify the units of the parameters ``Q0``, ``P`` and
  ``R``. See :ref:`sub_sub_sec_general_commands`.
* ``SCALE`` (may repeat): Specify the scaling of short-ranged interactions. See
  :ref:`sub_sub_sec_general_commands`.
* ``DIELECTRIC``: Specify scalar relative permittivity, must be at least 1.0.
* ``ATOM`` (may repeat): Specify the pre-charge and radius for a given atom type.
  Three data fields must be given: ``ffatype``, ``Q0``, ``R``.
* ``BOND`` (may repeat): Specify a bond charge increment for a given combination of atom types.
  Three data fields must be given: ``ffatype0``, ``ffatype1`` and ``P``.

**Example**:

.. literalinclude:: ../yaff/data/test/parameters_water_fixq.txt

.. _sub_sub_sec_general_commands:


General commands
----------------

**UNIT**

The ``UNIT`` command is used to specify the units of the parameters given in
the parameter file. The format is as follows::

    PREFIX:UNIT PARAMETER_NAME UNIT_NAME

where ``PARAMETER_NAME`` refers to one of the parameters discussed above, e.g.
``K``, ``R0``, etc. The ``UNIT_NAME`` may be any mathematical expression
involving the following constants: ``coulomb``, ``kilogram``, ``gram``,
``miligram``, ``unified``, ``meter``, ``centimeter``, ``milimeter``,
``micrometer``, ``nanometer``, ``angstrom``, ``picometer``, ``liter``,
``joule``, ``calorie``, ``electronvolt``, ``newton``, ``second``, ``hertz``,
``nanosecond``, ``femtosecond``, ``picosecond``, ``pascal``, ``e``.

A unit must be specified for each parameter. There are no default units. This
convention is introduced to make sure that there can be no confusion about the
units of the parameters.


**SCALE**

The ``SCALE`` command is used to determine how pairwise interactions are scaled
for atoms that are involved in bond, bend or torsion terms. The command adheres
to the following format::

    PREFIX:SCALE N FACTOR

where ``N`` is ``1``, ``2`` or ``3`` and represents the number bonds between
two atoms to which the scaling of the pairwise interaction is applied. ``1``
is for bonded atoms, ``2`` applies to atoms separated by two bonds and ``3``
is used for atoms separated by three bonds. The ``FACTOR`` determines the amount
of scaling and must lie in the range [0.0,1.0]. When set to zero, the pairwise
term is completely disabled.


Example force field file
========================

The following is an example for a reasonable non-polarizable water FF. The
parameters were generated with an old beta version of our in-house parameter
calibration software. Don't expect it to be a great water model!

.. literalinclude:: ../yaff/data/test/parameters_water.txt


Beyond force field parameter files
==================================

One does not have to use parameter files to construct force fields. It is also
possible to construct them with some Python code, which can be useful in some
corner cases.

The larger part of the force field `parts` in Yaff are not aware
of the actual atom types. ``EXPREP`` and ``DAMPDISP`` are the only exceptions.
This means that one may construct a force field in which every bond has
different parameters, irrespective of the atom types involved. The following
simple example illustrates this by creating an `unphysical` model for water,
were the bond lengths of the two O-H bonds differ::

    # A system object for a single water molecule
    system = System(
        numbers=np.array([8, 1, 1]),
        pos=np.array([[-4.583, 5.333, 1.560], [-3.777, 5.331, 0.943],
                      [-5.081, 4.589, 1.176]])*angstrom,
        ffatypes=['O', 'H', 'H'],
        bonds=np.array([[0, 1], [0, 2]]),
    )
    # A valence force field with only two harmonic bond terms, which have
    # different rest value parameters.
    part = ForcePartValence(system)
    part.add_term(Harmonic(fc=1.0, rv=1.0*angstrom, Bond(0, 1)))
    part.add_term(Harmonic(fc=1.0, rv=1.1*angstrom, Bond(0, 2)))
    ff = ForceField(system, [part])

Modifying or constructing force fields at this level of detail may be useful
in the following cases:

* When one wants to perform restrained molecular dynamics simulations, this
  approach allows one to add a restraint term. The following example assumes
  that a ``valence_part`` is already present after the force field is created
  with a parameter file::

    ff = ForceField(system, 'parameters.txt')
    ff.part_valence.add_term(Harmonic(fc=0.3, rv=2.1*angstrom, Bond(15, 23)))

* One may also derive a force field without paying attention to transferability
  of parameters, e.g. when a force field is designed for one specific molecule.
  In that case, each bond, bend, etc. may have different parameters (except for
  symmetry considerations). This is similar to the elastic network models in
  coarse-grained protein simulations.

* In some cases, one is interested in constructing a molecular system with
  certain geometric prescriptions. Building such a structure from scratch can
  be very difficult for complex systems. One may design a non-physical force
  field such that the optimal geometry satisfies the geometrical criteria.

* Several combinations of internal coordinates and valence energy terms are not
  supported through the parameter file, simply because they are uncommon. The
  above example shows how one can program any combination of internal coordinate
  and valence energy term in the force field.
