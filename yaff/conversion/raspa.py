# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
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
'''raspa

    Reading/writing of RASPA output/input files
'''

import numpy as np
import os

from yaff.system import System
from yaff.pes.ff import ForceField
from yaff.pes.eos import PREOS
from yaff.log import log
from yaff.external.lammpsio import get_lammps_ffatypes

from molmod.units import kcalmol, angstrom, kelvin, amu, pascal, deg
from molmod.constants import boltzmann
from molmod.periodic import periodic

__all__ = ['write_raspa_input','read_raspa_loading']

def write_raspa_input(guests, parameters, host=None, workdir='.',
        guestdata=None, hostname='host'):
    """
       Prepare input files that can be used to run RASPA simulations. Only a
       small subset of the full capabilities of RASPA can be explored.

       **Arguments:**

       guests
            A list specifying the guest molecules. Each entry can be one of
            two types: (i) the filename of a system file
            describing one guest molecule, (ii) a System instance of
            one guest molecule

       parameters
            Force-field parameters describing guest-guest and optionally
            host-guest interaction.
            Three types are accepted: (i) the filename of the parameter
            file, which is a text file that adheres to YAFF parameter
            format, (ii) a list of such filenames, or (iii) an instance of
            the Parameters class.

       **Optional arguments:**

       host
            Two types are accepted: (i) the filename of a system file
            describing the host system, (ii) a System instance of the host

       workdir
            The directory where output files will be placed.

       guestdata
            List with the same length as guests. Each entry contains a tuple
            either looking as (name, Tc, Pc, omega) or (name). In the latter
            case, the parameters Tc, Pc, and omega will be loaded from a data
            file based on the name. Otherwise, these will be left blank in the
            input file.
    """
    # Load the guest Systems
    guests = [System.from_file(guest) if isinstance(guest, str) else guest
                        for guest in guests]
    for guest in guests:
        assert isinstance(guest, System)
    if guestdata is None:
        guestdata = [("guest%03d"%iguest,) for iguest in range(len(guests))]
    assert len(guests)==len(guestdata)
    # Load the host System
    if host is not None:
        if isinstance(host, str):
            host = System.from_file(host)
        assert isinstance(host, System)
        complex = host
    # Merge all systems
    for guest in guests:
        complex = complex.merge(guest)
    # Generate the ForceField, we don't really care about settings such as
    # cutoffs etc. We only need the parameters in the end
    ff = ForceField.generate(complex, parameters)
    # Write the force-field parameters
    write_raspa_forcefield(ff, workdir)
    # Write masses and charges of all atoms
    ffatypes, ffatype_ids = write_pseudo_atoms(ff, workdir)
    # Write the guest information
    if host==None: counter = 0
    else: counter = host.natom
    for iguest, (guest, data) in enumerate(zip(guests, guestdata)):
        write_guest(guest, data, workdir,
         [ffatypes[iffa] for iffa in ffatype_ids[counter:counter+guest.natom]])
        counter += guest.natom
    # Write the host coordinates to a cif file
    if host is not None:
        dump_cif(host, os.path.join(workdir, '%s.cif'%hostname),
            ffatypes=[ffatypes[iffa] for iffa in ffatype_ids[:host.natom]])
    # A fairly standard input file for GCMC simulations
    dump_input(workdir, hostname, [data[0] for data in guestdata])


def dump_input(workdir, hostname, guestnames):
    """
        Write an input file for RASPA GCMC simulations, which can be used as a
        template for actual simulations.
    """
    with open(os.path.join(workdir,'simulation.template'),'w') as f:
        f.write("SimulationType                MonteCarlo\n")
        f.write("NumberOfCycles                100000\n")
        f.write("NumberOfInitializationCycles  0\n")
        f.write("PrintEvery                    1000\n\n")
        f.write("ContinueAfterCrash            no\n")
        f.write("WriteBinaryRestartFileEvery   1000\n\n")
        f.write("Forcefield\n")
        f.write("RemoveAtomNumberCodeFromLabel no\n")
        f.write("CutOffVDW                     12.0\n")
        f.write("ChargeMethod                  ewald\n")
        f.write("EwaldPrecision                1e-6\n")
        f.write("OmitAdsorbateAdsorbateCoulombInteractions no\n\n")
        f.write("Framework 0\n")
        f.write("FrameworkName                 %s\n"%hostname)
        f.write("UnitCells   1 1 1\n\n")
        f.write("ExternalTemperature 298.0\n")
        f.write("ExternalPressure  300.0\n\n")
        f.write("ComputeDensityProfile3DVTKGrid yes\n")
        f.write("WriteDensityProfile3DVTKGridEvery 1000\n")
        f.write("DensityProfile3DVTKGridPoints 150 150 150\n")
        f.write("AverageDensityOverUnitCellsVTK yes\n")
        f.write("DensityAveragingTypeVTK UnitCell\n\n")
        f.write("Movies yes\n")
        f.write("WriteMoviesEvery 5000\n\n")
        for guestname in guestnames:
            f.write("Component 0 MoleculeName             %s\n"%(guestname))
            f.write("            MoleculeDefinition\n")
            f.write("            TranslationProbability   0.3333333333\n")
            f.write("            RotationProbability      0.3333333333\n")
            f.write("            ReinsertionProbability   0.3333333333\n")
            f.write("            SwapProbability          1.0\n")
            f.write("            CreateNumberOfMolecules  0\n")


def dump_cif(host, fn, ffatypes=None):
    '''Write a CIF file

       **Arguments:**

       host
            A System instance

       fn
            The name of the new CIF file.

       **Optional arguments:**

       ffatypes
            A NumPy array containing atomtypes. If not give, each atom will be
            assigned a unique atom type
    '''
    if host.cell.nvec != 3:
        raise TypeError('The CIF format only supports 3D periodic systems.')
    symbols = [periodic[host.numbers[i]].symbol for i in range(host.natom)]
    if ffatypes == None:
        ffatypes = [symbols[i]+str(i+1) for i in range(host.natom)]
    assert len(ffatypes)==host.natom
    # Conversion to fractional coordinates
    frac = np.einsum('ab,ib->ia',host.cell.gvecs,host.pos)
    with open(fn, 'w') as f:
        f.write('data_\n')
        f.write('_symmetry_space_group_name_H-M       \'P1\'\n')
        f.write('_audit_creation_method            \'Yaff\'\n')
        f.write('_symmetry_Int_Tables_number       1\n')
        f.write('_symmetry_cell_setting            triclinic\n')
        f.write('loop_\n')
        f.write('_symmetry_equiv_pos_as_xyz\n')
        f.write('  x,y,z\n')
        lengths, angles = host.cell.parameters
        f.write('_cell_length_a     %12.6f\n' % (lengths[0]/angstrom))
        f.write('_cell_length_b     %12.6f\n' % (lengths[1]/angstrom))
        f.write('_cell_length_c     %12.6f\n' % (lengths[2]/angstrom))
        f.write('_cell_angle_alpha  %12.6f\n' % (angles[0]/deg))
        f.write('_cell_angle_beta   %12.6f\n' % (angles[1]/deg))
        f.write('_cell_angle_gamma  %12.6f\n' % (angles[2]/deg))
        f.write('loop_\n')
        f.write('_atom_site_label\n')
        f.write('_atom_site_type_symbol\n')
        f.write('_atom_site_fract_x\n')
        f.write('_atom_site_fract_y\n')
        f.write('_atom_site_fract_z\n')
        for i in range(host.natom):
            f.write('%10s %3s % 12.6f % 12.6f % 12.6f\n' %
             (ffatypes[i], symbols[i], frac[i,0], frac[i,1], frac[i,2]))


def write_guest(guest, data, workdir, ffas):
    """
        Write a guest.def file. Guest molecule is assumed to be rigid.

        **Arguments:**

        guest
            A System instance

        data
            A tuple. If there is one entry, it should be the name of the
            molecule. It will be attempted to load critical parameters based
            on this name. Otherwise, there have to be four entries:
            (name, Tc, Pc, omega)

        workdir
            Directory where output file will be written

        ffas
            The atomtypes of the guest atoms
    """
    fn = os.path.join(workdir, "%s.def"%data[0])
    with open(fn,'w') as f:
        f.write("# critical constants: Temperature [T], Pressure [Pa],"
                " and Acentric factor [-]\n")
        if len(data)==4:
            # Critical parameters were supplied
            Tc, Pc, omega = data[1], data[2], data[3]
        else:
            # Try to load critical parameters from file
            try:
                eos = PREOS.from_name(data[0])
                Tc, Pc, omega = eos.Tc, eos.Pc, eos.omega
            # Failed to load critical parameters
            except ValueError:
                Tc, Pc, omega = None, None, None
        if Tc is None:
            f.write("Tc\nPc\nOmega\n")
        else:
            f.write("%20.8f\n%20.8f\n%20.8f\n" %
                (Tc/kelvin, Pc/pascal, omega))
        f.write("# total number Of atoms\n%d\n"%(guest.natom))
        f.write("# number of groups\n%d\n"%1)
        f.write("# group0\nrigid\n")
        f.write("# number of atoms\n%d\n"%(guest.natom))
        f.write("# atomic positions\n")
        for iatom in range(guest.natom):
            f.write("%d %12s %12.6f %12.6f %12.6f\n"%(iatom,ffas[iatom],
                guest.pos[iatom,0]/angstrom,guest.pos[iatom,1]/angstrom,
                guest.pos[iatom,2]/angstrom))
        f.write("# Chiral centers Bond  BondDipoles Bend  UrayBradley "
                "InvBend  Torsion Imp. Torsion Bond/Bond Stretch/Bend "
                "Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb\n")
        f.write("               0 %4d            0    0            0       0"
                "        0            0         0            0         0"
                "               0            0        0            0\n"  %
                 (guest.bonds.shape[0]))
        f.write("# Bond stretch: atom n1-n2, type, parameters\n")
        for ibond, (a,b) in enumerate(guest.bonds):
            f.write("%d %d RIGID_BOND\n"%(a,b))
        f.write("# Number of config moves\n0\n")


def write_pseudo_atoms(ff, workdir):
    """
        Write pseudo_atoms.def file
    """
    system = ff.system
    if system.masses is None: system.set_standard_masses()
    # If no charges are present, we write q=0
    if system.charges is None:
        charges = np.zeros(system.natom)
    else: charges = system.charges
    # RASPA cannot handle Gaussian charges
    if system.radii is not None:
        if log.do_warning:
            log.warn("Atomic radii were specified, but RASPA will not take "
                     "this into account for electrostatic interactions")
    # We need to write an entry for each atomtype, specifying the charge. If
    # atoms of the same atomtype show different charges, this means the
    # atomtypes need to be fine grained
    ffatypes, ffatype_ids = get_lammps_ffatypes(ff)
    # Write the file
    with open(os.path.join(workdir,'pseudo_atoms.def'),'w') as f:
        f.write("#number of pseudo atoms\n%d\n"%(len(ffatypes)))
        f.write("#type      print     as   chem  oxidation         mass       "
                "charge   polarization B-factor  radii  connectivity "
                "anisotropic anisotropic-type   tinker-type\n")
        for iffa, ffa in enumerate(ffatypes):
            iatom = np.where(ffatype_ids==iffa)[0][0]
            symbol = periodic[system.numbers[iatom]].symbol
            f.write("%-12s yes %6s %6s %10d %12.8f %+32.20f         %6.3f   "
                    "%6.3f %6.3f         %5d       %5d       %10s         %5d\n"
                 % (ffa, symbol, symbol, 0, system.masses[iatom]/amu,
                    charges[iatom],0.0,1.0,1.0,0,0,"absolute",0) )
    return ffatypes, ffatype_ids


def write_raspa_forcefield(ff, workdir):
    """
        Write the force_field_mixing_rules.def and force_field.def filse.
        Only Lennard-Jones and MM3 pair potentials are currently supported.

        **Arguments:**

        ff
            A ForceField instance

        workdir
            The directory where the output file will be placed
    """
    prefix = None
    # Loop over parts to find vdW pair potentials
    for part in ff.parts:
        if part.name.startswith('pair'):
            if part.name.startswith('pair_mm3'):
                eps, sig = part.pair_pot.epsilons/kcalmol, part.pair_pot.sigmas/angstrom
                prefix = "MM3_VDW"
            elif part.name.startswith('pair_lj'):
                eps, sig = part.pair_pot.epsilons/boltzmann/kelvin, part.pair_pot.sigmas/angstrom
                prefix = "LENNARD_JONES"
            elif part.name.startswith('pair_ei'):
                pass
            else:
                raise NotImplementedError("Could not convert pair potential"
                    " %s to RASPA" % (part.name))
    if prefix is None:
        raise ValueError("No pair potential found")
    # Group epsilon and sigma parameters per atom type
    epsilons, sigmas = [], []
    for iffa, ffa in enumerate(ff.system.ffatypes):
        mask = ff.system.ffatype_ids==iffa
        assert np.all(eps[mask]==eps[mask][0])
        assert np.all(sig[mask]==sig[mask][0])
        epsilons.append(eps[mask][0])
        sigmas.append(sig[mask][0])
    epsilons, sigmas = np.asarray(epsilons), np.asarray(sigmas)
    # Write the mixing rules file
    with open(os.path.join(workdir,'force_field_mixing_rules.def'),'w') as f:
        f.write("# general rule for shifted vs truncated\ntruncated\n")
        f.write("# general rule tailcorrections\nyes\n")
        f.write("# number of defined interactions\n%d\n"%(len(ff.system.ffatypes)))
        f.write("# type interaction\n")
        for iffa, ffa in enumerate(ff.system.ffatypes):
            f.write("%-20s %s %10.6f %10.6f\n" % (ffa+"_", prefix, epsilons[iffa], sigmas[iffa]))
        f.write("# general mixing rule\nLorentz-Berthelot\n")
    # Write the force_field.def file, which is empty in this case
    with open(os.path.join(workdir,'force_field.def'),'w') as f:
        f.write("# rules to overwrite\n0\n")
        f.write("# number of defined interactions\n0\n")
        f.write("# mixing rules to overwrite\n0\n")


def read_raspa_loading(fn):
    P, fugacity, T, N, Nerr = None, None, None, None, None
    with open(fn,'r') as f:
        for line in f:
            if "Partial pressure:" in line:
                P = float(line.split()[2])*pascal
            elif "Partial fugacity:" in line:
                fugacity = float(line.split()[2])*pascal
            elif "External temperature:" in line:
                T = float(line.split()[2])*kelvin
            elif "Average loading absolute" in line and "molecules" in line:
                N = float(line.split()[5])
                Nerr = float(line.split()[7])
                break
            elif "absolute adsorption" in line:
                w = line.split()
                if len(w)==14:
                    N = float(w[4][:-1])
                    Nerr = np.nan
    if P is None:
        raise ValueError("Failed to read `Partial pressure:` from file %s"%(fn))
    if fugacity is None:
        raise ValueError("Failed to read `Partial fugacity:` from file %s"%(fn))
    if T is None:
        raise ValueError("Failed to read `External temperature:` from file %s"%(fn))
    if N is None:
        raise ValueError("Failed to read `Average loading absolute` "
                         "and `absolute adsorption` from file %s"%(fn))
    return T, P, fugacity, N, Nerr
