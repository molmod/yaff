#!/usr/bin/env python

import numpy as np

from molmod.units import *
from yaff import log
log.set_level(log.silent)
from molmod.periodic import periodic

from common import get_ff, get_system

if __name__=='__main__':
    nmol = 97
    # Initial density
    rho0 = 420*kilogram/meter**3
    system = get_system(nmol, rho0=rho0)
    # Loop over cutoff values
    rcuts = np.linspace(10.0,50.0,21)*angstrom
    vtens = np.zeros((3,3))
    print("%15s | %15s %15s | %15s %15s"%("rcut [AA]","E [kcalmol]","P [MPa]",
            "E+E_tail [kcalmol]","P+P_tail [MPa]"))
    print("="*80)
    for rcut in rcuts:
        print("%15.6f"%(rcut/angstrom),end='')
        row = []
        for tailcorr in [False, True]:
            log.set_level(log.silent)
            ff = get_ff(system, rcut, tailcorr=tailcorr)
            log.set_level(log.silent)
            vtens[:] = 0.0
            e = ff.compute(vtens=vtens)
            p = np.trace(vtens)/ff.system.cell.volume/3.0
            row.append(e)
            row.append(p)
            print(" | %15.6f %15.6f" % (e/kcalmol,p/1e6/pascal),end='')
        print("\n",end='')

