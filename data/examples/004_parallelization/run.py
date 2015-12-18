#!/usr/bin/env python

import os, time, numpy as np, matplotlib.pyplot as plt

from tools import make_system

def test_scaling(name, verbose=True, make_plot=True):
    # Run the test for the following number of threads
    nthreads = [1,2,4]
    # Run the test for the following unit cell repetitions
    # Make sure that they are in order of increasing atoms to get a nice plot
    nrep = np.array([ [2,2,2],
                      [3,3,3],
                      [4,4,4],
                      [5,5,5],
                      [6,6,6],
                      [8,8,8],
                      [9,9,9],
                      [10,10,10], ], dtype='i4')
    # Repeat each calculation ncycles times, just to assure that overhead has a
    # small relative contribution to wall time
    ncycles = 10
    # Set up array that will contain all timings
    results = np.zeros((len(nthreads),nrep.shape[0]))
    # Start!
    for ithread, nthread in enumerate(nthreads):
        os.environ["OMP_NUM_THREADS"] = "%d"%nthread
        for irep in xrange(nrep.shape[0]):
            t0 = time.time()
            # Call the actual computation script
            os.system("./ewald.py %d %d %d %d" %
                    (nrep[irep,0],nrep[irep,0],nrep[irep,0],ncycles))
            # Store the wall clock time
            results[ithread,irep] = time.time() - t0
    # Number of atoms in one unit cell
    natom = make_system(1,1,1).natom
    # Total number of atoms for each of the selected number of cell repetitions
    natom = np.prod(nrep, axis=1)*natom
    if verbose:
        # Print some output
        print "="*80
        print name
        print "="*80
        print "%10s"%"#atoms",
        for nthread in nthreads: print "%10d" % nthread,
        print "\n",
        print "-"*80
        for irep in xrange(nrep.shape[0]):
            print "%10d" % natom[irep],
            for ithread in xrange(len(nthreads)):
                print "  %8.1f" % results[ithread,irep],
            print "\n",
    if make_plot:
        # Make a plot
        plt.clf()
        for ithread, nthread in enumerate(nthreads):
            plt.plot(natom,results[ithread]/ncycles,label="%2d threads"%nthread, marker='o')
        plt.xlabel("# atoms")
        plt.ylabel("Walltime/cycle [s]")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig("%s.png"%name)

if __name__=='__main__':
    test_scaling("ewald")
