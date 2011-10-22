Whish list
##########

#. All the TODO items of the Howto.
#. A Cook
#. Proper trunction of electrostatic interactions.
#. ACKS2
#. Restart functionality.
#. Constraints and restraints.
#. An optional select argument for the iterative algorithms.
#. MPI
#. Cell lists.
#. NlogN scaling of the electrostatics: develop a real-space-grid Poisson solver
   based on a Hermite-Polynomial basis set. (Iteratively finite element method)
   Do not use point charges to computed the interactions with the long-range
   potential, but rather smeared charges. (The residual short-range part can
   be moved to the real-space sum). Gaussian x Poly can be intergrated
   analytically.
#. Corretly treat the periodic boundary conditions in very skewed cells.
   The current implementation of the minimum image convention is, just like in
   most MD codes, rather naive.
#. Reference system, so users know what to cite.
