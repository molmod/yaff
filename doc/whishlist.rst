Whish list
##########

#. ``with`` interface for log and timer
#. All the TODO items of the Howto.
#. skin parameter in neighborlist
#. ACKS2
#. Restart functionality.
#. Nonbonding parameters without mixing rules, i.e. using tabulated cross
   parameters. This may be more accurate and will be much faster.
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
