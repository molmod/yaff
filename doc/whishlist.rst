Whish list
##########

1. All the TODO items of the Howto.
2. A timer object.
3. ACKS2
4. Restart functionality.
5. Nonbonding parameters without mixing rules, i.e. using tabulated cross parameters
6. MPI
7. Cell lists.
8. NlogN scaling of the electrostatics: develop a real-space-grid Poisson solver
   based on a Hermite-Polynomial basis set. (Iteratively finite element method)
   Do not use point charges to computed the interactions with the long-range
   potential, but rather smeared charges. (The residual short-range part can
   be moved to the real-space sum). Gaussian x Poly can be intergrated
   analytically.
9. Corretly treat the periodic boundary conditions in very skewed cells.
   The current implementation of the minimum image convention is, just like in
   most MD codes, rather naive.
