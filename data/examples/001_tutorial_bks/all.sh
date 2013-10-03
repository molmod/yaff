#!/bin/bash
(cd init; ./mksystem.py)
(cd opt; ./simulation.py)
(cd opt; ./analysis.py)
(cd nvt; ./simulation.py 300 310)
(cd nvt; ./analysis.py 300 310 10)
