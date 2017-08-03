#!/usr/bin/env bash
chmod +x init/mksystem.py opt/simulation.py opt/analysis.py nvt/simulation.py nvt/analysis.py
(cd init; ./mksystem.py)
(cd opt; ./simulation.py)
(cd opt; ./analysis.py)
(cd nvt; ./simulation.py 300 310)
(cd nvt; ./analysis.py 300 310 10)
