#!/usr/bin/env python

import subprocess

subprocess.check_output(["python", "mksystem.py"], cwd='init')
subprocess.check_output(["python", "simulation.py"], cwd='opt')
subprocess.check_output(["python", "analysis.py"], cwd='opt')
subprocess.check_output(["python", "simulation.py", "300", "310"], cwd='nvt')
subprocess.check_output(["python", "analysis.py", "300", "310", "10"], cwd='nvt')
