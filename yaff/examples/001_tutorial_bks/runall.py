#!/usr/bin/env python

import subprocess

subprocess.check_call(["python", "mksystem.py"], cwd='init')
subprocess.check_call(["python", "simulation.py"], cwd='opt')
subprocess.check_call(["python", "analysis.py"], cwd='opt')
subprocess.check_call(["python", "simulation.py", "300", "310"], cwd='nvt')
subprocess.check_call(["python", "analysis.py", "300", "310", "10"], cwd='nvt')
