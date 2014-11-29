#!/bin/bash
for i in $(find yaff data/examples | egrep "\.pyc$|\.py~$|\.pyc~$|\.bak$|\.so$") ; do rm -v ${i}; done

for d in data/examples/*; do (cd $d; echo Claning in $d; ./clean.sh); done

rm -vr doctrees
rm -v yaff/pes/ext.c
(cd doc; make clean)
rm -v MANIFEST
rm -vr dist
rm -vr build
rm -v trajectory_*.xyz
