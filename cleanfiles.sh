#!/bin/bash
for i in $(find yaff examples | egrep "\.pyc$|\.py~$|\.pyc~$|\.bak$|\.so$") ; do rm -v ${i}; done

for d in examples/*; do (cd $d; ./clean.sh); done

rm -vr doctrees
rm -v yaff/pes/ext.c
rm -rv doc/_build
rm -v MANIFEST
rm -vr dist
rm -vr build
