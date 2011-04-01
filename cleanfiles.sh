#!/bin/bash
for i in $(find yaff | egrep "\.pyc$|\.py~$|\.pyc~$|\.bak$|\.so$") ; do rm -v ${i}; done

rm -v yaff/ext.c
rm -v MANIFEST
rm -vr dist
rm -vr build
