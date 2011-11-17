#!/bin/bash
echo Cleaning code in \'`pwd`\' and subdirectories
for file in `find yaff input *.c *.py doc | egrep "(\.py$)|(\.c$)|(\.h$)|(\.pyx$)|(\.pxd$)|(\.rst$)|(\.txt$)"`; do
  echo Cleaning ${file}
  sed -i -e $'s/\t/    /' ${file}
  sed -i -e $'s/[ \t]\+$//' ${file}
  sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' ${file}
done
