#!/bin/bash
echo Cleaning code in \'`pwd`\' and subdirectories
for file in `find yaff *.c *.py | egrep "(\.py$)|(\.c$)|(\.pyx$)"`; do
  echo Cleaning ${file}
  sed -i -e $'s/\t/    /' ${file}
  sed -i -e $'s/[ \t]\+$//' ${file}
  sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' ${file}
done
