#!/bin/bash

# Use libgomp that was compiled with the --disable-linux-futex configuration option
export LD_LIBRARY_PATH=./gcc-6.0.0-no-futex-lib/lib64:./gcc-6.0.0-no-futex-lib/lib:
# Store whether all tests pass or not
succes=true

# Include the Yaff extension header files
CFLAGS="-I./../yaff/pes"
# Collect object files generated during compilation
# TODO Do this in a more robust way
OBJ=""
for object in ../build/temp*/yaff/pes/*.o
do
    if [[ "$object" == *ext.o ]]; then continue; fi
    OBJ="$OBJ $object"
done

for name in test_ewald
do
    # Compile the test program
    gcc -fopenmp $CFLAGS -fPIC -o $name $name.c $OBJ -lm
    # Run helgrind
    valgrind --tool=helgrind --log-file=$name.log --suppressions=./gcc-6.0.0-no-futex-lib/valgrind.supp ./$name
    echo $name
    tail -n1 $name.log
    # Find out number of reported errors
    nerr=$(awk '/ERROR SUMMARY/{print $4F}' $name.log)
    if [ "$nerr" != "0" ]
    then
        succes=false
        cat $name.log
    fi
done

if [ "$succes" == true ]; then exit 0; else exit 1; fi
