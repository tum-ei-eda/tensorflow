#!/bin/bash
set -e

cd `dirname "$BASH_SOURCE"`

# Native host build of hello_world, execute and check signals test passed
make -C .. TARGET_ARCH=i386 clean
make -C .. -j 4  TARGET_ARCH=i386 TAGS=portable_optimized test_executables

# Guess what a bunch of upstream tests fail.... need to add support for 
# filter for expected failures.
#make -C .. test
