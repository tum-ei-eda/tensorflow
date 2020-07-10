#!/bin/bash
set -e

cd `dirname "$BASH_SOURCE"`

# Native host build of hello_world, execute and check signals test passed
make -C .. TARGET_ARCH=i386 clean
make -C .. -j 4  TARGET_ARCH=i386 test_executables
make -C .. test
