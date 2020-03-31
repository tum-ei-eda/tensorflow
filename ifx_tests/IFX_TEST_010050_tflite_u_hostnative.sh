#!/bin/bash
set -e

cd `dirname "$BASH_SOURCE"`

# Native host build of hello_world, execute and check signals test passed
make -C .. -j 4  test_executables
make -C .. test
