#!/bin/bash

set -e

# IMPORTANT: can only compile/test a SINGLE test as recording of sequence of kernel
# variants selected for each op corresponds to the prepare sequence for single Invoke or
# series of tests.
# 
# Tests that Invoke more than one model or invoke kernels in a different order from prepare calls
# wil fail.

RECORD_KERNEL_VARIANTS=( TAGS="portable_optimized record_model autodump"
)

USE_RECORDED_VARIANTS=( TAGS="portable_optimized recorded_model" \
)

TESTS=( \
   conv conv_packed depthwise_conv depthwise_conv_packed \ 
   fully_connected reduce pooling \
   hello_world  person_detection micro_speech
)

for test in "${TEST[@]}"
do
  echo "----------Testing $test ----------"
  make -s clean
  make -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_kernel_${test}_test
  make -s clean
  make -s "${USE_RECORDED_VARIANTS[@]}" -j8 test_kernel_${test}_test
done
