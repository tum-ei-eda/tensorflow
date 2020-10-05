#!/bin/bash

set -e

cd `dirname "$BASH_SOURCE"`


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

USED_TARGET=( TARGET=ifx_riscv32_mcu)

TESTS=( \
   kernel_conv kernel_conv_packed kernel_depthwise_conv kernel_depthwise_conv_packed
   kernel_fully_connected kernel_pooling 
   hello_world person_detection micro_speech
)

for test in "${TESTS[@]}"
do
  echo "----------Testing $test ----------"
  make -C .. -s "${RECORD_KERNEL_VARIANTS[@]}" clean
  make -C .. -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_${test}_test
  make -C .. -s "${USE_RECORDED_VARIANTS[@]}" "${USED_TARGET[@]}" clean
  make -C .. -s "${USE_RECORDED_VARIANTS[@]}" "${USED_TARGET[@]}" -j8 test_${test}_test
done
