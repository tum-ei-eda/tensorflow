#!/bin/bash

set -e

# IMPORTANT: can only compile/test a SINGLE test as recording of sequence of kernel
# variants selected for each op corresponds to the prepare sequence for single Invoke or
# series of tests.
# 
# Tests that Invoke more than one model or invoke kernels in a different order from prepare calls
# wil fail.

RECORD_KERNEL_VARIANTS=( TAGS=portable_optimized \
   LOCAL_CPPFLAGS="-DTF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT -DTF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES" \
)

USE_RECORDED_VARIANTS=( TAGS="portable_optimized generated" \
   LOCAL_CPPFLAGS="-DTF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS" \
)

echo "----------Testing conv----------"
make -s clean
make -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_kernel_conv_test
make -s clean
make -s "${USE_RECORDED_VARIANTS[@]}" -j8 test_kernel_conv_test

echo "----------Testing packed conv----------"
make -s clean
make -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_kernel_conv_packed_test
make -s clean
make -s "${USE_RECORDED_VARIANTS[@]}" -j8 test_kernel_conv_packed_test

echo "----------Testing depthwise conv----------"
make -s clean
make -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_kernel_depthwise_conv_test
make -s clean
make -s "${USE_RECORDED_VARIANTS[@]}" -j8 test_kernel_depthwise_conv_test

echo "----------Testing packed depthwise conv----------"
make -s clean
make -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_kernel_depthwise_conv_packed_test
make -s clean
make -s "${USE_RECORDED_VARIANTS[@]}" -j8 test_kernel_depthwise_conv_packed_test

echo "----------Testing fully connected----------"
make -s clean
make -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_kernel_fully_connected_test
make -s clean
make -s "${USE_RECORDED_VARIANTS[@]}" -j8 test_kernel_fully_connected_test

echo "----------Testing hello world----------"
make -s clean
make -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_hello_world_test
make -s clean
make -s "${USE_RECORDED_VARIANTS[@]}" -j8 test_hello_world_test

echo "----------Testing person_detection----------"
make -s clean
make -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_person_detection_test
make -s clean
make -s "${USE_RECORDED_VARIANTS[@]}" -j8 test_person_detection_test

echo "----------Testing micro_speech----------"
make -s clean
make -s "${RECORD_KERNEL_VARIANTS[@]}" -j8 test_micro_speech_test
make -s clean
make -s "${USE_RECORDED_VARIANTS[@]}" -j8 test_micro_speech_test



