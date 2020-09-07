#!/bin/bash

cd `dirname "$BASH_SOURCE"`/..

echo "----------Testing conv----------"
make clean > /dev/null 2>/dev/null
make TAGS=portable_optimized LOCAL_CPPFLAGS=-DPRECOMPILE -j8 test_kernel_conv_test > /dev/null 2>/dev/null
make clean > /dev/null 2>/dev/null
make -s TAGS=portable_optimized LOCAL_CPPFLAGS=-DEVAL -j8 test_kernel_conv_test

echo "----------Testing packed conv----------"
make clean > /dev/null 2>/dev/null
make TAGS=portable_optimized LOCAL_CPPFLAGS=-DPRECOMPILE -j8 test_kernel_conv_packed_test > /dev/null 2>/dev/null
make clean > /dev/null 2>/dev/null
make -s TAGS=portable_optimized LOCAL_CPPFLAGS=-DEVAL -j8 test_kernel_conv_packed_test

echo "----------Testing depthwise conv----------"
make clean > /dev/null 2>/dev/null
make TAGS=portable_optimized LOCAL_CPPFLAGS=-DPRECOMPILE -j8 test_kernel_depthwise_conv_test > /dev/null 2>/dev/null
make clean > /dev/null 2>/dev/null
make -s TAGS=portable_optimized LOCAL_CPPFLAGS=-DEVAL -j8 test_kernel_depthwise_conv_test

echo "----------Testing packed depthwise conv----------"
make clean > /dev/null 2>/dev/null
make TAGS=portable_optimized LOCAL_CPPFLAGS=-DPRECOMPILE -j8 test_kernel_depthwise_conv_packed_test > /dev/null 2>/dev/null
make clean > /dev/null 2>/dev/null
make -s TAGS=portable_optimized LOCAL_CPPFLAGS=-DEVAL -j8 test_kernel_depthwise_conv_packed_test

echo "----------Testing fully connected----------"
make clean > /dev/null 2>/dev/null
make TAGS=portable_optimized LOCAL_CPPFLAGS=-DPRECOMPILE -j8 test_kernel_fully_connected_test > /dev/null 2>/dev/null
make clean > /dev/null 2>/dev/null
make -s TAGS=portable_optimized LOCAL_CPPFLAGS=-DEVAL -j8 test_kernel_fully_connected_test

echo "----------Testing hello world----------"
make clean > /dev/null 2>/dev/null
make TAGS=portable_optimized LOCAL_CPPFLAGS=-DPRECOMPILE -j8 test_hello_world_test > /dev/null 2>/dev/null
make clean > /dev/null 2>/dev/null
make -s TAGS=portable_optimized LOCAL_CPPFLAGS=-DEVAL -j8 test_hello_world_test

echo "----------Testing person_detection----------"
make clean > /dev/null 2>/dev/null
make TAGS=portable_optimized LOCAL_CPPFLAGS=-DPRECOMPILE -j8 test_person_detection_test > /dev/null 2>/dev/null
make clean > /dev/null 2>/dev/null
make -s TAGS=portable_optimized LOCAL_CPPFLAGS=-DEVAL -j8 test_person_detection_test

echo "----------Testing micro_speech----------"
make clean > /dev/null 2>/dev/null
make TAGS=portable_optimized LOCAL_CPPFLAGS=-DPRECOMPILE -j8 test_micro_speech_test > /dev/null 2>/dev/null
make clean > /dev/null 2>/dev/null
make -s TAGS=portable_optimized LOCAL_CPPFLAGS=-DEVAL -j8 test_micro_speech_test



