#!/bin/bash


cd `dirname "$BASH_SOURCE"`


# Native host build of micro_speech, execute and check signals test passed
make -C .. TARGET=ifx_riscv32_mcu clean
make  -C .. -j 4 TARGET=ifx_riscv32_mcu TAGS=portable_optimized test_executables
make  -C .. TARGET=ifx_riscv32_mcu TAGS=portable_optimized test

