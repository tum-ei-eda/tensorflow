#!/bin/bash
set -e

cd `dirname "$BASH_SOURCE"`
make -C .. clean 
make -C .. TARGET=ifx_riscv32_mcu clean




