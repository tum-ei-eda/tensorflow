# Info

To use CMSIS-NN for RISCV optimized kernels instead of reference kernel add TAGS=cmsis-nn-riscv
to the make line. Some micro architectures have optimizations (V-Extension),
others don't. The kernels that doesn't have optimization for a certain micro
architecture fallback to use TFLu reference kernels.

The optimizations are almost exclusively made for int8 (symmetric) model. For
more details, please read
[CMSIS-NN doc](https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/README.md)

# Example 1

A simple way to compile a binary with CMSIS-NN optimizations.

```
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn \
TARGET=sparkfun_edge person_detection_int8_bin
```

