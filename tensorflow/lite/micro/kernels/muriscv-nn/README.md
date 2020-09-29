# Info

To use RISCV-NN optimized kernels instead of reference kernel add TAGS=muriscv-nn
to the make line. Some micro architectures have optimizations (V-Extension and P-Extension),
others don't. The kernels that doesn't have optimization for a certain micro
architecture fallback to use TFLu reference kernels.


