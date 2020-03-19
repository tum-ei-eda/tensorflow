#!/bin/bash
set -e
source ../SETTINGS_AND_VERSIONS.sh

while [[ "$1" != "" && "$1" != "--" ]]
do
    case "$1" in
    "--help"|"-?") 
        echo "`basename $0`: [--no_toco]"
        exit 1
        ;;
    "--no_toco")
        NOBUILD=1
        ;;
    "*")
        break
        ;;
    esac
    shift
done

# Build host native and IFX riscv32 builds of TF-lite(micro)
RISCV_SETTINGS=( TARGET_ARCH=${RV_TARGET_ARCH} TARGET_ABI=${RV_TARGET_ABI} )
# These builds only here to trigger error if build fails

TFLITE_MICRO_ROOT=${TOOLSPREFIX}/tflite_u-${TFLITE_MICRO_VERSION}
#bazel clean

# Build a statically linked toco command-line translator
# and TF-lite(micro) libraries

# Note need matching VC or VC build tools installation to build
# MinGW is NOT (visibly) supported.
# --subcommands logs compiler commands

(
  export PYTHON_BIN_PATH=$(which python)
  if [ -n "$MINGW64_HOST" ] 
  then
      PYTHON_BIN_PATH=$(cygpath --windows "${PYTHON_BIN_PATH}" )
      EXE_SUFFIX=.exe
  fi
#  export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
  export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'from distutils.sysconfig import get_python_lib;print(get_python_lib())')"
  export TF_NEED_GCP=0
  export TF_NEED_CUDA=0
  export TF_NEED_HDFS=0
  export TF_NEED_OPENCL=0
  export TF_ENABLE_XLA=0
  export TF_NEED_VERBS=0
  export TF_CUDA_CLANG=0
  export TF_NEED_ROCM=0
  export TF_NEED_MKL=0
  export TF_DOWNLOAD_MKL=0
  export TF_NEED_AWS=0
  export TF_NEED_MPI=0
  export TF_NEED_GDR=0
  export TF_NEED_S3=0
  export TF_NEED_OPENCL_SYCL=0
  export TF_SET_ANDROID_WORKSPACE=0
  export TF_NEED_COMPUTECPP=0
  export GCC_HOST_COMPILER_PATH=$(which gcc)
  export CC_OPT_FLAGS="-march=native"
  export TF_SET_ANDROID_WORKSPACE=0
  export TF_NEED_KAFKA=0
  export TF_NEED_TENSORRT=0 
  export TF_OVERRIDE_EIGEN_STRONG_INLINE=1
  export TF_DOWNLOAD_CLANG=0
  ./configure
)

if [ -z "$NOBUILD" ]
then
  bazel build --config=monolithic --subcommands=true //tensorflow/lite/toco:toco
  mkdir -p ${TFLITE_MICRO_ROOT}/bin  
  echo cp bazel-bin/tensorflow/lite/toco/toco${EXE_SUFFIX} ${TFLITE_MICRO_ROOT}/bin
  cp -f bazel-bin/tensorflow/lite/toco/toco${EXE_SUFFIX}  ${TFLITE_MICRO_ROOT}/bin
fi


make TARGET=ifx_riscv32_mcu ${RISCV_SETTINGS[@]} BUILD=DEBUG clean
make -j 4 TARGET=ifx_riscv32_mcu ${RISCV_SETTINGS[@]} BUILD=DEBUG microlite
make clean
make -j 4 microlite

# Actual payload - installed confiured copy of tflite(u) library and makefiles
make TARGET=ifx_riscv32_install_only ${RISCV_SETTINGS[@]} install
