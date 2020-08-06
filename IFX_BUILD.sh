#!/bin/bash
set -e
source ../SETTINGS_AND_VERSIONS.sh


# Monolothic needed to produce isntallable tf_tfl_translate	
BAZEL_CXX_BUILD_SETTINGS=(
		--config=opt
                --config=monolithic
)
BAZEL_REPO_OVERRIDES=(  )
TARGET_ARCH=native
LOCALJOBS=HOST_CPUS*0.7
while [[ "$1" != "" && "$1" != "--" ]]
do
    case "$1" in
    "--help"|"-?") 
        echo "`basename $0`: [--no_toco] [--no-config] [--no-build] [--no-tflite] " 1>&2
	echo "  [--gast|--debug] [--jobs EXPR] [--remote N] [--override-llvm]" 1>&2
	echo "  [--verbose]" 1>&2
        exit 1
        ;;
    "--no-config")
        NOCONFIG=1
        ;;
    "--no-build")
        NOBUILD=1
        ;;
    "--no-tflite")
        NOTFLITE=1
        ;;
    "--no-download-update")
        NOINSTALL=1
        NODOWNLOAD_UPDATE=1
        ;;
    "--no-install")
        NOINSTALL=1
        ;;
    "--with-pip")
	WITH_PIP=1
	;;
    "--opt")
        BAZEL_CXX_BUILD_SETTINGS=(
	  --config opt 
        )
        ;;

    "--fast")
	BAZEL_CXX_BUILD_SETTINGS=(
                  --copt=-O1 --cxxopt=-O1 --strip=never
                  --config=monolithic
	)
	;;
    "--debug")
	# includes workaround for mis-documented and buggy 
	# per_object_debug_info feature.
	# We use O1 because gcc has some bugs relating to constexpr
	# defintions if optimization is off completely
	BAZEL_CXX_BUILD_SETTINGS=(
                --config=monolithic --config=dbg
                --features=per_object_debug_info
                --define='per_object_debug_info_file=yes'
                --copt=-O1 --cxxopt=-O1 
                --strip=never --fission=yes 
	)
	;;
    "--jobs")
        LOCALJOBS="$2"
        shift
        ;;
    "--remote")
        BAZEL_REMOTE_OPTIONS=(
           --jobs="$2" --spawn_strategy=local,remote --strategy=CppCompile=remote --remote_executor=grpc://pistol:8980
        )
	      TARGET_ARCH=sandybridge
        ;;
    "--override-llvm")
       BAZEL_REPO_OVERRIDES=( --override_repository=llvm-project=$(readlink -f "../llvm-project")  )
    ;;
    "--verbose")
        VERBOSE=( --subcommands=true )
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


if [ -n "$RD_CLUSTER_LINUX_HOST" ]
then
  BAZEL_DISTDIR_OPTIONS=( --distdir /home/aifordes.work/share/bazel-distdir )
fi

BAZEL_CMDLINE_OPTIONS=( )

if [ -n "$MINGW64_HOST" ] 
then
  BAZEL_OPTIONS=(
      "${BAZEL_DISTDIR_OPTIONS[@]}"
      "${BAZEL_REPO_OVERRIDES[@]}"
      --local_cpu_resources="$LOCALJOBS"  "${VERBOSE[@]}"
      "${BAZEL_REMOTE_OPTIONS[@]}"
      "${BAZEL_CXX_BUILD_SETTINGS[@]}"
      #--cxxopt=-DTF_LITE_DISABLE_X86_NEON --copt=-DTF_LITE_DISABLE_X86_NEON
      --verbose_failures=yes
  )
elif [ -n "$RD_CLUSTER_LINUX_HOST" ]
then
  BAZEL_OPTIONS=(
      "${BAZEL_DISTDIR_OPTIONS[@]}"
      "${BAZEL_REPO_OVERRIDES[@]}"
      --local_cpu_resources="$LOCALJOBS" "${VERBOSE[@]}"
      "${BAZEL_REMOTE_OPTIONS[@]}"
      "${BAZEL_CXX_BUILD_SETTINGS[@]}"
     --cxxopt=-DTF_LITE_DISABLE_X86_NEON --copt=-DTF_LITE_DISABLE_X86_NEON
	    "--repository_cache=/home/aifordes.work/bazel_repo_cache" 
	    # Note: Cannot enable debug non-NFS scratch for bazel_root too small...
	    --verbose_failures=yes
  )
else
  BAZEL_OPTIONS=(
        "${BAZEL_DISTDIR_OPTIONS[@]}"
      "${BAZEL_REPO_OVERRIDES[@]}"
      --local_cpu_resources="$LOCALJOBS"  "${VERBOSE[@]}"
      "${BAZEL_REMOTE_OPTIONS[@]}"
      "${BAZEL_CXX_BUILD_SETTINGS[@]}"
      --cxxopt=-DTF_LITE_DISABLE_X86_NEON --copt=-DTF_LITE_DISABLE_X86_NEON
      --verbose_failures=yes
  )
fi
# Build a statically linked toco command-line translator
# and TF-lite(micro) libraries

# Note need matching VC or VC build tools installation to build
# MinGW is NOT (visibly) supported.
# --subcommands logs compiler commands
if [ -z "$NOCONFIG" ]
then
(
  export PYTHON_BIN_PATH=$(which python)
  if [ -n "$MINGW64_HOST" ] 
  then
      PYTHON_BIN_PATH=$(cygpath --windows "${PYTHON_BIN_PATH}" )
      EXE_SUFFIX=.exe
      export CC_OPT_FLAGS="/arch:AVX"
      # Wild patching orgy is needed... including yes in 2020
      # workarounds for pathname/command-line length limitations.
     source msys_env.sh
  else
    export TF_NEED_GCP=0
    export TF_NEED_HDFS=0
    export TF_NEED_OPENCL=0
    export TF_ENABLE_XLA=0
    export TF_NEED_VERBS=0
    export TF_CUDA_CLANG=0
    export TF_NEED_MKL=0
    export TF_DOWNLOAD_MKL=0
    export TF_NEED_AWS=0
    export TF_NEED_MPI=0
    export TF_NEED_GDR=0
    export TF_NEED_S3=0
    export TF_NEED_OPENCL_SYCL=0
    export TF_NEED_COMPUTECPP=0
    export GCC_HOST_COMPILER_PATH=$(which gcc)
    export CC_OPT_FLAGS="-march=${TARGET_ARCH} -Wno-sign-compare"
    export TF_SET_ANDROID_WORKSPACE=0
    export TF_NEED_KAFKA=0
    export TF_NEED_TENSORRT=0 
    export TF_DOWNLOAD_CLANG=0
  fi
#  export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"

  export TF_SET_ANDROID_WORKSPACE=0
  export TF_NEED_ROCM=0
  export TF_NEED_CUDA=0
  export TF_OVERRIDE_EIGEN_STRONG_INLINE=1
  export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'from distutils.sysconfig import get_python_lib;print(get_python_lib())')"
  ./configure

  rm -f .bazelrc.user
  echo configured bazel build: "${BAZEL_OPTIONS[@]}"
  for o in "${BAZEL_OPTIONS[@]}"
  do
    echo build "$o" >> .bazelrc.user
  done
)
else
  echo Skipping configure...
fi

BAZEL_TARGETS=( //tensorflow/compiler/mlir/lite:tf_tfl_translate )

if [ -n "$WITH_PIP" ]
then
    BAZEL_TARGETS=( "${BAZEL_TARGETS[@]}" //tensorflow/tools/pip_package:build_pip_package )
fi

if [ -z "$NOBUILD" ]
then
    # Some useful recipes from the past... comment out in case needed
    #  bazel fetch "${BAZEL_DISTDIR_OPTIONS[@]}" //tensorflow/compiler/mlir/lite:tf_tfl_translate
    #bazel build "${BAZEL_OPTIONS[@]}" //third_party/aws:aws || true  # EXPECTED FAILURE but needed to unpack packages
    #bazel build --local_cpu_resources="$LOCALJOBS" --config=dbg --strip=never "${VERBOSE[@]}" //tensorflow/compiler/mlir/lite:tf_tfl_translate
    #sed -e'1,$s/"+[cd]/"+g/g' -i bazel-$(basename $(pwd))/external/aws-checksums/source/intel/crc32c_sse42_asm.c  #  BUILD FAILS MISERABG:Y WITH GCC7 FFS
    # Wild patching orgy is needed... including yes in 2020
    # workarounds for pathname/command-line length limitations.
    if [ -n "$MINGW64_HOST" ] 
    then
       source msys_env.sh
    fi
    # 
    echo bazel build "${BAZEL_CMDLINE_OPTIONS[@]}"  "${BAZEL_TARGETS[@]}"
    bazel build   "${BAZEL_CMDLINE_OPTIONS[@]}"  "${BAZEL_TARGETS[@]}"
    if [ -z "$NOINSTALL" ]
    then
      mkdir -p ${TFLITE_MICRO_ROOT}/bin  
      rm -f ${TFLITE_MICRO_ROOT}/bin/*
      echo Installing to ${TFLITE_MICRO_ROOT}/bin
      cp bazel-bin/tensorflow/compiler/mlir/lite/tf_tfl_translate${EXE_SUFFIX} \
        ${TFLITE_MICRO_ROOT}/bin
    fi
fi

if [ -n "$WITH_PIP" ]
then
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pk
    mv /tmp/tensorflow_pk/tf*.whl .
    rm -rf /tmp/tensorflow_pk
fi

if [ -z "$NOTFLITE" ] 
then
    if [ -z "$NODOWNLOAD_UPDATE"]
    then
	    make mrproper clean_downloads
    else
        make mrproper
    fi
    make BUILD_TYPE=debug third_party_downloads
    echo make -j 4 TARGET=ifx_riscv32_mcu ${RISCV_SETTINGS[@]} BUILD_TYPE=debug microlite
    make -j 4 TARGET=ifx_riscv32_mcu ${RISCV_SETTINGS[@]} BUILD_TYPE=debug microlite
    make -j 4 TARGET=ifx_riscv32_mcu ${RISCV_SETTINGS[@]} microlite
    echo make -j 4 BUILD_TYPE=debug test_executables
    make -j 4 BUILD_TYPE=debug test_executables


    if [ -z "$NOINSTALL" ]
    then
        # Actual payload - installed confiured copy of tflite(u) library and makefiles
        make TARGET=ifx_riscv32_install_only ${RISCV_SETTINGS[@]} install
        # Set TAGS to use the portable_optimized kernels (by default) instead of the reference ones.
        echo 'TAGS ?= portable_optimized' >> ${TFLITE_MICRO_ROOT}/tools/make/installed_settings.inc
    fi
    
    # Clean up afterwards because bugs in downlaods from tflite(u)
    # poison VS-code bazel target discovery
    make mrproper
fi
