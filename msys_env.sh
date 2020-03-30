export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"
  if [ -z "$BAZEL_VC" ]; then
    #export BAZEL_VC=c:/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/BuildTools/VC
    export BAZEL_VC=c:/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Professional/VC
  fi
export TEST_TMPDIR=/c/TEMP/$USER
