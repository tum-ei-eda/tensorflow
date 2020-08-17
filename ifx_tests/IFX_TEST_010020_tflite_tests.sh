#!/bin/bash


cd `dirname "$BASH_SOURCE"`/..


# Native host build of micro_speech, execute and check signals test passed
bazel test //tensorflow/lite:all //tensorflow/lite/kernels/...:all //tensorflow/lite/core/...:all 

RES=$?
# Under Windows tflite test appear broken and unmaintained (fail to build)
case `uname` in

  MINGW*)
     if [[ ! $RES = 0 ]]
     then
        exit 122
     else
        exit 123
     fi
esac

exit $RES
