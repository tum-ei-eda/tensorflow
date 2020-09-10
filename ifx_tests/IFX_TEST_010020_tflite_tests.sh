#!/bin/bash


cd `dirname "$BASH_SOURCE"`/..

TEST_OPTIONS=()
case `uname` in

  MINGW*)
    TEST_OPTIONS=( --test_tag_filters=-no_windows )
    echo tflite tests not runnable under windows currently
   ;;

  *)
     bazel test "${TEST_OPTIONS[@]}" //tensorflow/lite:all //tensorflow/lite/kernels/...:all //tensorflow/lite/core/...:all 
   ;;
esac
