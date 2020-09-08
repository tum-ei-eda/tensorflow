#!/bin/bash


cd `dirname "$BASH_SOURCE"`/..

TEST_OPTIONS=()
case `uname` in

  MINGW*)
    TEST_OPTIONS=( --test_tag_filters=-no_windows )
esac
	
# Native host build of micro_speech, execute and check signals test passed
bazel test "${TEST_OPTIONS[@]}" //tensorflow/lite:all //tensorflow/lite/kernels/...:all //tensorflow/lite/core/...:all 


