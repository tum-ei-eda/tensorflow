#!/bin/bash


cd `dirname "$BASH_SOURCE"`/..


# Native host build of micro_speech, execute and check signals test passed
bazel test //tensorflow/lite:all //tensorflow/lite/kernels/...:all //tensorflow/lite/core/...:all
