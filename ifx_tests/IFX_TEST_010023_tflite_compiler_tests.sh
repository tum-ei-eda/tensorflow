#!/bin/bash


cd `dirname "$BASH_SOURCE"`/..


# Native host build of micro_speech, execute and check signals test passed
bazel test //tensorflow/compiler/mlir/...:all
