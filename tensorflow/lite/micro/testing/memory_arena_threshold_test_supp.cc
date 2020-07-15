/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/memory_arena_threshold_test_refvalues.h"

// NOTE: These values are measured on x86-64 with reference operator implementations
// TODO(b/158651472): Consider auditing these values on non-64 bit systems.

const int kKeywordModelTensorCount = 54;
const int kKeywordModelHeadSize = 1008;
const int kKeywordModelTailSize = 13984 + sizeof(TfLiteTensor)*kKeywordModelTensorCount;
const int kKeywordModellAdditionalOpTailAllocations = 1024;

const int kTestConvModelTensorCount = 15;
const int kTestConvModelHeadSize = 7744;
const int kTestConvModelTailSize = 2256 + sizeof(TfLiteTensor)*kTestConvModelTensorCount;
const int kTestConvModelAdditionalOpTailAllocations = 1528;