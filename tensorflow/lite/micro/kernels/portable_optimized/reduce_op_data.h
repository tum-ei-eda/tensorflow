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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_REDUCE_OP_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_REDUCE_OP_DATA_H_

// PORTABLE OPTIMIZED

// TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT: 
//  When set the names of kernel variants eval functions recorded and can be dumped
// via PointerCollect API.
// TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
//   When set prepare phase kernel variant selection code is dropped with 
// the eval functions recorded in tflite::micro::kernels::reduce::eval_functions used instead.
//
// Benefits smaller binary, used unnecessary eval function variants are not lnked.


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {
namespace ops {
namespace micro {
namespace reduce {

struct OpData;

typedef TfLiteStatus (*EvalVariantFptr)(
    TfLiteContext* context, OpData* op_data,
    TfLiteReducerParams* params, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* axis, TfLiteEvalTensor* output);

#define EVAL_FUNC(name) \
  TfLiteStatus name( \
     TfLiteContext* context, OpData* op_data, \
     TfLiteReducerParams* params, const TfLiteEvalTensor* input, \
     const TfLiteEvalTensor* axis, TfLiteEvalTensor* output);

EVAL_FUNC(ReduceFloatKeepDims);
EVAL_FUNC(ReduceFloatChangeDims);
EVAL_FUNC(ReduceInt8KeepDims);
EVAL_FUNC(ReduceInt8ChangeDims);
EVAL_FUNC(ReduceInt8ChangeDimsAndQuant);
EVAL_FUNC(ReduceUInt8KeepDims);
EVAL_FUNC(ReduceUInt8ChangeDims);
EVAL_FUNC(ReduceUInt8ChangeDimsAndQuant);

#undef EVAL_FUNC

struct OpData {
  int32_t multiplier;
  int shift;
  int temp_buffer_idx;
  int32_t* temp_buffer;
  int input_zp;
  float input_scale;
  int output_zp;
  float output_scale;
  EvalVariantFptr eval_function;
};

}  // namespace reduce
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_REDUCE_OP_DATA_H_
