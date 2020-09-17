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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_FULLY_CONNECTED_OP_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_FULLY_CONNECTED_OP_DATA_H_

// PORTABLE OPTIMIZED

// TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT: 
//  When set the names of kernel variants eval functions recorded and can be dumped
// via PointerCollect API.
// TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
//   When set prepare phase kernel variant selection code is dropped with 
// the eval functions recorded in tflite::micro::kernels::conv::eval_functions used instead.
//
// Benefits smaller binary, used unnecessary eval function variants are not lnked.


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {
namespace ops {
namespace micro {
namespace fully_connected {

struct OpData;

 typedef TfLiteStatus (*EvalVariantFptr)(TfLiteContext* context, TfLiteFullyConnectedParams* params,
      OpData* opData, const TfLiteTensor* input, const TfLiteTensor* weights,
      const TfLiteTensor* bias, TfLiteTensor* output);

#define EVAL_FUNC(name) \
  TfLiteStatus name( \
    TfLiteContext* context, TfLiteFullyConnectedParams* params, OpData* opData, \
    const TfLiteTensor* input, const TfLiteTensor* weights, \
    const TfLiteTensor* bias, TfLiteTensor* output)

template <typename T>
EVAL_FUNC(EvalQuantized);
EVAL_FUNC(EvalQuantizedInt8);
EVAL_FUNC(EvalQuantizedUint8WithOutputInt16);
EVAL_FUNC(EvalQuantizedPacked);
EVAL_FUNC(EvalQuantizedUInt8);
EVAL_FUNC(EvalFloat);

#undef EVAL_FUNC

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
  // A buffer containing the sum-of-weights factor
  int32_t* sum_of_weights_factor;
  // Eval function pointer
  EvalVariantFptr eval_function;
};

}  // namespace fully_connected
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_FULLY_CONNECTED_OP_DATA_H_
