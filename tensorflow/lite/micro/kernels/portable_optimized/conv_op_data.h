/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// PORTABLE OPTIMIZED

// Support recording of selected kernel variant in prepare phase for static extraction for
// a fixed tflite model.

// TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT: 
//  When set the names of kernel variants eval functions recorded and can be dumped
// via PointerCollect API.
// TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
//   When set prepare phase kernel variant selection code is dropped with 
// the eval functions recorded in tflite::micro::kernels::conv::eval_functions used instead.
//
// Benefits smaller binary, used unnecessary eval function variants are not lnked.


#ifndef TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_CONV_OP_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_CONV_OP_DATA_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace conv {


struct OpData;

#define EVAL_FUNC_DECL(name) \
  TfLiteStatus name( \
      TfLiteConvParams* params, OpData* data, \
      const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter, \
      const TfLiteEvalTensor* bias, TfLiteEvalTensor* output, TfLiteContext* context)

typedef EVAL_FUNC_DECL((*EvalVariantFptr));

EVAL_FUNC_DECL(EvalConvFloat);
EVAL_FUNC_DECL(EvalConvInt8Reference);
EVAL_FUNC_DECL(EvalConvInt8Padding);
EVAL_FUNC_DECL(EvalConvInt8);
EVAL_FUNC_DECL(EvalConvUInt8Packed);
EVAL_FUNC_DECL(EvalConvUInt8Padding);
EVAL_FUNC_DECL(EvalConvUInt8Reference);
EVAL_FUNC_DECL(EvalConvUInt8);
template <typename CONTAINER_T, size_t bits_per_item,
            size_t items_per_container>
class PackedConv {
 public:
  static EVAL_FUNC_DECL(EvalUint8PackedWeights);
};

#undef EVAL_FUNC_DECL


struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.

  // Cached tensor zero point values for quantized operations.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;

  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift (allocated dynamically).
  int32_t *per_channel_output_multiplier;
  int32_t *per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  // The precomputed sum of filters factor
  int32_t *sum_of_filters_factor;

    // Weights tensor packing information
  const TfLiteCustomSub8BitPackingDetails *custom_sub8bit_packing;

  // Eval function pointer
  EvalVariantFptr eval_function;
};



}  // namespace conv
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_CONV_OP_DATA_H_
