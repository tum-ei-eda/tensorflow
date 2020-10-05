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


#ifndef TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POOLING_OP_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POOLING_OP_DATA_H_

// PORTABLE OPTIMIZED

// TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT: 
//  When set the names of kernel variants eval functions recorded and can be dumped
// via PointerCollect API.
// TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
//   When set prepare phase kernel variant selection code is dropped with 
// the eval functions recorded in tflite::micro::kernels::pooling::eval_functions used instead.
//
// Benefits smaller binary, used unnecessary eval function variants are not lnked.





#include "tensorflow/lite/c/common.h"


namespace tflite {
namespace ops {
namespace micro {
namespace pooling {

struct OpData;

#define EVAL_FUNC_DECL(name) \
  void name( \
      const TfLiteContext* context, const TfLiteNode* node, \
      const TfLitePoolParams* params, const OpData* data, \
      const TfLiteEvalTensor* input, TfLiteEvalTensor* output)


typedef EVAL_FUNC_DECL((*EvalVariantFptr));

EVAL_FUNC_DECL(AverageEvalFloat);
EVAL_FUNC_DECL(AverageEvalQuantizedInt8);
EVAL_FUNC_DECL(AverageEvalQuantizedUInt8);
EVAL_FUNC_DECL(MaxEvalFloat);
EVAL_FUNC_DECL(MaxEvalQuantizedInt8);
EVAL_FUNC_DECL(MaxEvalQuantizedUInt8);

#undef EVAL_FUNC_DECL


struct OpData {
  TfLitePaddingValues padding;
  int32_t activation_min;
  int32_t activation_max;
  float activation_min_f32;
  float activation_max_f32;
  EvalVariantFptr eval_function;
};



}  // namespace pooling
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POOLING_OP_DATA_H_
