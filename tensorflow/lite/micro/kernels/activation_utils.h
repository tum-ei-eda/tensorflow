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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_ACTIVATION_UTILS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_ACTIVATION_UTILS_H_

#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"

#if defined(TF_LITE_USE_GLOBAL_FMINMAX)
#define TF_LITE_FMIN ::fmin
#define TF_LITE_FMAX ::fmax
#else
#define TF_LITE_FMIN std::fmin
#define TF_LITE_FMAX std::fmax
#endif

namespace tflite {
namespace ops {
namespace micro {

// Returns the floating point value for a fused activation:
inline float ActivationValFloat(TfLiteFusedActivation act, float a) {
  switch (act) {
    case kTfLiteActNone:
      return a;
    case kTfLiteActRelu:
      return TF_LITE_FMAX(0.0f, a);
    case kTfLiteActRelu1:
      return TF_LITE_FMAX(-1.0f, TF_LITE_FMIN(a, 1.0f));
    case kTfLiteActRelu6:
      return TF_LITE_FMAX(0.0f, TF_LITE_FMIN(a, 6.0f));
    case kTfLiteActTanh:
      return std::tanh(a);
    case kTfLiteActSignBit:
      return std::signbit(a);
    case kTfLiteActSigmoid:
      return 1.0f / (1.0f + std::exp(-a));
  }
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#undef TF_LITE_FMIN
#undef TF_LITE_FMAX

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ACTIVATION_UTILS_H_
