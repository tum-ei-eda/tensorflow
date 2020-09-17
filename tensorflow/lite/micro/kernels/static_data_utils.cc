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




#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/kernels/static_init_support.h"


#if TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT

namespace tflite {
namespace ops {
namespace micro {


CppPODStructInitializer TfLitePaddingValuesSubStruct(TfLitePaddingValues &pv) {

  auto init = new CppItems();
  *init 
    << pv.width
    << pv.height
    << pv.width_offset
    << pv.height_offset;

  CppPODStructInitializer res(init);
  return res;
}

CppNamedStruct TfLiteCustomSub8BitPackingDetailsStructPtr(const char *name, const TfLiteCustomSub8BitPackingDetails &pv) {

  auto init = new CppItems();
  *init 
    << pv.bits_per_item
    << pv.container_bits
    << pv.packed_minor_dims
    << "{}"; // Empty initializer 
  CppNamedStruct res(name, "const TfLiteCustomSub8BitPackingDetails", init);
  return res;
}


}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif // TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT
