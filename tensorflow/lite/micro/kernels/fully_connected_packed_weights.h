/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_FULLY_CONNECTED_PACKED_WEIGHTS_H_
#define TENSORFLOW_LITE_MICRO_FULLY_CONNECTED_PACKED_WEIGHTS_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {
namespace ops {
namespace micro {



//
//  Uint8 Quantized fully connect kernel for < 8-bit packed weights
// "little-endian" format (first weight in LSB) ordering assumed.
//
// TODO Use specializations to handle fast case where dimensions
// allow efficient loop-unroll etc.
// accum_container_depth should really be  a params value
//

template <typename CONTAINER_T, size_t bits_per_item, size_t items_per_container>
void FullyConnectedUint8PackedWeights(
        const FullyConnectedParams& params,
        const RuntimeShape& input_shape, const uint8* input_data,
        const RuntimeShape& filter_shape, const CONTAINER_T* filter_data,
        const RuntimeShape& bias_shape, const int32* bias_data,
        const RuntimeShape& output_shape, uint8* output_data) {
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);


#if IFX_DEBUG_LOGGING
  std::cout << "Packed implementation!: filter_offset = " << std::dec << filter_offset << std::endl;
#endif
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const unsigned int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const unsigned int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  const unsigned int accum_container_depth = (accum_depth + (items_per_container-1u))/items_per_container;
  const int32 mask = (1<<bits_per_item)-1;
#if IFX_DEBUG_LOGGING
  std::cout << "Packed implementation!: accum-depth = " << std::dec << accum_depth << std::endl;
  bool once = false;
#endif

  unsigned int final_container_begin = accum_depth-(accum_depth%items_per_container);
  for (int b = 0; b < batches; ++b) {
    for (unsigned int out_c = 0; out_c < output_depth; ++out_c) {
      int32 acc = 0;
      const uint8_t *input_vals;
      CONTAINER_T filter_vals;
      unsigned int d = 0;
      unsigned int container = 0;
      for (;;) {
        input_vals = &input_data[b * accum_depth + d];
        filter_vals = filter_data[out_c * accum_container_depth + container];
        // Exit loop once last complete container processed...
        // Next container is setup
        if (d >= final_container_begin)
          break;
        // Unrollable loop!!
        for( unsigned int i = 0; i < items_per_container; ++i) {
            int32 input_val = input_vals[i] + input_offset;
            int32 filter_val = (filter_vals & mask) + filter_offset;
#if IFX_DEBUG_LOGGING
            if( !once ) {
                std::cout <<  std::dec << input_val << "*" << filter_val << ", ";
            }
#endif
            filter_vals >>= bits_per_item;
            acc += filter_val * input_val;
        }
        d += items_per_container;
        ++container;
      }
      // Remaining items if accum_depth%items_per_container !=0
      // TODO template params to handle no bias / weight container type
      // aligned cases.

      unsigned int i = 0;
      while( d < accum_depth ) {
          int32 input_val = input_vals[i] + input_offset;
          int32 filter_val = (filter_vals & mask) + filter_offset;
#if IFX_DEBUG_LOGGING
        if( !once ) {
          std::cout <<  std::dec << input_val << "*" << filter_val << ", ";
        }
#endif
          filter_vals >>= bits_per_item;
          acc += filter_val * input_val;
          ++d;
          ++i;
      }

      if (bias_data) {
        acc += bias_data[out_c];
      }
#if IFX_DEBUG_LOGGING
      if( !once ) {
        std::cout << "RAW ACC " << acc << std::endl;
      }
      once = true;
#endif
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<uint8>(acc);
    }

  }
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite


#endif  // TENSORFLOW_LITE_MICRO_FULLY_CONNECTED_PACKED_WEIGHTS_H_
