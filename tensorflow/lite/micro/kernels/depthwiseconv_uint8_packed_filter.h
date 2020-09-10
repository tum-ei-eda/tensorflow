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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_DEPTHWISECONV_UINT8_PACKED_FILTER_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_DEPTHWISECONV_UINT8_PACKED_FILTER_H_

#include <algorithm>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace ops {
namespace micro {

  
namespace depthwise_conv {

template <typename CONTAINER_T, size_t bits_per_item, size_t items_per_container>
struct DepthwiseConvPackedFilter {

  static inline void Run(const DepthwiseParams& params,
                         const RuntimeShape& input_shape,
                         const uint8_t* input_data,
                         const RuntimeShape& filter_shape,
                         const CONTAINER_T* filter_data,
                         const RuntimeShape& bias_shape, const int32_t* bias_data,
                         const RuntimeShape& output_shape, uint8_t* output_data, int32_t *accbuf) {
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    const int32_t input_offset = params.input_offset;
    const int32_t filter_offset = params.weights_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_multiplier = params.output_multiplier;
    const int output_shift = params.output_shift;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    const int num_packed_containers = std::ceil((float)output_depth / items_per_container);
    const int32_t mask = (1<<bits_per_item)-1;

    for (int b = 0; b < batches; ++b) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;

          for (int i = 0; i < output_depth; ++i) {
            accbuf[i] = 0;
          }

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;

            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // If the location is outside the bounds of the input image,
              // use zero as a default value.
              if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height)) {

                const int container_offset = (filter_y * filter_width + filter_x) * num_packed_containers;
                int input_channel = 0;
                for (int channel_container = 0; channel_container < num_packed_containers; ++channel_container) {
                  CONTAINER_T filter_vals = filter_data[container_offset + channel_container];
                  int number_elements_in_container = std::min(output_depth - channel_container * items_per_container, items_per_container);
                  for (int element = 0; element < number_elements_in_container; element++) {
                    const unsigned int output_channel = channel_container*items_per_container + element;
                    int32_t input_val = input_data[Offset(input_shape, b, in_y, in_x, input_channel)];
                    int32_t filter_val = filter_vals & mask;
                    filter_vals >>= bits_per_item;
                    accbuf[output_channel] += (filter_val + filter_offset) * (input_val + input_offset);
                    if ((output_channel+1) % depth_multiplier == 0) {
                      input_channel ++;
                    }
                  }
                }
              }
            }
          }
          for (int oc =0; oc < output_depth; ++oc) {
            int32_t acc = accbuf[oc];
            if (bias_data) {
              acc += bias_data[oc];
            }
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                static_cast<uint8_t>(acc);
          }

        }  // out_x
      } // out_y
    } // batch b
  }
};

}  // namespace depthwise_conv

inline void DepthwiseConvPackedFilter(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8_t* input_data, const RuntimeShape& filter_shape,
    const void* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    uint8_t* output_data,
    const TfLiteCustomSub8BitPackingDetails& packing_details,
    int32_t *acc_buf) {
  // We need to allocated output_deptch size buffer for accumulators.

  unsigned int bits_per_item = packing_details.bits_per_item;
  unsigned int container_bits = packing_details.container_bits;
  unsigned int packed_minor_dims = packing_details.packed_minor_dims;

  // TODO Check alignment run-length marches minor dimension.
  TFLITE_CHECK(packed_minor_dims == 1);
  switch (bits_per_item) {
    case 4: {
      TFLITE_CHECK(container_bits == 8);
      using KERNEL = depthwise_conv::DepthwiseConvPackedFilter<uint8_t, 4, 8 / 4>;
      KERNEL::Run(params, input_shape, input_data, filter_shape,
                  static_cast<const uint8_t*>(filter_data), bias_shape,
                  bias_data, output_shape, output_data, acc_buf);
      return;
    }
    case 5: {
      TFLITE_CHECK(container_bits == 16);
      using KERNEL = depthwise_conv::DepthwiseConvPackedFilter<uint16_t, 5, 16 / 5>;
      KERNEL::Run(params, input_shape, input_data, filter_shape,
                  static_cast<const uint16_t*>(filter_data), bias_shape,
                  bias_data, output_shape, output_data, acc_buf);
      return;
    }
    case 6: {
      TFLITE_CHECK(container_bits == 32);
      using KERNEL = depthwise_conv::DepthwiseConvPackedFilter<uint32_t, 6, 32 / 6>;
      KERNEL::Run(params, input_shape, input_data, filter_shape,
                  static_cast<const uint32_t*>(filter_data), bias_shape,
                  bias_data, output_shape, output_data, acc_buf);
      return;
    }
    default: {
      TFLITE_ABORT;
      return;
    }
  }
}


}  // namespace micro
}  // namespace ops
}  // namespace tflite


#endif  // TENSORFLOW_LITE_MICRO_KERNELS_DEPTHWISECONV_UINT8_PACKED_FILTER_H_
