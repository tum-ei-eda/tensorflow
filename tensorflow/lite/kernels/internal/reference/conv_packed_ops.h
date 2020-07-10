/*
 * conv_packed_ops.h
 *
 *  Created on: 29.06.2020
 *      Author: krusejakob
 */

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_CONV_PACKED_OPS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_CONV_PACKED_OPS_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace reference_ops {

template <typename CONTAINER_T, size_t bits_per_item, size_t items_per_container>
void ConvUint8PackedWeights(
        const ConvParams& params,
        const RuntimeShape& input_shape, const uint8* input,
        const RuntimeShape& filter_shape, const CONTAINER_T* filter,
        const RuntimeShape& bias_shape, const int32* bias,
        const RuntimeShape& output_shape, uint8* output) {
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;

  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int* in_dims = reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());

  // Packing constants
  const int num_packed_containers = std::ceil((float)input_depth / items_per_container);
  const int32 mask = (1<<bits_per_item)-1;

  for (int batch = 0; batch < batches; ++batch) {
    uint32 offset_input0 = batch * in_dims[1];
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;

          int32 acc = 0;

          unsigned int container_offset = out_channel * filter_height * filter_width * num_packed_containers;

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            uint32 offset_container1 = container_offset + filter_y * filter_width * num_packed_containers;
            uint32 offset_input1 = (offset_input0 + in_y) * in_dims[2];
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              uint32 offset_container2 = offset_container1 + filter_x * num_packed_containers;
              uint32 offset_input2 = (offset_input1 + in_x) * in_dims[3];
              if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                                  (in_y < input_height)) {

                for (int channel_container = 0; channel_container < num_packed_containers; ++channel_container) {
                  CONTAINER_T filter_vals = filter[offset_container2 + channel_container];
                  int number_elements_in_container = std::min(input_depth - channel_container * items_per_container, items_per_container);
                  for (int element = 0; element < number_elements_in_container; element++) {
                    int32 input_val = input[offset_input2 + channel_container*items_per_container + element];
                    int32 filter_val = filter_vals & mask;
                    filter_vals >>= bits_per_item;
                    acc += (filter_val + filter_offset) * (input_val + input_offset);
                  }
                }
              }
            }
          }
          if (bias) {
            acc += bias[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<uint8>(acc);
        }
      }
    }
  }
}

TfLiteStatus EvalConvQuantizedPacked(
        const ConvParams &params,
        const TfLiteTensor* input,
        const TfLiteTensor* filter, const TfLiteTensor* bias,
        TfLiteTensor* output,
        TfLiteContext* context,
        const TfLiteCustomSub8BitPackingDetails &custom) {

  TF_LITE_KERNEL_LOG(context, "Using packed implementation of convolutional layer with %d bit.", custom.bits_per_item);

  unsigned int bits_per_item = custom.bits_per_item;
  unsigned int container_bits = custom.container_bits;
  unsigned int packed_minor_dims =  custom.packed_minor_dims;
  switch (bits_per_item) {

      case 4: {
          if(container_bits != 8)
            break;
          reference_ops::ConvUint8PackedWeights<uint8_t, 4, 8 / 4>(
              params,
              GetTensorShape(input), GetTensorData<uint8_t>(input),
              GetTensorShape(filter), GetTensorData<uint8_t>(filter),
              GetTensorShape(bias), GetTensorData<int32_t>(bias),
              GetTensorShape(output), GetTensorData<uint8_t>(output)
              );
          return kTfLiteOk;
      }
      case 5: {
          if(container_bits != 16)
            break;
          reference_ops::ConvUint8PackedWeights<uint16_t, 5, 16 / 5>(
              params,
              GetTensorShape(input), GetTensorData<uint8_t>(input),
              GetTensorShape(filter), GetTensorData<uint16_t>(filter),
              GetTensorShape(bias), GetTensorData<int32_t>(bias),
              GetTensorShape(output), GetTensorData<uint8_t>(output)
              );
          return kTfLiteOk;
      }
      case 6: {
          if(container_bits != 32)
            break;
          reference_ops::ConvUint8PackedWeights<uint32_t, 6, 32 / 6>(
                          params,
                          GetTensorShape(input), GetTensorData<uint8_t>(input),
                          GetTensorShape(filter), GetTensorData<uint32_t>(filter),
                          GetTensorShape(bias), GetTensorData<int32_t>(bias),
                          GetTensorShape(output), GetTensorData<uint8_t>(output)
                          );
          return kTfLiteOk;
      }
      default: {
          TF_LITE_KERNEL_LOG(context, " Packed Weight bitwidth (%d) not supported.",
                             bits_per_item);
          return kTfLiteError;
      }
  }
  TF_LITE_KERNEL_LOG(context, "Container bitwidth %d not supported for %d bit packed values",
                     container_bits, bits_per_item);
  return kTfLiteError;
}

}  // namespace reference_ops
}  // namespace tflite


#endif /* TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_CONV_PACKED_OPS_H_ */
