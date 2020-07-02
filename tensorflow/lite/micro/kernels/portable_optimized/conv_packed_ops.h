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

namespace tflite {
namespace reference_integer_ops {

template <typename CONTAINER_T, size_t bits_per_item, size_t items_per_container>
void ConvUint8PackedWeights(
        const ConvParams& params,
        const RuntimeShape& input_shape, const uint8* input_data,
        const RuntimeShape& filter_shape, const CONTAINER_T* filter_data,
        const RuntimeShape& bias_shape, const int32* bias_data,
        const RuntimeShape& output_shape, uint8* output_data) {
  // Do nothing yet
}

}  // namespace reference_integer_ops
}  // namespace tflite



#endif /* TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_CONV_PACKED_OPS_H_ */
