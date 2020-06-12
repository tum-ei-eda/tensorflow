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

#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

//#define IFX_DEBUG_LOGGING 1
#if IFX_DEBUG_LOGGING
#include <iostream>
#endif

namespace tflite {
namespace ops {
namespace micro {
namespace fully_connected {
namespace {

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
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

TfLiteStatus CalculateOpData(TfLiteContext* context,
                             TfLiteFusedActivation activation,
                             TfLiteType data_type, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }
  return status;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = nullptr;
  if (context->AllocatePersistentBuffer(context, sizeof(OpData), &data) ==
      kTfLiteError) {
    return nullptr;
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // @IFX_PATCH@
  if( auto custom_qinfo = filter->quantization.custom )
  {
    TF_LITE_ENSURE_MSG(context, custom_qinfo->size == 6,
                       "Unrecognized custom quantization info block");
    // TODO Check magic number
    // TODO Define TfLite Struct to reinterpret (known aligned!) data
    TF_LITE_ENSURE_MSG(context, custom_qinfo->data[4] >= 4 && custom_qinfo->data[4] <= 6,
                       "Currently unsupported packing format" );
  }


  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  return CalculateOpData(context, params->activation, input->type, input,
                         filter, bias, output, data);
}

TfLiteStatus EvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                               const OpData& data, const TfLiteTensor* input,
                               const TfLiteTensor* filter,
                               const TfLiteTensor* bias, TfLiteTensor* output) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -filter->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = data.output_multiplier;
  // TODO(b/138810107): Figure out whether output shift should be inverted
  op_params.output_shift = -data.output_shift;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;

  reference_integer_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
      GetTensorShape(filter), GetTensorData<int8_t>(filter),
      GetTensorShape(bias), GetTensorData<int32_t>(bias),
      GetTensorShape(output), GetTensorData<int8_t>(output));
  return kTfLiteOk;
}



//
// @IFX_PATCH@PoC
//  Uint8 Quantized fully connect kernel for < 8-bit packed weights
// "little-endian" format (first weight in LSB) ordering assumed.
//
// TODO Use specializations to handle fast case where dimensions
// allow efficient loop-unroll etc.
// accum_container_depth should really be  a params value
//

template <typename CONTAINER_T, size_t bits_per_item, size_t items_per_container>
void EvalFullyConnectedUint8PackedWeightsImpl(
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
#endif
  bool once = false;
  unsigned int final_container_begin = accum_depth-(accum_depth%items_per_container);
  for (int b = 0; b < batches; ++b) {
    for (unsigned int out_c = 0; out_c < output_depth; ++out_c) {
      int32 acc = 0;
      unsigned int t;
      const uint8_t *input_vals;
      CONTAINER_T filter_vals;
      unsigned int i;
      unsigned int last_d = 0;
      unsigned int d = 0;
      unsigned int container = 0;
      for (;;) {
        input_vals = &input_data[b * accum_depth + d];
        filter_vals = filter_data[out_c * accum_container_depth + container];
        i = 0;
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

#if IFX_DEBUG_LOGGING
      if( !once ) {
        std::cout << "RAW ACC " << acc << std::endl;
      }
      once = true;
#endif
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<uint8>(acc);
    }

  }
}

template <typename CONTAINER_T, size_t bits_per_item, size_t items_per_container>
inline void EvalFullyConnectedUint8PackedWeights(
        const FullyConnectedParams& params,
        const TfLiteTensor* input,
        const TfLiteTensor* filter, const TfLiteTensor* bias,
        TfLiteTensor* output) {

    const RuntimeShape &input_shape = GetTensorShape(input);
    auto input_data = GetTensorData<uint8_t>(input);
    const RuntimeShape &filter_shape = GetTensorShape(filter);
    auto filter_data =  GetTensorData<CONTAINER_T>(filter);
    const RuntimeShape &bias_shape = GetTensorShape(bias);
    auto bias_data = GetTensorData<int32_t>(bias);
    const RuntimeShape &output_shape = GetTensorShape(output);
    auto output_data = GetTensorData<uint8>(output);

    // here could "Intercept" arguments for offlikne pre-interpretation
    return EvalFullyConnectedUint8PackedWeightsImpl<CONTAINER_T, bits_per_item, items_per_container>(
            params,
            input_shape, input_data,
            filter_shape, filter_data,
            bias_shape, bias_data,
            output_shape, output_data);
}



TfLiteStatus EvalQuantizedPacked(
        const FullyConnectedParams &params,
        const TfLiteTensor* input,
        const TfLiteTensor* filter, const TfLiteTensor* bias,
        TfLiteTensor* output,
        TfLiteContext* context,
        const TfLiteUInt8Array &custom) {

    unsigned int bits_per_item = custom.data[4];
    unsigned int container_bits = custom.data[5]; // TODO does it even make sense to pass this??

    // TODO Check magic # in custom data vector
    switch (bits_per_item) {

        case 4: {
            assert(container_bits == 8);
            EvalFullyConnectedUint8PackedWeights<uint8_t, 4, 8 / 4>(params, input,
                                                                    filter, bias,
                                                                    output);
            return kTfLiteOk;
        }
        case 5: {
            assert(container_bits == 16);
            EvalFullyConnectedUint8PackedWeights<uint16_t, 5, 16 / 5>(params, input,
                                                                      filter, bias,
                                                                      output);
            return kTfLiteOk;
        }
        case 6: {
            assert(container_bits == 32);
            EvalFullyConnectedUint8PackedWeights<uint32_t, 6, 32 / 6>(params, input,
                                                                      filter, bias,
                                                                      output);
            return kTfLiteOk;
        }
        default: {
            TF_LITE_KERNEL_LOG(context, " Packed Weight bitwidth (%d) not supported.",
                               bits_per_item);
            return kTfLiteError;
        }
    }
}


TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           const OpData& data, const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data.output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -data.output_shift;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;

#define TF_LITE_FULLY_CONNECTED(func, output_data_type)                \
  func(                                                                \
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input), \
      GetTensorShape(filter), GetTensorData<uint8_t>(filter),          \
      GetTensorShape(bias), GetTensorData<int32_t>(bias),              \
      GetTensorShape(output), GetTensorData<output_data_type>(output))
  switch (output->type) {
    case kTfLiteUInt8:
      if( filter->quantization.custom )  {
            return EvalQuantizedPacked(
                    op_params,
                    input, filter, bias, output,
                    context,
                    *filter->quantization.custom);
      } else {
        TF_LITE_FULLY_CONNECTED(reference_ops::FullyConnected, uint8_t);
      }
      break;
    case kTfLiteInt16:
      TF_LITE_FULLY_CONNECTED(reference_ops::FullyConnected, int16_t);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(output->type), output->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFusedActivation activation,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(activation, &output_activation_min,
                           &output_activation_max);
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  tflite::reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(filter), GetTensorData<float>(filter),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32:
      return EvalFloat(context, node, params->activation, input, filter, bias,
                       output);
    case kTfLiteInt8:
      return EvalQuantizedInt8(context, node, data, input, filter, bias,
                               output);

    case kTfLiteUInt8:
      return EvalQuantized(context, node, data, input, filter, bias, output);

    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FULLY_CONNECTED() {
  static TfLiteRegistration r = {/*init=*/fully_connected::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/fully_connected::Prepare,
                                 /*invoke=*/fully_connected::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
