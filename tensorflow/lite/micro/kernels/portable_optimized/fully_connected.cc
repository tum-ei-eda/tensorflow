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

// PORTABLE OPTIMIZED

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
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/micro/kernels/fully_connected_packed_weights.h"
#include "tensorflow/lite/micro/kernels/portable_optimized/fully_connected_op_data.h"
#include "tensorflow/lite/micro/kernels/pointer_collector.h"
#include "tensorflow/lite/micro/kernels/static_init_support.h"


//#define IFX_DEBUG_LOGGING 1
#if IFX_DEBUG_LOGGING
#include <iostream>
#endif


namespace tflite {
namespace ops {
namespace micro {
namespace fully_connected {

KERNEL_VARIANT_COLLECT_INFO(
  "fully_connected", 
  "struct OpData;\n",
  "#include \"tensorflow/lite/micro/kernels/portable_optimized/fully_connected_op_data.h\"",
  "OpData",
  "   TfLiteContext* context, TfLiteFullyConnectedParams* params,\n"
  "   OpData* opData, const TfLiteTensor* input, const TfLiteTensor* weights,\n"
  "   const TfLiteTensor* bias, TfLiteTensor* output"
);

typedef TfLiteStatus (*EvalVariantFptr)(TfLiteContext* context, TfLiteFullyConnectedParams* params,
    OpData* opData, const TfLiteTensor* input, const TfLiteTensor* weights,
    const TfLiteTensor* bias, TfLiteTensor* output);

#if TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT

static CppItems *static_opdata(OpData &od, size_t len_sowf)
{
  auto init = new CppItems();

  CppNamedVec<int32_t> sowf("sum_of_weights_factor", "int32_t", od.sum_of_weights_factor, len_sowf);
  
  *init << od.output_multiplier
       << od.output_shift
       << od.output_activation_min
       << od.output_activation_max
       << od.input_quantized_index
       << sowf
       << od.eval_function;

  return init;
}
#endif


constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

TfLiteStatus CalculateOpData(TfLiteContext* context,
                             TfLiteFullyConnectedParams* params,
                             TfLiteType data_type, const TfLiteTensor* input,
                             const TfLiteTensor* weights,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, weights, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }
  return status;
}

template <typename T>
void PrecomputeSumOfWeightsFactor(const int32_t* bias, const T* weights,
                                         int32_t* sum_of_weights_factor,
                                         int cols, int rows,
                                         int32_t weights_offset,
                                         int32_t input_offset) {
  for (int row = 0; row < rows; row++) {
    int32_t sum_of_weights = 0;
    for (int col = 0; col < cols; col++) {
      sum_of_weights += weights[col];
    }
    weights += cols;
    sum_of_weights_factor[row] =
        (sum_of_weights + cols * weights_offset) * input_offset;
    if (bias) {
      sum_of_weights_factor[row] += bias[row];
    }
  }
}

#if TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  EvalVariantFptr recordedVariant();
  OpData *recordedStaticOpData();
#endif

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
#if TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  return recordedStaticOpData();
#else
  void* raw = context->AllocatePersistentBuffer(context, sizeof(OpData));
  OpData* data = reinterpret_cast<OpData*>(raw);
  *data = {};
  return raw;
#endif
}

void Free(TfLiteContext* context, void* buffer) {}


template <typename T>
inline void CalculateOutputNodes(T* output, const T* input, const T* weights,
                                 const int32_t* sum_of_weights_factor,
                                 int32_t sum_of_inputs_factor, int accum_depth,
                                 int output_depth, int32_t output_offset,
                                 int32_t output_multiplier, int output_shift,
                                 int32_t activation_min,
                                 int32_t activation_max) {
  for (int out_c = 0; out_c < output_depth; out_c++) {
    // Multiply and accumulate inputs and weights
    int32_t accum = *sum_of_weights_factor + sum_of_inputs_factor;
    for (int d = 0; d < accum_depth; ++d) {
      accum += weights[d] * input[d];
    }
    // Re-quantize and clamp
    accum =
        MultiplyByQuantizedMultiplier(accum, output_multiplier, output_shift);
    accum += output_offset;
    accum = ActivationFunctionWithMinMax(accum, activation_min, activation_max);
    *output = static_cast<T>(accum);
    // Increment pointers
    output++;
    sum_of_weights_factor++;
    weights += accum_depth;
  }
}

template <typename T>
TfLiteStatus EvalQuantized(
    TfLiteContext* context, TfLiteFullyConnectedParams* params, OpData* opData,
    const TfLiteTensor* input, const TfLiteTensor* weights,
    const TfLiteTensor* bias, TfLiteTensor* output) {
  // Get input info
  const T* input_data = GetTensorData<T>(input);

  // Get weights info
  const T* weights_data = GetTensorData<T>(weights);
  const int32_t weights_offset = -weights->params.zero_point;
  RuntimeShape weights_shape = GetTensorShape(weights);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int accum_depth = weights_shape.Dims(weights_dim_count - 1);

  // Get output info
  T* output_data = GetTensorData<T>(output);
  const int32_t output_offset = output->params.zero_point;
  RuntimeShape output_shape = GetTensorShape(output);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);
  const int32_t output_multiplier = opData->output_multiplier;
  // TODO(b/138810107): Figure out whether output shift should be inverted
  const int output_shift = -opData->output_shift;
  const int32_t output_activation_min = opData->output_activation_min;
  const int32_t output_activation_max = opData->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, weights_shape.Dims(weights_dim_count - 2));

  // Get factor pre-computed in the Prepare-phase
  const int32_t* sum_of_weights_factor = opData->sum_of_weights_factor;

  for (int b = 0; b < batches; ++b) {
    // Pre-compute factor for this output-batch
    int32_t sum_of_inputs_factor = 0;
    if (weights_offset != 0) {
      for (int d = 0; d < accum_depth; ++d) {
        sum_of_inputs_factor += input_data[d];
      }
      sum_of_inputs_factor *= weights_offset;
    }
    // Calculate output-nodes using pre-computed factors
    CalculateOutputNodes(output_data, input_data, weights_data,
                         sum_of_weights_factor, sum_of_inputs_factor,
                         accum_depth, output_depth, output_offset,
                         output_multiplier, output_shift, output_activation_min,
                         output_activation_max);
    output_data += output_depth;
    input_data += accum_depth;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedInt8(
    TfLiteContext* context, TfLiteFullyConnectedParams* params, OpData* opData,
    const TfLiteTensor* input, const TfLiteTensor* weights,
    const TfLiteTensor* bias, TfLiteTensor* output) {
  return EvalQuantized<int8_t>(context, params, opData,
                               input, weights, bias, output);
}

TfLiteStatus EvalQuantizedUInt8(
    TfLiteContext* context, TfLiteFullyConnectedParams* params, OpData* opData,
    const TfLiteTensor* input, const TfLiteTensor* weights,
    const TfLiteTensor* bias, TfLiteTensor* output) {
  return EvalQuantized<uint8_t>(context, params, opData,
                               input, weights, bias, output);
}


TfLiteStatus EvalQuantizedUint8WithOutputInt16(
    TfLiteContext* context, TfLiteFullyConnectedParams* params, OpData* opData,
        const TfLiteTensor* input, const TfLiteTensor* weights,
        const TfLiteTensor* bias, TfLiteTensor* output) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -weights->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = opData->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -opData->output_shift;
  op_params.quantized_activation_min = opData->output_activation_min;
  op_params.quantized_activation_max = opData->output_activation_max;
  reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
      GetTensorShape(weights), GetTensorData<uint8_t>(weights),
      GetTensorShape(bias), GetTensorData<int32_t>(bias),
      GetTensorShape(output), GetTensorData<int16_t>(output));
  return kTfLiteOk;
}

template <typename CONTAINER_T, size_t bits_per_item,
              size_t items_per_container>
TfLiteStatus PackedFullyConnected<CONTAINER_T, bits_per_item, items_per_container>::EvalUint8PackedWeights(
    TfLiteContext* context, TfLiteFullyConnectedParams* params, OpData* opData,
    const TfLiteTensor* input, const TfLiteTensor* weights,
    const TfLiteTensor* bias, TfLiteTensor* output) {

  const RuntimeShape& input_shape = GetTensorShape(input);
  auto input_data = GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = GetTensorShape(weights);
  auto filter_data = GetTensorData<CONTAINER_T>(weights);
  const RuntimeShape& bias_shape = GetTensorShape(bias);
  auto bias_data = GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = GetTensorShape(output);
  auto output_data = GetTensorData<uint8_t>(output);

  tflite::FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -weights->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = opData->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -opData->output_shift;
  op_params.quantized_activation_min = opData->output_activation_min;
  op_params.quantized_activation_max = opData->output_activation_max;

  FullyConnectedUint8PackedWeights<CONTAINER_T, bits_per_item,
                                   items_per_container>(
      op_params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data);
  return kTfLiteOk;
}

template
class PackedFullyConnected<uint8_t, 4, 8/4>;

template
class PackedFullyConnected<uint16_t, 4, 16/4>;

template
class PackedFullyConnected<uint32_t, 4, 32/4>;

template
class PackedFullyConnected<uint16_t, 5, 16/5>;

template
class PackedFullyConnected<uint32_t, 5, 32/5>;

template
class PackedFullyConnected<uint32_t, 6, 32/6>;

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteFullyConnectedParams* params, OpData* opData,
               const TfLiteTensor* input, const TfLiteTensor* weights,
               const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  tflite::reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(weights), GetTensorData<float>(weights),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if ! TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  int rows = 0;
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (weights->type == kTfLiteInt8 || weights->type == kTfLiteUInt8) {
    // Calculate data for quantized operation
    auto* params =
        reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input->type, input,
                                          weights, bias, output, data));
    // Pre-compute factors for quantized operation
    const int32_t weights_offset = -weights->params.zero_point;
    RuntimeShape weights_shape = GetTensorShape(weights);
    TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
    rows = weights_shape.Dims(0);
    const int cols = weights_shape.Dims(1);

    void* raw = context->AllocatePersistentBuffer(context, sizeof(int32_t) * rows);
    data->sum_of_weights_factor = reinterpret_cast<int32_t*>(raw);
    const int32_t input_offset = -input->params.zero_point;
    const int32_t* bias_data = GetTensorData<int32_t>(bias);

    if (weights->type == kTfLiteInt8) {
      PrecomputeSumOfWeightsFactor<int8_t>(bias_data,
                                           GetTensorData<int8_t>(weights),
                                           data->sum_of_weights_factor, cols,
                                           rows, weights_offset, input_offset);
    } else if (weights->quantization.details.type ==
               kTfLiteSub8BitPackedUniformDetail) {
      // TODO implement pre-compute sum-of-weights for packed weights...
    } else {
      PrecomputeSumOfWeightsFactor<uint8_t>(bias_data,
                                            GetTensorData<uint8_t>(weights),
                                            data->sum_of_weights_factor, cols,
                                            rows, weights_offset, input_offset);
    }
  }

  bool use_packed = (weights->quantization.details.type == kTfLiteSub8BitPackedUniformDetail);

  switch (weights->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalFloat);
      break;
    case kTfLiteInt8:
      switch (output->type) {
        case kTfLiteInt8:
          data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalQuantizedInt8);
          break;
        default:
          TF_LITE_KERNEL_LOG(context, "Quantized int8 _t expects output int8");
          return kTfLiteError;
      }
      break;
    case kTfLiteUInt8:
      switch (output->type) {
        case kTfLiteUInt8:
          // TODO packed variant pre-computing to eliminate tensor_offset
          // addition
          if (use_packed) {
            const TfLiteCustomSub8BitPackingDetails& custom =
                  *weights->quantization.details.data.custom_sub8bit_packing;
            unsigned int bits_per_item = custom.bits_per_item;
            unsigned int container_bits = custom.container_bits;
            switch (bits_per_item) {
              case 4: {
                if (container_bits == 8) {
                  data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                      (PackedFullyConnected<uint8_t, 4, 8/4>::EvalUint8PackedWeights));
                } else if (container_bits == 16) {
                  data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                      (PackedFullyConnected<uint16_t, 4, 16/4>::EvalUint8PackedWeights));
                } else if (container_bits == 32) {
                  data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                      (PackedFullyConnected<uint32_t, 4, 32/4>::EvalUint8PackedWeights));
                } else {
                  TF_LITE_KERNEL_LOG(context, " Packed Implementation not supported.");
                  return kTfLiteError;
                }
                break;
              }
              case 5: {
                if (container_bits == 16) {
                  data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                      (PackedFullyConnected<uint16_t, 5, 16/5>::EvalUint8PackedWeights));
                } else if (container_bits == 32) {
                  data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                      (PackedFullyConnected<uint32_t, 5, 32/5>::EvalUint8PackedWeights));
                } else {
                  TF_LITE_KERNEL_LOG(context, " Packed Implementation not supported.");
                  return kTfLiteError;
                }
                break;
              }
              case 6: {
                if (container_bits == 32) {
                  data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                      (PackedFullyConnected<uint32_t, 6, 32/6>::EvalUint8PackedWeights));
                } else {
                  TF_LITE_KERNEL_LOG(context, " Packed Implementation not supported.");
                  return kTfLiteError;
                }
                break;
              }
              default: {
                TF_LITE_KERNEL_LOG(context, " Packed Weight bitwidth (%d) not supported.",
                                   bits_per_item);
                return kTfLiteError;
              }
            }

          } else {
            data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalQuantizedUInt8);
          }
          break;
        case kTfLiteInt16:
          data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalQuantizedUint8WithOutputInt16);
          break;
        default:
          TF_LITE_KERNEL_LOG(context,
                             "Quantized uint8_t expects output uint8_t or int16");
          return kTfLiteError;
      }
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Weight type %d not currently supported.",
                         weights->type);
      return kTfLiteError;
  }

  TF_LITE_MICRO_RECORD_OP_USER_DATA("fully_connected", static_opdata(*data, static_cast<size_t>(rows)));

#endif // ! TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  return kTfLiteOk;
}


TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* opData = reinterpret_cast<OpData*>(node->user_data);

  return opData->eval_function(context, params, opData,
                             input, weights, bias, output);
}

}  // namespace fully_connected

TfLiteRegistration Register_FULLY_CONNECTED() {
  return {/*init=*/fully_connected::Init,
          /*free=*/fully_connected::Free,
          /*prepare=*/fully_connected::Prepare,
          /*invoke=*/fully_connected::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
