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

// Support recording of selected kernel variant in prepare phase for static
// extraction for a fixed tflite model.

// TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT:
//  When set the names of kernel variants eval functions recorded and can be
//  dumped
// via PointerCollect API.
// TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
//   When set prepare phase kernel variant selection code is dropped with
// the eval functions recorded in tflite::micro::kernels::conv::eval_functions
// used instead.
//
// Benefits smaller binary, used unnecessary eval function variants are not
// lnked.

#include "tensorflow/lite/kernels/internal/reference/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/conv_packed_ops.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/portable_optimized/conv_op_data.h"
#include "tensorflow/lite/micro/kernels/static_data_utils.h"
#include "tensorflow/lite/micro/kernels/static_init_support.h"

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

namespace tflite {
namespace ops {
namespace micro {
namespace conv {

KERNEL_VARIANT_COLLECT_INFO(
    "conv", "struct OpData;\n",
    "#include "
    "\"tensorflow/lite/micro/kernels/portable_optimized/conv_op_data.h\"",
    "OpData",
    "    TfLiteConvParams* params, OpData* data,\n"
    "    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter, \n"
    "    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output, "
    "TfLiteContext* context");

#if TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT

static CppItems* static_opdata(OpData& od, size_t output_depth) {
  auto init = new CppItems();

  *init << TfLitePaddingValuesSubStruct(od.padding) << od.input_zero_point
        << od.filter_zero_point << od.output_zero_point << od.output_multiplier
        << od.output_shift
        << CppNamedVec<int32_t>("per_channel_output_multiplier", "int32_t",
                                od.per_channel_output_multiplier, output_depth)
        << CppNamedVec<int32_t>("per_channel_output_shift", "int32_t",
                                od.per_channel_output_shift, output_depth)
        << od.output_activation_min << od.output_activation_max
        << CppNamedVec<int32_t>("sum_of_filters_factor", "int32_t",
                                od.sum_of_filters_factor, output_depth);
  if (od.custom_sub8bit_packing) {
    *init << TfLiteCustomSub8BitPackingDetailsStructPtr(
        "custom_sub8bit_packing", *od.custom_sub8bit_packing);
  } else {
    *init << "nullptr";
  }
  *init << od.eval_function;

  return init;
}
#endif

// Defined in code generated via PointerCollector::writeCppSyntaxPointerTable

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

#if TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
EvalVariantFptr recordedVariant();
#endif

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
#if TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  OpData* recordedStaticOpData();
  return recordedStaticOpData();
#else
  void* raw = context->AllocatePersistentBuffer(context, sizeof(OpData));
  TFLITE_DCHECK(raw != nullptr);
  OpData* data = reinterpret_cast<OpData*>(raw);
  *data = {};
  return raw;
#endif
}

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, int width, int height,
                             int filter_width, int filter_height, int out_width,
                             int out_height, const TfLiteType data_type,
                             OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int output_channels = filter->dims->data[kConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift),
        output_channels));
  }
  return kTfLiteOk;
}

template <typename T>
inline void PrecomputeSumOfFiltersFactor(const int32_t* bias,
                                         const TfLiteTensor* filters,
                                         int32_t* sum_of_filters_factor,
                                         RuntimeShape filter_shape,
                                         int32_t input_offset,
                                         int32_t filter_offset = 0) {
  if (filters->type == kTfLiteInt8) {
    // Ensure that the filter offset is 0 in the signed integer case
    TFLITE_DCHECK_EQ(filter_offset, 0);
  }
  const T* filter_data = GetTensorData<T>(filters);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int input_depth = filter_shape.Dims(3);
  const int output_depth = filter_shape.Dims(0);

  int filter_size = filter_width * filter_height * input_depth;

  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    int32_t sum_of_filter_factor = filter_size * filter_offset;

    for (int filter_index = filter_size * out_channel;
         filter_index < filter_size + filter_size * out_channel;
         ++filter_index) {
      sum_of_filter_factor += filter_data[filter_index];
    }
    sum_of_filters_factor[out_channel] = sum_of_filter_factor * input_offset;
    if (bias) {
      sum_of_filters_factor[out_channel] += bias[out_channel];
    }
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if !TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  int input_width = input->dims->data[2];
  int input_height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int output_width = output->dims->data[2];
  int output_height = output->dims->data[1];

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  int output_depth = 0;
  if (filter->type == kTfLiteInt8 || filter->type == kTfLiteUInt8) {
    const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
    const int32_t* bias_data = GetTensorData<int32_t>(bias);

    const int32_t filter_offset = -data->filter_zero_point;
    RuntimeShape filter_shape = GetTensorShape(filter);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);

    output_depth = filter_shape.Dims(0);

    void* raw = context->AllocatePersistentBuffer(
        context, sizeof(int32_t) * output_depth);
    data->sum_of_filters_factor = reinterpret_cast<int32_t*>(raw);

    raw = context->AllocatePersistentBuffer(context,
                                            sizeof(int32_t) * output_depth);
    data->per_channel_output_multiplier = reinterpret_cast<int32_t*>(raw);

    raw = context->AllocatePersistentBuffer(context,
                                            sizeof(int32_t) * output_depth);
    data->per_channel_output_shift = reinterpret_cast<int32_t*>(raw);

    // Precompute the sum of filters
    const int32_t input_offset = -data->input_zero_point;
    if (filter->type == kTfLiteUInt8) {
      if (filter->quantization.details.type !=
          kTfLiteSub8BitPackedUniformDetail) {
        PrecomputeSumOfFiltersFactor<uint8_t>(
            bias_data, filter, data->sum_of_filters_factor, filter_shape,
            input_offset, filter_offset);
      }
    } else {
      PrecomputeSumOfFiltersFactor<int8_t>(bias_data, filter,
                                           data->sum_of_filters_factor,
                                           filter_shape, input_offset, 0);
    }
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type, data));

  // Determine which version to use
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;
  bool use_reference =
      ((dilation_width_factor != 1) || (dilation_height_factor != 1));
  bool use_padding =
      (data->padding.height != 0 || data->padding.width != 0 ||
       data->padding.height_offset != 0 || data->padding.width_offset != 0);
  bool use_packed =
      (filter->quantization.details.type == kTfLiteSub8BitPackedUniformDetail);
  if (use_packed) {
    data->custom_sub8bit_packing =
        filter->quantization.details.data.custom_sub8bit_packing;
  } else {
    data->custom_sub8bit_packing = nullptr;
  }
  // Set the function pointer that is used during inference here
  switch (filter->type) {
    case kTfLiteFloat32: {
      data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalConvFloat);
      break;
    }
    case kTfLiteInt8: {
      if (use_reference) {
        data->eval_function =
            TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalConvInt8Reference);
      } else if (use_padding) {
        data->eval_function =
            TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalConvInt8Padding);
      } else {
        data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalConvInt8);
      }
      break;
    }
    case kTfLiteUInt8: {
      if (use_packed) {
        const TfLiteCustomSub8BitPackingDetails &custom = *data->custom_sub8bit_packing;
        unsigned int bits_per_item = custom.bits_per_item;
        unsigned int container_bits = custom.container_bits;
        switch (bits_per_item) {
            case 4: {
                if(container_bits == 8) {
                  data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                      (PackedConv<uint8_t, 4, 8/4>::EvalUint8PackedWeights));
                } else if (container_bits == 16) {
                  data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                      (PackedConv<uint16_t, 4, 16/4>::EvalUint8PackedWeights));
                } else if (container_bits == 32) {
                  data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                      (PackedConv<uint32_t, 4, 32/4>::EvalUint8PackedWeights));
                } else {
                  TF_LITE_KERNEL_LOG(context, " Packed Implementation not supported.");
                  return kTfLiteError;
                }
                break;
            }
            case 5: {
              if (container_bits == 16) {
                data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                    (PackedConv<uint16_t, 5, 16/5>::EvalUint8PackedWeights));
              } else if (container_bits == 32) {
                data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                    (PackedConv<uint32_t, 5, 32/5>::EvalUint8PackedWeights));
              } else {
                TF_LITE_KERNEL_LOG(context, " Packed Implementation not supported.");
                return kTfLiteError;
              }
              break;
            }
            case 6: {
              if (container_bits == 32) {
                data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
                    (PackedConv<uint32_t, 6, 32/6>::EvalUint8PackedWeights));
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
      } else if (use_reference) {
        data->eval_function =
            TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalConvUInt8Reference);
      } else if (use_padding) {
        data->eval_function =
            TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalConvUInt8Padding);
      } else {
        data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalConvUInt8);
      }
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }
#endif

  TF_LITE_MICRO_RECORD_OP_USER_DATA("conv", static_opdata(*data, output_depth));

  return kTfLiteOk;
}

void Free(TfLiteContext* context, void* buffer) {}

template <typename CONTAINER_T, size_t bits_per_item,
              size_t items_per_container>
TfLiteStatus PackedConv<CONTAINER_T, bits_per_item, items_per_container>::EvalUint8PackedWeights(
    TfLiteConvParams* params, OpData* data,
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output,
    TfLiteContext* context) {
  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = -data->filter_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  ConvUint8PackedWeights<CONTAINER_T, bits_per_item, items_per_container>(
        op_params,
        tflite::micro::GetTensorShape(input), tflite::micro::GetTensorData<uint8_t>(input),
        tflite::micro::GetTensorShape(filter), tflite::micro::GetTensorData<CONTAINER_T>(filter),
        tflite::micro::GetTensorShape(bias), tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output), tflite::micro::GetTensorData<uint8_t>(output)
        );
    return kTfLiteOk;
}

template
class PackedConv<uint8_t, 4, 8/4>;

template
class PackedConv<uint16_t, 4, 16/4>;

template
class PackedConv<uint32_t, 4, 32/4>;

template
class PackedConv<uint16_t, 5, 16/5>;

template
class PackedConv<uint32_t, 5, 32/5>;

template
class PackedConv<uint32_t, 6, 32/6>;

TfLiteStatus EvalConvUInt8Reference(TfLiteConvParams* params, OpData* data,
                                    const TfLiteEvalTensor* input,
                                    const TfLiteEvalTensor* filter,
                                    const TfLiteEvalTensor* bias,
                                    TfLiteEvalTensor* output,
                                    TfLiteContext* context) {
  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = -data->filter_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  TfLiteEvalTensor* im2col = nullptr;

  reference_ops::Conv(op_params, tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<uint8_t>(input),
                      tflite::micro::GetTensorShape(filter),
                      tflite::micro::GetTensorData<uint8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetTensorData<int32_t>(bias),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<uint8_t>(output),
                      tflite::micro::GetTensorShape(im2col),
                      tflite::micro::GetTensorData<uint8_t>(im2col), nullptr);
  return kTfLiteOk;
}

TfLiteStatus EvalConvInt8Reference(TfLiteConvParams* params, OpData* data,
                                   const TfLiteEvalTensor* input,
                                   const TfLiteEvalTensor* filter,
                                   const TfLiteEvalTensor* bias,
                                   TfLiteEvalTensor* output,
                                   TfLiteContext* context) {
  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = -data->filter_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  reference_integer_ops::ConvPerChannel(
      op_params, data->per_channel_output_multiplier,
      data->per_channel_output_shift, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalConvUInt8(TfLiteConvParams* params, OpData* data,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           const TfLiteEvalTensor* bias,
                           TfLiteEvalTensor* output, TfLiteContext* context) {
  const int32_t filter_offset = -data->filter_zero_point;
  const int32_t output_offset = data->output_zero_point;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const uint8_t* filter_data = tflite::micro::GetTensorData<uint8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;

  const int32_t output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
  const int32_t output_activation_min = data->output_activation_min;
  const int32_t output_activation_max = data->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    uint32_t offset_input0 = batch * in_dims[1];
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        uint32_t filter_index = 0;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = out_x * stride_width;
          const int in_y_origin = out_y * stride_height;
          int32_t acc = 0;

          const int32_t ker_y_start = MAX(0, -in_y_origin);
          const int32_t ker_x_start = MAX(0, -in_x_origin);

          const int32_t ker_y_end =
              MIN(filter_height, input_height - in_y_origin);
          const int32_t ker_x_end =
              MIN(filter_width, input_width - in_x_origin);

          for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            uint32_t offset_input1 = (offset_input0 + in_y) * in_dims[2];
            for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              uint32_t offset_input2 = (offset_input1 + in_x) * in_dims[3];
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                int32_t input_val = input_data[offset_input2 + in_channel];
                int32_t filter_val = filter_data[filter_index++];

                acc += (filter_val + filter_offset) * input_val;
              }
            }
          }

          acc += data->sum_of_filters_factor[out_channel];
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                              output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<uint8_t>(acc);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalConvUInt8Padding(TfLiteConvParams* params, OpData* data,
                                  const TfLiteEvalTensor* input,
                                  const TfLiteEvalTensor* filter,
                                  const TfLiteEvalTensor* bias,
                                  TfLiteEvalTensor* output,
                                  TfLiteContext* context) {
  const int32_t input_offset = -data->input_zero_point;
  const int32_t filter_offset = -data->filter_zero_point;
  const int32_t output_offset = data->output_zero_point;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const uint8_t* filter_data = tflite::micro::GetTensorData<uint8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;

  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int32_t output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
  const int32_t output_activation_min = data->output_activation_min;
  const int32_t output_activation_max = data->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    uint32_t offset_input0 = batch * in_dims[1];
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          uint32_t offset_filter0 = out_channel * fi_dims[1];

          int32_t acc = 0;

          const int32_t ker_y_start = MAX(0, -in_y_origin);
          const int32_t ker_x_start = MAX(0, -in_x_origin);

          const int32_t ker_y_end =
              MIN(filter_height, input_height - in_y_origin);
          const int32_t ker_x_end =
              MIN(filter_width, input_width - in_x_origin);

          for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            uint32_t offset_filter1 = (offset_filter0 + filter_y) * fi_dims[2];
            uint32_t offset_input1 = (offset_input0 + in_y) * in_dims[2];
            for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              uint32_t offset_filter2 =
                  (offset_filter1 + filter_x) * fi_dims[3];
              uint32_t offset_input2 = (offset_input1 + in_x) * in_dims[3];
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                int32_t input_val = input_data[offset_input2 + in_channel];
                int32_t filter_val = filter_data[offset_filter2 + in_channel];

                acc +=
                    (filter_val + filter_offset) * (input_val + input_offset);
              }
            }
          }
          if (bias) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                              output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<uint8_t>(acc);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalConvInt8(TfLiteConvParams* params, OpData* data,
                          const TfLiteEvalTensor* input,
                          const TfLiteEvalTensor* filter,
                          const TfLiteEvalTensor* bias,
                          TfLiteEvalTensor* output, TfLiteContext* context) {
  const int32_t output_offset = data->output_zero_point;

  const int32_t* output_multiplier = data->per_channel_output_multiplier;
  const int32_t* output_shift = data->per_channel_output_shift;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int32_t output_activation_min = data->output_activation_min;
  const int32_t output_activation_max = data->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    uint32_t offset_input0 = batch * in_dims[1];
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          uint32_t offset_filter0 = out_channel * fi_dims[1];

          const int32_t ker_y_start = MAX(0, -in_y_origin);
          const int32_t ker_x_start = MAX(0, -in_x_origin);

          const int32_t ker_y_end =
              MIN(filter_height, input_height - in_y_origin);
          const int32_t ker_x_end =
              MIN(filter_width, input_width - in_x_origin);

          int32_t acc = 0;

          for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            uint32_t offset_filter1 = (offset_filter0 + filter_y) * fi_dims[2];
            uint32_t offset_input1 = (offset_input0 + in_y) * in_dims[2];

            for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              uint32_t offset_filter2 =
                  (offset_filter1 + filter_x) * fi_dims[3];
              uint32_t offset_input2 = (offset_input1 + in_x) * in_dims[3];

              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                int32_t input_val = input_data[offset_input2 + in_channel];
                int32_t filter_val = filter_data[offset_filter2 + in_channel];
                acc += filter_val * input_val;
              }
            }
          }
          acc += data->sum_of_filters_factor[out_channel];
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalConvInt8Padding(TfLiteConvParams* params, OpData* data,
                                 const TfLiteEvalTensor* input,
                                 const TfLiteEvalTensor* filter,
                                 const TfLiteEvalTensor* bias,
                                 TfLiteEvalTensor* output,
                                 TfLiteContext* context) {
  const int32_t input_offset = -data->input_zero_point;
  const int32_t output_offset = data->output_zero_point;

  const int32_t* output_multiplier = data->per_channel_output_multiplier;
  const int32_t* output_shift = data->per_channel_output_shift;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int32_t output_activation_min = data->output_activation_min;
  const int32_t output_activation_max = data->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    uint32_t offset_input0 = batch * in_dims[1];
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          uint32_t offset_filter0 = out_channel * fi_dims[1];

          const int32_t ker_y_start = MAX(0, -in_y_origin);
          const int32_t ker_x_start = MAX(0, -in_x_origin);

          const int32_t ker_y_end =
              MIN(filter_height, input_height - in_y_origin);
          const int32_t ker_x_end =
              MIN(filter_width, input_width - in_x_origin);

          int32_t acc = 0;

          for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            uint32_t offset_filter1 = (offset_filter0 + filter_y) * fi_dims[2];
            uint32_t offset_input1 = (offset_input0 + in_y) * in_dims[2];

            for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              uint32_t offset_filter2 =
                  (offset_filter1 + filter_x) * fi_dims[3];
              uint32_t offset_input2 = (offset_input1 + in_x) * in_dims[3];

              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                int32_t input_val = input_data[offset_input2 + in_channel];
                int32_t filter_val = filter_data[offset_filter2 + in_channel];
                acc += filter_val * (input_val + input_offset);
              }
            }
          }
          if (bias) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalConvFloat(TfLiteConvParams* params, OpData* data,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           const TfLiteEvalTensor* bias,
                           TfLiteEvalTensor* output, TfLiteContext* context) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  TfLiteEvalTensor* im2col = nullptr;
  reference_ops::Conv(op_params, tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<float>(input),
                      tflite::micro::GetTensorShape(filter),
                      tflite::micro::GetTensorData<float>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetTensorData<float>(bias),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<float>(output),
                      tflite::micro::GetTensorShape(im2col),
                      tflite::micro::GetTensorData<float>(im2col));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  return data->eval_function(params, data, input, filter, bias, output,
                             context);
}

}  // namespace conv

TfLiteRegistration Register_CONV_2D() {
  return {/*init=*/conv::Init,
          /*free=*/conv::Free,
          /*prepare=*/conv::Prepare,
          /*invoke=*/conv::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
