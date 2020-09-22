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

// Support recording of selected kernel variant in prepare phase for static
// extraction for a fixed tflite model.

// TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT:
//  When set the names of kernel variants eval functions recorded and can be
//  dumped
// via PointerCollect API.
// TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
//   When set prepare phase kernel variant selection code is dropped with
// the eval functions recorded in
// tflite::micro::kernels::depthwise_conv::eval_functions used instead.
//
// Benefits smaller binary, used unnecessary eval function variants are not
// lnked.

#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/portable_optimized/depthwise_conv_op_data.h"
#include "tensorflow/lite/micro/kernels/static_data_utils.h"
#include "tensorflow/lite/micro/kernels/static_init_support.h"

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {

KERNEL_VARIANT_COLLECT_INFO(
    "depthwise_conv", "struct OpData;\n",
    "#include "
    "\"tensorflow/lite/micro/kernels/portable_optimized/"
    "depthwise_conv_op_data.h\"",
    "OpData",
    "    TfLiteContext* context, const TfLiteDepthwiseConvParams& params,\n"
    "    OpData* data, const TfLiteEvalTensor* input, const TfLiteEvalTensor* "
    "filter, \n"
    "    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output");
//#define TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT 1
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
                                od.sum_of_filters_factor, output_depth)
        << CppNamedVec<int32_t>("acc_buf", "int32_t", od.acc_buf, output_depth);
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

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Depthwise conv is quantized along dimension 3:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

// Size of the cached buffer we'll be using to hold reordered weights.
constexpr int kReshapedFilterDataSize = 1 * 1024;

#if TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
EvalVariantFptr recordedVariant();
#endif

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
#if TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  OpData* recordedStaticOpData();
  return recordedStaticOpData();
#else
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = context->AllocatePersistentBuffer(context, sizeof(OpData));
  TFLITE_DCHECK(data != nullptr);
  return data;
#endif
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  int unused_output_height, unused_output_width;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      filter_height, filter_width, params->padding, &unused_output_height,
      &unused_output_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
    return tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels);
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
  const int num_filters = filter_shape.Dims(3);

  int filter_size = filter_width * filter_height;

  for (int out_channel = 0; out_channel < num_filters; ++out_channel) {
    int32_t sum_of_filter_factor = filter_size * filter_offset;

    for (int filter_index = out_channel;
         filter_index < filter_size * num_filters;
         filter_index += num_filters) {
      sum_of_filter_factor += filter_data[filter_index];
    }
    sum_of_filters_factor[out_channel] = sum_of_filter_factor * input_offset;

    if (bias) {
      sum_of_filters_factor[out_channel] += bias[out_channel];
    }
  }
}

inline void PrecomputeSumOfPackedFiltersFactor(
    const int32_t* bias, const TfLiteTensor* filters,
    int32_t* sum_of_filters_factor, RuntimeShape filter_shape,
    int32_t input_offset, int32_t filter_offset,
    const TfLiteCustomSub8BitPackingDetails& packing_details) {
  if (filters->type == kTfLiteInt8) {
    // Ensure that the filter offset is 0 in the signed integer case
    TFLITE_DCHECK_EQ(filter_offset, 0);
  }
  const uint8_t* filter_data = GetTensorData<uint8_t>(filters);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int num_filters = filter_shape.Dims(3);

  int filter_size = filter_width * filter_height;

  uint32_t mask = (1u << packing_details.bits_per_item) - 1u;
  uint32_t container = 0;
  size_t bits_in_container = 0;
  const uint8_t* filter_byte_p = filter_data;

  for (int out_channel = 0; out_channel < num_filters; ++out_channel) {
    sum_of_filters_factor[out_channel] = filter_size * filter_offset;
  }

  for (int filter_coord = 0; filter_coord < filter_size; ++filter_coord) {
    for (int out_channel = 0; out_channel < num_filters; ++out_channel) {
      if (bits_in_container < packing_details.bits_per_item) {
        container = 0;
        switch (packing_details.container_bits) {
          case 32u:
            container = *reinterpret_cast<const uint32_t*>(filter_byte_p);
            filter_byte_p += 4;
            break;
          case 16u:
            container = *reinterpret_cast<const uint16_t*>(filter_byte_p);
            filter_byte_p += 2;
            break;
          case 8u:
            container = *filter_byte_p;
            filter_byte_p += 1;
            break;
          default:
            TFLITE_ASSERT_FALSE;
        }
        bits_in_container = packing_details.container_bits;
      }
      uint32_t filter_val = (container & mask);
      sum_of_filters_factor[out_channel] += filter_val;
      container >>= packing_details.bits_per_item;
      bits_in_container -= packing_details.bits_per_item;
    }
    bits_in_container = 0;
  }

  for (int out_channel = 0; out_channel < num_filters; ++out_channel) {
    sum_of_filters_factor[out_channel] *= input_offset;
    if (bias) {
      sum_of_filters_factor[out_channel] += bias[out_channel];
    }
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if !TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

  // Per channel quantization is only needed for int8_t inference. For other
  // quantized types, only a single scale and zero point is needed.
  const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
  // Dynamically allocate per-channel quantization parameters.
  data->per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  TF_LITE_ENSURE(context, data->per_channel_output_multiplier != nullptr);
  data->per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  TF_LITE_ENSURE(context, data->per_channel_output_multiplier != nullptr);

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kDepthwiseConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  const int32_t input_offset = -input->params.zero_point;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;

  auto filter_shape = GetTensorShape(filter);

  const int output_depth = SizeOfDimension(filter, 3);
  // Selection structure mirrors that in Eval.   Could select a final
  // kernel variant here...

  if (filter->type == kTfLiteInt8 || filter->type == kTfLiteUInt8) {
    const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
    const int32_t* bias_data = GetTensorData<int32_t>(bias);

    const int32_t filter_offset = -filter->params.zero_point;
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);

    void* raw = context->AllocatePersistentBuffer(
        context, sizeof(int32_t) * num_channels);
    data->sum_of_filters_factor = static_cast<int32_t*>(raw);

    // Precompute the sum of filters
    if (filter->type == kTfLiteUInt8) {
      if (filter->quantization.details.type ==
          kTfLiteSub8BitPackedUniformDetail) {
        PrecomputeSumOfPackedFiltersFactor(
            bias_data, filter, data->sum_of_filters_factor, filter_shape,
            input_offset, filter_offset,
            *filter->quantization.details.data.custom_sub8bit_packing);
      } else {
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

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height, data_type,
                                        data));

  // Determine which version to use
  bool need_acc_buf = false;
  // Check if optimized filter width is used
  const bool use_optimized_filter_width = (GetTensorShape(filter).Dims(0) != 1);
  const bool use_reference =
      ((dilation_width_factor != 1) || (dilation_height_factor != 1) ||
       use_optimized_filter_width);
  const int input_depth = GetTensorShape(input).Dims(3);
  const int needed_size =
      output_depth * filter_width * filter_height * input_depth;
  const bool use_optimized_size =
      ((filter_width == 8) && (input_offset == 0) && (input_depth == 1) &&
       (needed_size <= kReshapedFilterDataSize) && input->type == kTfLiteUInt8);
  if (!use_reference && !use_optimized_size &&
      !(input->type == kTfLiteFloat32)) {
    need_acc_buf = true;
  }
  if (need_acc_buf) {
    void* raw = context->AllocatePersistentBuffer(
        context, sizeof(int32_t) * output_depth);
    data->acc_buf = static_cast<int32_t*>(raw);
  }

  const bool use_padding =
      (data->padding.height != 0 || data->padding.width != 0 ||
       data->padding.height_offset != 0 || data->padding.width_offset != 0);
  const bool use_packed =
      (filter->quantization.details.type == kTfLiteSub8BitPackedUniformDetail);
  if (use_packed) {
    data->custom_sub8bit_packing =
        filter->quantization.details.data.custom_sub8bit_packing;
  } else {
    data->custom_sub8bit_packing = nullptr;
  }

  // Set the function pointer that is used during inference here
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalFloat);
      break;
    case kTfLiteInt8: {
      if (use_reference) {
        data->eval_function =
            TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalInt8Reference);
      } else if (use_padding) {
        // Use the version that can handle padding
        data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalInt8Padding);
      } else {
        data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalInt8);
      }
      break;
    }
    case kTfLiteUInt8: {
      if (use_packed) {
        if (use_padding) {
          data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
              DepthwiseConvPackedFilterWithPadding);
        } else {
          data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
              DepthwiseConvPackedFilterWithoutPadding);
        }
      } else if (use_reference) {
        data->eval_function =
            TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalUInt8Reference);
      } else if (use_optimized_size) {
        data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(
            DepthwiseConvOptimizedForFilterWidthEight);
      } else if (use_padding) {
        data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalUInt8Padding);
      } else {
        data->eval_function = TT_LITE_MICRO_EVAL_VARIANT_FPTR(EvalUInt8);
      }
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
#endif

  TF_LITE_MICRO_RECORD_OP_USER_DATA("depthwise_conv",
                                    static_opdata(*data, output_depth));

  return kTfLiteOk;
}

TfLiteStatus DepthwiseConvOptimizedForFilterWidthEight(
    TfLiteContext* context, const TfLiteDepthwiseConvParams& params,
    OpData* data, const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const uint8_t* filter_data = tflite::micro::GetTensorData<uint8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = data->output_activation_min;
  const int32_t output_activation_max = data->output_activation_max;
  const int32_t input_offset = -data->input_zero_point;
  const int32_t filter_offset = -data->filter_zero_point;
  const int32_t output_offset = data->output_zero_point;
  const int32_t output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
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

  static int16_t reshaped_filter_data[kReshapedFilterDataSize];
  const int needed_size =
      output_depth * filter_width * filter_height * input_depth;
  if (needed_size > kReshapedFilterDataSize) {
    TF_LITE_KERNEL_LOG(
        context,
        "Size too large for reshaped weight buffer (%d needed, %d available)",
        needed_size, kReshapedFilterDataSize);
    return kTfLiteError;
  }

  RuntimeShape reshaped_filter_shape;
  reshaped_filter_shape.BuildFrom(
      {1, output_depth, filter_height, filter_width});

  // If this is the first time through, repack the weights into a cached buffer
  // so that they can be accessed sequentially.
  static bool is_reshaped_filter_initialized = false;
  if (!is_reshaped_filter_initialized) {
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
      for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
        for (int oc = 0; oc < output_depth; ++oc) {
          const uint8_t* current_filter =
              filter_data + Offset(filter_shape, 0, filter_y, filter_x, oc);
          int16_t* reshaped_filter =
              reshaped_filter_data +
              Offset(reshaped_filter_shape, 0, oc, filter_y, filter_x);
          *reshaped_filter =
              static_cast<int16_t>(*current_filter) + filter_offset;
        }
      }
    }
    is_reshaped_filter_initialized = true;
  }

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t acc = 0;
            int in_y_start = in_y_origin;
            int filter_y_start = 0;
            if (in_y_origin < 0) {
              in_y_start = 0;
              filter_y_start = 0 - in_y_origin;
            }
            int filter_y_end = filter_height;
            if ((in_y_origin + filter_height) >= input_height) {
              filter_y_end -= (in_y_origin + filter_height) - input_height;
            }
            int in_y = in_y_start;
            int in_x_start = in_x_origin;
            int filter_x_start = 0;
            bool is_out_of_x_bounds = false;
            if (in_x_origin < 0) {
              in_x_start = 0;
              filter_x_start = 0 - in_x_origin;
              is_out_of_x_bounds = true;
            }
            int filter_x_end = filter_width;
            if ((in_x_origin + filter_width) >= input_width) {
              filter_x_end -= (in_x_origin + filter_width) - input_width;
              is_out_of_x_bounds = true;
            }
            for (int filter_y = filter_y_start; filter_y < filter_y_end;
                 ++filter_y, ++in_y) {
              const uint8_t* current_input =
                  input_data + Offset(input_shape, b, in_y, in_x_start, ic);
              if ((filter_width == 8) && !is_out_of_x_bounds) {
                int16_t* current_filter =
                    reshaped_filter_data + Offset(reshaped_filter_shape, 0, oc,
                                                  filter_y, filter_x_start);
                const uint32_t input_vals0 =
                    *reinterpret_cast<const uint32_t*>(current_input);
                current_input += 4;
                const int32_t filter_vals0 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                current_filter += 2;
                const uint8_t input_val0 = input_vals0 & 0xff;
                const int16_t filter_val0 = filter_vals0 & 0xffff;
                acc += filter_val0 * input_val0;
                const uint8_t input_val1 = (input_vals0 >> 8) & 0xff;
                const int16_t filter_val1 = (filter_vals0 >> 16) & 0xffff;
                acc += filter_val1 * input_val1;

                const int32_t filter_vals1 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                current_filter += 2;
                const uint8_t input_val2 = (input_vals0 >> 16) & 0xff;
                const int16_t filter_val2 = filter_vals1 & 0xffff;
                acc += filter_val2 * input_val2;
                const uint8_t input_val3 = (input_vals0 >> 24) & 0xff;
                const int16_t filter_val3 = (filter_vals1 >> 16) & 0xffff;
                acc += filter_val3 * input_val3;

                const uint32_t input_vals1 =
                    *reinterpret_cast<const uint32_t*>(current_input);
                const int32_t filter_vals2 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                current_filter += 2;
                const uint8_t input_val4 = input_vals1 & 0xff;
                const int16_t filter_val4 = filter_vals2 & 0xffff;
                acc += filter_val4 * input_val4;
                const uint8_t input_val5 = (input_vals1 >> 8) & 0xff;
                const int16_t filter_val5 = (filter_vals2 >> 16) & 0xffff;
                acc += filter_val5 * input_val5;

                const int32_t filter_vals3 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                const uint8_t input_val6 = (input_vals1 >> 16) & 0xff;
                const int16_t filter_val6 = filter_vals3 & 0xffff;
                acc += filter_val6 * input_val6;
                const uint8_t input_val7 = (input_vals1 >> 24) & 0xff;
                const int16_t filter_val7 = (filter_vals3 >> 16) & 0xffff;
                acc += filter_val7 * input_val7;
              } else {
                const uint8_t* current_filter =
                    filter_data +
                    Offset(filter_shape, 0, filter_y, filter_x_start, oc);
                for (int filter_x = filter_x_start; filter_x < filter_x_end;
                     ++filter_x) {
                  int32_t input_val = *current_input;
                  current_input += input_depth;
                  int32_t filter_val = *current_filter;
                  current_filter += output_depth;
                  acc +=
                      (filter_val + filter_offset) * (input_val + input_offset);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[oc];
            }
            acc = reference_ops::depthwise_conv::DepthwiseConvRound<
                DepthwiseConvOutputRounding::kAwayFromZero>(
                acc, output_multiplier, output_shift);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                static_cast<uint8_t>(acc);
          }
        }
      }
    }
  }
  return kTfLiteOk;
}

struct DepthwiseConvPackedTraits {
  struct WithPadding {
    WithPadding(const int32_t input_offset, OpData* data,
                const int32_t* bias_data)
        : input_offset_(input_offset), bias_data_(bias_data) {}

    inline void SumOfFiltersCorrectionAndBias(int32_t& raw_sum,
                                              uint32_t out_chan) const {
      raw_sum += bias_data_[out_chan];
    }

    inline int32_t OffsetInputValue(int32_t input_value) const {
      return input_offset_ + input_value;
    };

    const int32_t input_offset_;
    const int32_t* bias_data_;
  };

  struct WithoutPadding {
    WithoutPadding(const int input_offset, OpData* data,
                   const int32_t* bias_data)
        : sum_of_filters_factor_(data->sum_of_filters_factor) {}

    inline void SumOfFiltersCorrectionAndBias(int32_t& raw_sum,
                                              uint32_t out_chan) const {
      raw_sum += sum_of_filters_factor_[out_chan];
    }

    inline int32_t OffsetInputValue(int32_t input_value) const {
      return input_value;
    };

    const int32_t* sum_of_filters_factor_;
  };
};

template <typename CONTAINER_T, size_t bits_per_item,
          size_t items_per_container, class PADDING_TRAIT>
struct DepthwiseConvPacked {
  static inline void Run(const TfLiteDepthwiseConvParams& params, OpData* data,
                         const TfLiteEvalTensor* input,
                         const TfLiteEvalTensor* filter,
                         const TfLiteEvalTensor* bias,
                         TfLiteEvalTensor* output) {
    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
    const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
    const CONTAINER_T* filter_data = static_cast<const CONTAINER_T*>(
        tflite::micro::GetTensorData<void>(filter));
    const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
    const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);

    const PADDING_TRAIT pad_traits(-data->input_zero_point, data, bias_data);
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t output_activation_min = data->output_activation_min;
    const int32_t output_activation_max = data->output_activation_max;
    const int32_t filter_offset = -data->filter_zero_point;
    const int32_t output_offset = data->output_zero_point;
    const int32_t output_multiplier = data->output_multiplier;
    const int output_shift = -data->output_shift;
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

    int32_t* accbuf = data->acc_buf;

    const int* in_dims =
        reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());

    const unsigned int num_packed_containers =
        output_depth / items_per_container;
    const unsigned int elts_partial_container =
        output_depth % items_per_container;

    const int32_t mask = (1 << bits_per_item) - 1;

    for (int b = 0; b < batches; ++b) {
      const uint32_t input_offset0 = in_dims[1] * b;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;

          for (int i = 0; i < output_depth; ++i) {
            accbuf[i] = 0;
          }

          // First container...
          const CONTAINER_T* filter_vals_container_p = filter_data;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);

            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              const uint8_t* input_p =
                  &input_data[in_dims[3] * (input_offset1 + in_x)];

              // If the location is outside the bounds of the input image,
              // use zero as a default value.
              if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height)) {
                unsigned int output_channel = 0;
                // Full containers ...
                for (unsigned int container = 0;
                     container < num_packed_containers; ++container) {
                  CONTAINER_T filter_vals = *filter_vals_container_p;
                  ++filter_vals_container_p;
                  // Unrollable loop!
                  for (unsigned int element = 0; element < items_per_container;
                       ++element) {
                    int32_t offset_input_val =
                        pad_traits.OffsetInputValue(*input_p);
                    int32_t offset_filter_val =
                        (filter_vals & mask) + filter_offset;

                    filter_vals >>= bits_per_item;
                    accbuf[output_channel] +=
                        offset_filter_val * offset_input_val;
                    ++output_channel;
                    if (output_channel % depth_multiplier == 0) {
                      ++input_p;
                    }
                  }
                }

                // Might be a last, partial, container
                if (elts_partial_container) {
                  CONTAINER_T filter_vals = *filter_vals_container_p;
                  ++filter_vals_container_p;
                  for (unsigned int element = 0;
                       element < elts_partial_container; ++element) {
                    int32_t offset_input_val =
                        pad_traits.OffsetInputValue(*input_p);
                    int32_t offset_filter_val =
                        (filter_vals & mask) + filter_offset;
                    ;
                    filter_vals >>= bits_per_item;
                    accbuf[output_channel] +=
                        offset_filter_val * offset_input_val;
                    ++output_channel;
                    if (output_channel % depth_multiplier == 0) {
                      ++input_p;
                    }
                  }
                }
              } else {
                // In case of padding, we need to increment the container count
                // for next iteration
                filter_vals_container_p += num_packed_containers;
                if (elts_partial_container) {
                  filter_vals_container_p++;
                }
              }
            }
          }

          for (int oc = 0; oc < output_depth; ++oc) {
            int32_t acc = accbuf[oc];
            pad_traits.SumOfFiltersCorrectionAndBias(acc, oc);
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                                output_shift);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                static_cast<uint8_t>(acc);
          }
        }
      }
    }
  }
};

template <class PADDING_TRAITS>
inline TfLiteStatus DepthwiseConvPackedFilter(
    TfLiteContext* context, const TfLiteDepthwiseConvParams& params,
    OpData* data, const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  // We need to allocate output_depth size buffer for accumulators.
  const TfLiteCustomSub8BitPackingDetails& packing_details =
      *data->custom_sub8bit_packing;
  unsigned int bits_per_item = packing_details.bits_per_item;
  unsigned int container_bits = packing_details.container_bits;
  unsigned int packed_minor_dims = packing_details.packed_minor_dims;

  // TODO Check alignment run-length marches minor dimension.
  TFLITE_CHECK(packed_minor_dims == 1);
  switch (bits_per_item) {
    case 4: {
      TFLITE_CHECK(container_bits == 8);
      using KERNEL = DepthwiseConvPacked<uint8_t, 4, 8 / 4, PADDING_TRAITS>;
      KERNEL::Run(params, data, input, filter, bias, output);
      return kTfLiteOk;
    }
    case 5: {
      TFLITE_CHECK(container_bits == 16);
      using KERNEL = DepthwiseConvPacked<uint16_t, 5, 16 / 5, PADDING_TRAITS>;
      KERNEL::Run(params, data, input, filter, bias, output);
      return kTfLiteOk;
    }
    case 6: {
      TFLITE_CHECK(container_bits == 32);
      using KERNEL = DepthwiseConvPacked<uint32_t, 6, 32 / 6, PADDING_TRAITS>;
      KERNEL::Run(params, data, input, filter, bias, output);
      return kTfLiteOk;
    }
    default: {
      TFLITE_ABORT;
      return kTfLiteError;
    }
  }
}

TfLiteStatus DepthwiseConvPackedFilterWithPadding(
    TfLiteContext* context, const TfLiteDepthwiseConvParams& params,
    OpData* data, const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  return DepthwiseConvPackedFilter<DepthwiseConvPackedTraits::WithPadding>(
      context, params, data, input, filter, bias, output);
}

TfLiteStatus DepthwiseConvPackedFilterWithoutPadding(
    TfLiteContext* context, const TfLiteDepthwiseConvParams& params,
    OpData* data, const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  return DepthwiseConvPackedFilter<DepthwiseConvPackedTraits::WithoutPadding>(
      context, params, data, input, filter, bias, output);
}

TfLiteStatus EvalUInt8Padding(TfLiteContext* context,
                              const TfLiteDepthwiseConvParams& params,
                              OpData* data, const TfLiteEvalTensor* input,
                              const TfLiteEvalTensor* filter,
                              const TfLiteEvalTensor* bias,
                              TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const uint8_t* filter_data = tflite::micro::GetTensorData<uint8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = data->output_activation_min;
  const int32_t output_activation_max = data->output_activation_max;
  const int32_t input_offset = -data->input_zero_point;
  const int32_t filter_offset = -data->filter_zero_point;
  const int32_t output_offset = data->output_zero_point;
  const int32_t output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
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

  int32_t* acc_buf = data->acc_buf;
  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      const int32_t ker_y_start = MAX(0, -in_y_origin);
      const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int32_t ker_x_start = MAX(0, -in_x_origin);
        const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc_buf[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int output_channel = m + in_channel * depth_multiplier;
                int32_t input_val = input_data[input_offset2 + in_channel];
                int32_t filter_val =
                    filter_data[filter_offset2 + output_channel];
                acc_buf[output_channel] +=
                    (input_val + input_offset) * (filter_val + filter_offset);
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          if (bias_data) {
            acc_buf[i] += bias_data[i];
          }
          acc_buf[i] = MultiplyByQuantizedMultiplier(
              acc_buf[i], output_multiplier, output_shift);
          acc_buf[i] += output_offset;
          acc_buf[i] = std::max(acc_buf[i], output_activation_min);
          acc_buf[i] = std::min(acc_buf[i], output_activation_max);
          output_data[out_base + i] = static_cast<uint8_t>(acc_buf[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalUInt8(TfLiteContext* context,
                       const TfLiteDepthwiseConvParams& params, OpData* data,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const uint8_t* filter_data = tflite::micro::GetTensorData<uint8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = data->output_activation_min;
  const int32_t output_activation_max = data->output_activation_max;
  const int32_t filter_offset = -data->filter_zero_point;
  const int32_t output_offset = data->output_zero_point;
  const int32_t output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
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

  int32_t* acc_buf = data->acc_buf;
  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height);
      const int32_t ker_y_start = MAX(0, -in_y_origin);
      const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width);
        const int32_t ker_x_start = MAX(0, -in_x_origin);
        const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc_buf[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int output_channel = m + in_channel * depth_multiplier;
                int32_t input_val = input_data[input_offset2 + in_channel];
                int32_t filter_val =
                    filter_data[filter_offset2 + output_channel];
                acc_buf[output_channel] +=
                    input_val * (filter_val + filter_offset);
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          acc_buf[i] += data->sum_of_filters_factor[i];
          acc_buf[i] = MultiplyByQuantizedMultiplier(
              acc_buf[i], output_multiplier, output_shift);
          acc_buf[i] += output_offset;
          acc_buf[i] = std::max(acc_buf[i], output_activation_min);
          acc_buf[i] = std::min(acc_buf[i], output_activation_max);
          output_data[out_base + i] = static_cast<uint8_t>(acc_buf[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context,
                       const TfLiteDepthwiseConvParams& params, OpData* data,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params.activation, &output_activation_min,
                           &output_activation_max);

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params.stride_width;
  op_params.stride_height = params.stride_height;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.dilation_height_factor = params.dilation_height_factor;
  op_params.depth_multiplier = params.depth_multiplier;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.float_activation_max = output_activation_max;
  op_params.float_activation_min = output_activation_min;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = -data->filter_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are
  // +ve-means-left.
  op_params.output_shift = -data->output_shift;

  tflite::reference_ops::DepthwiseConv(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<float>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<float>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalInt8Padding(TfLiteContext* context,
                             const TfLiteDepthwiseConvParams& params,
                             OpData* data, const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* filter,
                             const TfLiteEvalTensor* bias,
                             TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
  const int32_t* output_multiplier = data->per_channel_output_multiplier;
  const int32_t* output_shift = data->per_channel_output_shift;

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);

  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t input_offset = -data->input_zero_point;
  const int32_t output_offset = data->output_zero_point;
  const int32_t output_activation_min = std::numeric_limits<int8_t>::min();
  const int32_t output_activation_max = std::numeric_limits<int8_t>::max();

  // Check dimensions of the tensors.
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

  int32_t* acc_buf = data->acc_buf;
  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      const int32_t ker_y_start = MAX(0, -in_y_origin);
      const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int32_t ker_x_start = MAX(0, -in_x_origin);
        const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc_buf[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int out_channel = m + in_channel * depth_multiplier;
                int32_t input_val = input_data[input_offset2 + in_channel];
                int32_t filter_val = filter_data[filter_offset2 + out_channel];
                acc_buf[out_channel] += (input_val + input_offset) * filter_val;
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          if (bias) {
            acc_buf[i] += bias_data[i];
          }
          acc_buf[i] = MultiplyByQuantizedMultiplier(
              acc_buf[i], output_multiplier[i], output_shift[i]);
          acc_buf[i] += output_offset;
          acc_buf[i] = std::max(acc_buf[i], output_activation_min);
          acc_buf[i] = std::min(acc_buf[i], output_activation_max);
          output_data[out_base + i] = static_cast<int8_t>(acc_buf[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalInt8(TfLiteContext* context,
                      const TfLiteDepthwiseConvParams& params, OpData* data,
                      const TfLiteEvalTensor* input,
                      const TfLiteEvalTensor* filter,
                      const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  const int32_t* output_multiplier = data->per_channel_output_multiplier;
  const int32_t* output_shift = data->per_channel_output_shift;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);

  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_offset = data->output_zero_point;
  const int32_t output_activation_min = std::numeric_limits<int8_t>::min();
  const int32_t output_activation_max = std::numeric_limits<int8_t>::max();

  // Check dimensions of the tensors.
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

  int32_t* acc_buf = data->acc_buf;
  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height);
      const int32_t ker_y_start = MAX(0, -in_y_origin);
      const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width);
        const int32_t ker_x_start = MAX(0, -in_x_origin);
        const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc_buf[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int out_channel = m + in_channel * depth_multiplier;
                int32_t input_val = input_data[input_offset2 + in_channel];
                int32_t filter_val = filter_data[filter_offset2 + out_channel];
                acc_buf[out_channel] += input_val * filter_val;
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          acc_buf[i] += data->sum_of_filters_factor[i];
          acc_buf[i] = MultiplyByQuantizedMultiplier(
              acc_buf[i], output_multiplier[i], output_shift[i]);
          acc_buf[i] += output_offset;
          acc_buf[i] = std::max(acc_buf[i], output_activation_min);
          acc_buf[i] = std::min(acc_buf[i], output_activation_max);
          output_data[out_base + i] = static_cast<int8_t>(acc_buf[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalInt8Reference(TfLiteContext* context,
                               const TfLiteDepthwiseConvParams& params,
                               OpData* data, const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output) {
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params.stride_width;
  op_params.stride_height = params.stride_height;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.dilation_height_factor = params.dilation_height_factor;
  op_params.depth_multiplier = params.depth_multiplier;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = data->output_zero_point;
  op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  reference_integer_ops::DepthwiseConvPerChannel(
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

TfLiteStatus EvalUInt8Reference(TfLiteContext* context,
                                const TfLiteDepthwiseConvParams& params,
                                OpData* data, const TfLiteEvalTensor* input,
                                const TfLiteEvalTensor* filter,
                                const TfLiteEvalTensor* bias,
                                TfLiteEvalTensor* output) {
  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params.stride_width;
  op_params.stride_height = params.stride_height;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.dilation_height_factor = params.dilation_height_factor;
  op_params.depth_multiplier = params.depth_multiplier;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.input_offset = -data->input_zero_point;
  op_params.weights_offset = -data->filter_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are
  // +ve-means-left.
  op_params.output_shift = -data->output_shift;

  tflite::reference_ops::DepthwiseConv(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<uint8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<uint8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<uint8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteDepthwiseConvParams& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;

  return data->eval_function(context, params, data, input, filter, bias,
                             output);
}

}  // namespace depthwise_conv

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
  return {/*init=*/depthwise_conv::Init,
          /*free=*/nullptr,
          /*prepare=*/depthwise_conv::Prepare,
          /*invoke=*/depthwise_conv::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
