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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/portable_optimized/pooling_op_data.h"
#include "tensorflow/lite/micro/kernels/static_data_utils.h"
#include "tensorflow/lite/micro/kernels/static_init_support.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pooling {

TFLM_COLLECT_KERNEL_INFO(
    "pooling", "struct OpData;\n",
    "#include "
    "\"tensorflow/lite/micro/kernels/portable_optimized/pooling_op_data.h\"",
    "OpData",
    "    const TfLiteContext* context, const TfLiteNode* node,\n"
    "    const TfLitePoolParams* params, const OpData* data, \n"
    "    const TfLiteEvalTensor* input, TfLiteEvalTensor* output");

#if TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT

static CppItems* static_opdata(OpData& od) {
  auto init = new CppItems();

  *init << TfLitePaddingValuesSubStruct(od.padding) << od.activation_min
        << od.activation_max << od.activation_min_f32 << od.activation_max_f32
        << od.eval_function;

  return init;
}
#endif

enum class PoolingType { Max, Average };

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus CalculateOpData(const TfLiteContext* context,
                             const TfLitePoolParams* params,
                             const TfLiteTensor* input,
                             const TfLiteTensor* output, OpData* data) {
  // input: batch, height, width, channel
  int height = SizeOfDimension(input, 1);
  int width = SizeOfDimension(input, 2);

  int out_height, out_width;

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      /*dilation_rate_height=*/1,
      /*dilation_rate_width=*/1, height, width, params->filter_height,
      params->filter_width, params->padding, &out_height, &out_width);

  return kTfLiteOk;
}

void AverageEvalFloat(const TfLiteContext* context, const TfLiteNode* node,
                      const TfLitePoolParams* params, const OpData* data,
                      const TfLiteEvalTensor* input, TfLiteEvalTensor* output) {
  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = data->activation_min_f32;
  op_params.float_activation_max = data->activation_max_f32;
  reference_ops::AveragePool(op_params, tflite::micro::GetTensorShape(input),
                             tflite::micro::GetTensorData<float>(input),
                             tflite::micro::GetTensorShape(output),
                             tflite::micro::GetTensorData<float>(output));
}

void AverageEvalQuantizedInt8(const TfLiteContext* context,
                              const TfLiteNode* node,
                              const TfLitePoolParams* params,
                              const OpData* data, const TfLiteEvalTensor* input,
                              TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteInt8);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  reference_integer_ops::AveragePool(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
}

void AverageEvalQuantizedUInt8(const TfLiteContext* context,
                               const TfLiteNode* node,
                               const TfLitePoolParams* params,
                               const OpData* data,
                               const TfLiteEvalTensor* input,
                               TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteUInt8);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  reference_ops::AveragePool(op_params, tflite::micro::GetTensorShape(input),
                             tflite::micro::GetTensorData<uint8_t>(input),
                             tflite::micro::GetTensorShape(output),
                             tflite::micro::GetTensorData<uint8_t>(output));
}

void MaxEvalFloat(const TfLiteContext* context, const TfLiteNode* node,
                  const TfLitePoolParams* params, const OpData* data,
                  const TfLiteEvalTensor* input, TfLiteEvalTensor* output) {
  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = data->activation_min_f32;
  op_params.float_activation_max = data->activation_max_f32;
  reference_ops::MaxPool(op_params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<float>(input),
                         tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<float>(output));
}

void MaxEvalQuantizedInt8(const TfLiteContext* context, const TfLiteNode* node,
                          const TfLitePoolParams* params, const OpData* data,
                          const TfLiteEvalTensor* input,
                          TfLiteEvalTensor* output) {
  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  reference_integer_ops::MaxPool(op_params,
                                 tflite::micro::GetTensorShape(input),
                                 tflite::micro::GetTensorData<int8_t>(input),
                                 tflite::micro::GetTensorShape(output),
                                 tflite::micro::GetTensorData<int8_t>(output));
}

void MaxEvalQuantizedUInt8(const TfLiteContext* context, const TfLiteNode* node,
                           const TfLitePoolParams* params, const OpData* data,
                           const TfLiteEvalTensor* input,
                           TfLiteEvalTensor* output) {
  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  reference_ops::MaxPool(op_params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<uint8_t>(input),
                         tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<uint8_t>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  data->eval_function(context, node, params, data, input, output);
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
#if TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  OpData* recordedStaticOpData();
  return recordedStaticOpData();
#else
  auto raw = context->AllocatePersistentBuffer(context, sizeof(OpData));
  TFLITE_DCHECK(raw != nullptr);
  OpData* data = reinterpret_cast<OpData*>(raw);
	*data = {};
  return raw;
#endif
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node,
                     PoolingType pooling_type) {

#if !TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, data));

  if (input->type == kTfLiteFloat32) {
    CalculateActivationRange(params->activation, &data->activation_min_f32,
                             &data->activation_max_f32);
  } else if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    CalculateActivationRangeQuantized(context, params->activation, output,
                                      &data->activation_min,
                                      &data->activation_max);
  }

  switch (pooling_type) {
    case PoolingType::Average:
      // Inputs and outputs share the same type, guaranteed by the converter.
      switch (input->type) {
        case kTfLiteFloat32:
          data->eval_function =
              TFLM_SET_KERNEL_VARIANT(AverageEvalFloat);
          break;
        case kTfLiteUInt8:
          data->eval_function =
              TFLM_SET_KERNEL_VARIANT(AverageEvalQuantizedUInt8);
          break;
        case kTfLiteInt8:
          data->eval_function =
              TFLM_SET_KERNEL_VARIANT(AverageEvalQuantizedInt8);
          break;
        default:
          TF_LITE_KERNEL_LOG(context,
                             "Input type %s is not currently supported",
                             TfLiteTypeGetName(input->type));
          return kTfLiteError;
      }
      break;

    case PoolingType::Max:
      switch (input->type) {
        case kTfLiteFloat32:
          data->eval_function = TFLM_SET_KERNEL_VARIANT(MaxEvalFloat);
          break;
        case kTfLiteUInt8:
          data->eval_function =
              TFLM_SET_KERNEL_VARIANT(MaxEvalQuantizedUInt8);
          break;
        case kTfLiteInt8:
          data->eval_function =
              TFLM_SET_KERNEL_VARIANT(MaxEvalQuantizedInt8);
          break;
        default:
          TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                             TfLiteTypeGetName(input->type));
          return kTfLiteError;
      }
  }
#endif

  TFLM_RECORD_OP_USER_DATA("pooling", static_opdata(*data));

  return kTfLiteOk;
}

TfLiteStatus PrepareAverage(TfLiteContext* context, TfLiteNode* node) {
  return Prepare(context, node, PoolingType::Average);
}

TfLiteStatus PrepareMax(TfLiteContext* context, TfLiteNode* node) {
  return Prepare(context, node, PoolingType::Max);
}

}  // namespace pooling

TfLiteRegistration Register_AVERAGE_POOL_2D() {
  return {/*init=*/pooling::Init,
          /*free=*/nullptr,
          /*prepare=*/pooling::PrepareAverage,
          /*invoke=*/pooling::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_MAX_POOL_2D() {
  return {/*init=*/pooling::Init,
          /*free=*/nullptr,
          /*prepare=*/pooling::PrepareMax,
          /*invoke=*/pooling::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
