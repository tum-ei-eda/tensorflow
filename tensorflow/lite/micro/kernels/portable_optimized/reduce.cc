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

#include "tensorflow/lite/kernels/internal/reference/reduce.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pointer_collector.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace reduce {

constexpr int kMaxNumberOfAxis = 4;
constexpr int kMaxNumberOfReducedAxis = 2;

struct OpData;

typedef bool (*OpEvalHandler)(TfLiteContext* context, OpData* op_data,
                              TfLiteReducerParams* params,
                              const TfLiteEvalTensor* input,
                              const TfLiteEvalTensor* axis,
                              TfLiteEvalTensor* output);
struct OpData {
  int32_t multiplier;
  int shift;
  int temp_buffer_idx;
  int input_zp;
  float input_scale;
  int output_zp;
  float output_scale;
  OpEvalHandler eval_function;
};

KERNEL_VARIANT_COLLECT_INFO(
    "reduce", "struct OpData;\n",
    "",
    "OpData",
    "    TfLiteContext* context,\n"
    "    OpData* op_data, TfLiteReducerParams* params,\n"
    "    const TfLiteEvalTensor* input, const TfLiteEvalTensor* axis,\n"
    "    TfLiteEvalTensor* output\n");

void* InitMean(TfLiteContext* context, const char* buffer, size_t length) {
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}


void ResolveAxis(const int* axis_data, int axis_count,
                 tflite::MeanParams* op_params) {
  int i = 0;
  for (; i < axis_count; ++i) {
    op_params->axis[i] = static_cast<int16_t>(axis_data[i]);
  }
  for (; i < 4; ++i) {
    op_params->axis[i] = 1;
  }
  op_params->axis_count = axis_count;
}

bool ReduceFloatKeepDims(TfLiteContext* context, OpData* op_data,
                         TfLiteReducerParams* params,
                         const TfLiteEvalTensor* input,
                         const TfLiteEvalTensor* axis,
                         TfLiteEvalTensor* output) {
  tflite::MeanParams op_params;
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  ResolveAxis(tflite::micro::GetTensorData<int>(axis), num_axis, &op_params);
  reference_ops::Mean(op_params, tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<float>(input),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<float>(output));
  return true;
}

bool ReduceFloatChangeDims(TfLiteContext* context, OpData* op_data,
                           TfLiteReducerParams* params,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* axis,
                           TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  return reference_ops::Mean(
      tflite::micro::GetTensorData<float>(input), input->dims->data,
      input->dims->size, tflite::micro::GetTensorData<float>(output),
      output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, tflite::micro::GetTensorData<float>(output));
}

bool ReduceInt8KeepDims(TfLiteContext* context, OpData* op_data,
                        TfLiteReducerParams* params,
                        const TfLiteEvalTensor* input,
                        const TfLiteEvalTensor* axis,
                        TfLiteEvalTensor* output) {
  tflite::MeanParams op_params;
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  ResolveAxis(tflite::micro::GetTensorData<int>(axis), num_axis, &op_params);
  reference_integer_ops::Mean(
      op_params, op_data->multiplier, op_data->shift,
      tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input), op_data->input_zp,
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output), op_data->output_zp);
  return true;
}

bool ReduceInt8ChangeDims(TfLiteContext* context, OpData* op_data,
                          TfLiteReducerParams* params,
                          const TfLiteEvalTensor* input,
                          const TfLiteEvalTensor* axis,
                          TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  int32_t* temp_buffer = static_cast<int32_t*>(
      context->GetScratchBuffer(context, op_data->temp_buffer_idx));
  return reference_ops::Mean(
      tflite::micro::GetTensorData<int8_t>(input), input->dims->data,
      input->dims->size, tflite::micro::GetTensorData<int8_t>(output),
      output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, temp_buffer);
}

bool ReduceInt8ChangeDimsAndQuant(TfLiteContext* context, OpData* op_data,
                                  TfLiteReducerParams* params,
                                  const TfLiteEvalTensor* input,
                                  const TfLiteEvalTensor* axis,
                                  TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  int32_t* temp_buffer = static_cast<int32_t*>(
      context->GetScratchBuffer(context, op_data->temp_buffer_idx));
  return reference_ops::QuantizedMeanOrSum(
      tflite::micro::GetTensorData<int8_t>(input), op_data->input_zp,
      op_data->input_scale, input->dims->data, input->dims->size,
      tflite::micro::GetTensorData<int8_t>(output), op_data->output_zp,
      op_data->output_scale, output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, temp_buffer, false);
}

bool ReduceUInt8KeepDims(TfLiteContext* context, OpData* op_data,
                         TfLiteReducerParams* params,
                         const TfLiteEvalTensor* input,
                         const TfLiteEvalTensor* axis,
                         TfLiteEvalTensor* output) {
  tflite::MeanParams op_params;
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  ResolveAxis(tflite::micro::GetTensorData<int>(axis), num_axis, &op_params);
  reference_ops::Mean(op_params, tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<uint8_t>(input),
                      op_data->input_zp, op_data->input_scale,
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<uint8_t>(output),
                      op_data->output_zp, op_data->output_scale);
  return true;
}

bool ReduceUInt8ChangeDims(TfLiteContext* context, OpData* op_data,
                           TfLiteReducerParams* params,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* axis,
                           TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  int32_t* temp_buffer = static_cast<int32_t*>(
      context->GetScratchBuffer(context, op_data->temp_buffer_idx));
  return reference_ops::Mean(
      tflite::micro::GetTensorData<uint8_t>(input), input->dims->data,
      input->dims->size, tflite::micro::GetTensorData<uint8_t>(output),
      output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, temp_buffer);
}

bool ReduceUInt8ChangeDimsAndQuant(TfLiteContext* context, OpData* op_data,
                                   TfLiteReducerParams* params,
                                   const TfLiteEvalTensor* input,
                                   const TfLiteEvalTensor* axis,
                                   TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  int32_t* temp_buffer = static_cast<int32_t*>(
      context->GetScratchBuffer(context, op_data->temp_buffer_idx));
  return reference_ops::QuantizedMeanOrSum(
      tflite::micro::GetTensorData<uint8_t>(input), op_data->input_zp,
      op_data->input_scale, input->dims->data, input->dims->size,
      tflite::micro::GetTensorData<uint8_t>(output), op_data->output_zp,
      op_data->output_scale, output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, temp_buffer, false);
}

TfLiteStatus PrepareSimple(TfLiteContext* context, TfLiteNode* node) {
  // Inputs Tensor (dtype depends on quantization):
  // [0] = Input
  // [1] = Axis

  // Outputs Tensor (dtype depends on quantization):
  // [0] = Output

  // Validate number of inputs and outputs
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Validate axis type
  const TfLiteTensor* axis = GetInput(context, node, 1);
  TF_LITE_ENSURE_TYPES_EQ(context, axis->type, kTfLiteInt32);
  return kTfLiteOk;
}

TfLiteStatus PrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* output = GetOutput(context, node, 0);
  if (input->type == kTfLiteInt8) {
    const double real_multiplier = static_cast<double>(input->params.scale) /
                                   static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, &op_data->multiplier, &op_data->shift);
  }

  int output_size = NumElements(output);
  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    context->RequestScratchBufferInArena(context, output_size * sizeof(int32_t),
                                         &op_data->temp_buffer_idx);
    op_data->input_zp = input->params.zero_point;
    op_data->input_scale = input->params.scale;
    op_data->output_zp = output->params.zero_point;
    op_data->output_scale = output->params.scale;
  }

  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node));
  // TODO(b/144955155): Support uint8_t(b/144955155) and int8_t(b/144955018)

  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  int axis_count = static_cast<int>(ElementCount(*axis->dims));
  auto axis_data = tflite::micro::GetTensorData<int>(axis);
  // TODO(b/146571391): Support only 4D Input and 2D Axis for Mean until
  // scratch tensor allocation has been implemented in (b/132070898)
  bool is_valid_inputs = (input->dims->size == 4 && axis_count == 2 &&
                          ((axis_data[0] == 1 && axis_data[1] == 2) ||
                           (axis_data[0] == 2 && axis_data[1] == 1)));
  TfLiteReducerParams* params =
      reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);
  TF_LITE_ENSURE_MSG(
      context, is_valid_inputs == true,
      "Number of Input "
      "dimensions != 4 OR the Axis is not either [1, 2] or [2, 1]");
  switch (input->type) {
    case kTfLiteFloat32: {
      // TODO(b/139102329): Handle the below special case in the combined
      // reference method.
      // Defer to specialized implementation for 4D Mean across axes 1 & 2.
      if (params->keep_dims) {
        op_data->eval_function =
            TLITE_MICRO_SELECTED_KERNEL_VARIANT(ReduceFloatKeepDims);
      } else {
        op_data->eval_function =
            TLITE_MICRO_SELECTED_KERNEL_VARIANT(ReduceFloatChangeDims);
      }
    } break;
    case kTfLiteInt8: {
      if (params->keep_dims) {
        op_data->eval_function =
            TLITE_MICRO_SELECTED_KERNEL_VARIANT(ReduceInt8KeepDims);
      } else if (op_data->input_zp == op_data->output_zp &&
                 op_data->input_scale == op_data->output_scale) {
        op_data->eval_function =
            TLITE_MICRO_SELECTED_KERNEL_VARIANT(ReduceInt8ChangeDims);
      } else {
        op_data->eval_function =
            TLITE_MICRO_SELECTED_KERNEL_VARIANT(ReduceInt8ChangeDimsAndQuant);
      }
    } break;
    case kTfLiteUInt8: {
      if (params->keep_dims) {
        op_data->eval_function =
            TLITE_MICRO_SELECTED_KERNEL_VARIANT(ReduceUInt8KeepDims);
      } else if (op_data->input_zp == op_data->output_zp &&
                 op_data->input_scale == op_data->output_scale) {
        op_data->eval_function =
            TLITE_MICRO_SELECTED_KERNEL_VARIANT(ReduceUInt8ChangeDims);
      } else {
        op_data->eval_function =
            TLITE_MICRO_SELECTED_KERNEL_VARIANT(ReduceInt8ChangeDimsAndQuant);
      }
    } break;
    default:
      // TODO(b/144955155): Support uint8_t(b/144955155) and int8_t(b/144955018)
      TF_LITE_ENSURE_MSG(context, false,
                         "Currently, only float32, int8 or uint8 input type "
                         "is supported.");
  }
  return kTfLiteOk;
}


TfLiteStatus EvalMean(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TfLiteReducerParams* params =
      reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  auto eval_ok =
      op_data->eval_function(context, op_data, params, input, axis, output);
  TF_LITE_ENSURE_MSG(context, eval_ok, "Evaluation failed.");
  return kTfLiteOk;
}
}  // namespace reduce

TfLiteRegistration Register_MEAN() {
  return {/*init=*/reduce::InitMean,
          /*free=*/nullptr,
          /*prepare=*/reduce::PrepareMeanOrSum,
          /*invoke=*/reduce::EvalMean,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite
