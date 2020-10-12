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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/kernels/portable_optimized/reduce_op_data.h"
#include "tensorflow/lite/micro/kernels/static_init_support.h"

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

namespace tflite {
namespace ops {
namespace micro {
namespace reduce {

TFLM_COLLECT_KERNEL_INFO(
    "reduce",
    "struct OpData;\n",
    "#include \"tensorflow/lite/micro/kernels/portable_optimized/reduce_op_data.h\"",
    "OpData",
    "    TfLiteContext* context,\n"
    "    OpData* op_data, TfLiteReducerParams* params,\n"
    "    const TfLiteEvalTensor* input, const TfLiteEvalTensor* axis,\n"
    "    TfLiteEvalTensor* output\n"
);

typedef TfLiteStatus (*EvalVariantFptr)(
    TfLiteContext* context, OpData* op_data,
    TfLiteReducerParams* params, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* axis, TfLiteEvalTensor* output);

#if TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT

static CppItems *static_opdata(OpData &od, size_t buf_size)
{
  auto init = new CppItems();

  CppNamedVec<int32_t> tb("temp_buffer", "int32_t", od.temp_buffer, buf_size);

  *init << od.multiplier
       << od.shift
       << od.temp_buffer_idx
       << tb
       << od.input_zp
       << od.mean_multiplier
       << od.mean_shift
       << od.output_zp
       << od.eval_function;

  return init;
}
#endif

constexpr int kMaxNumberOfAxis = 4;
constexpr int kMaxNumberOfReducedAxis = 2;

#if TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  EvalVariantFptr recordedVariant();
  OpData *recordedStaticOpData();
#endif

void* InitMean(TfLiteContext* context, const char* buffer, size_t length) {
#if TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
  return recordedStaticOpData();
#else
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
#endif
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

// This method parses the input 'axis' to remove duplicates and handle negative
// values, and returns a valid 'out_axis'
inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* out_axis,
                        int* out_num_axis) {
  *out_num_axis = 0;  // Just in case.
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0) {
    return true;
  }
  // o(n^2) is fine since out_num_axis should be really small, mostly <= 4
  for (int64_t idx = 0; idx < num_axis; ++idx) {
    // Handle negative index. A positive index 'p_idx' can be represented as a
    // negative index 'n_idx' as: n_idx = p_idx-num_dims
    // eg: For num_dims=3, [0, 1, 2] is the same as [-3, -2, -1]  */
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    TFLITE_DCHECK(current >= 0 && current < num_dims);
    bool is_dup = false;
    for (int j = 0; j < *out_num_axis; ++j) {
      if (out_axis[j] == current) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup) {
      out_axis[*out_num_axis] = current;
      *out_num_axis += 1;
    }
  }
  return true;
}

template <typename In, typename Out>
inline bool Reduce(const In* input_data, const int* input_dims,
                   const int* output_dims, const int input_num_dims,
                   const int output_num_dims, const int* axis,
                   const int num_axis, int* input_iter,
                   Out reducer(const Out current, const In in),
                   Out* output_data) {
  // Reset input iterator.
  for (int idx = 0; idx < input_num_dims; ++idx) {
    input_iter[idx] = 0;
  }
  // Iterate through input_data.
  do {
    size_t input_offset =
        ReducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset = ReducedOutputOffset(input_num_dims, input_dims,
                                               input_iter, num_axis, axis);
    output_data[output_offset] =
        reducer(output_data[output_offset], input_data[input_offset]);
  } while (NextIndex(input_num_dims, input_dims, input_iter));
  return true;
}

// This method expects that output_data has been initialized.
template <typename In, typename Out>
inline bool ReduceSumImpl(const In* input_data, const int* input_dims,
                          const int* output_dims, const int input_num_dims,
                          const int output_num_dims, const int* axis,
                          const int num_axis, int* input_iter,
                          Out* output_data) {
  auto reducer = [](const Out current, const In in) -> Out {
    const Out actual_in = static_cast<Out>(in);
    return current + actual_in;
  };
  return Reduce<In, Out>(input_data, input_dims, output_dims, input_num_dims,
                         output_num_dims, axis, num_axis, input_iter, reducer,
                         output_data);
}

template <typename T>
inline void Mean(const tflite::MeanParams& op_params, OpData* op_data,
                 const RuntimeShape& unextended_input_shape,
                 const T* input_data,
                 const RuntimeShape& unextended_output_shape, T* output_data) {

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      int32_t value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          value += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          MultiplyByQuantizedMultiplier(value, op_data->multiplier, op_data->shift);
    }
  }
}

template<>
inline void Mean<float>(const tflite::MeanParams& op_params, OpData* op_data,
                 const RuntimeShape& unextended_input_shape,
                 const float* input_data,
                 const RuntimeShape& unextended_output_shape, float* output_data) {

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      float value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          value += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          value / (input_width * input_height);
    }
  }
}

template <typename T, typename U>
inline bool Mean(const T* input_data, const int* input_dims,
                 const int input_num_dims, T* output_data,
                 const int* output_dims, const int output_num_dims,
                 const int* axis, const int num_axis_dimensions, bool keep_dims,
                 int* temp_index, int* resolved_axis, U* temp_sum) {
  // Reset output data.
  size_t num_outputs = 1;
  for (int idx = 0; idx < output_num_dims; ++idx) {
    size_t current = static_cast<size_t>(output_dims[idx]);
    // Overflow prevention.
    if (num_outputs > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_outputs *= current;
  }
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    output_data[idx] = T();
    temp_sum[idx] = U();
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  if (!ReduceSumImpl<T, U>(input_data, input_dims, output_dims, input_num_dims,
                           output_num_dims, resolved_axis, num_resolved_axis,
                           temp_index, temp_sum)) {
    return false;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  size_t num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx) {
    size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
    // Overflow prevention.
    if (current > (std::numeric_limits<size_t>::max() / num_elements_in_axis)) {
      return false;
    }
    num_elements_in_axis *= current;
  }

  if (num_elements_in_axis > 0) {
    for (size_t idx = 0; idx < num_outputs; ++idx) {
      output_data[idx] =
          static_cast<T>(temp_sum[idx] / static_cast<U>(num_elements_in_axis));
    }
  }
  return true;
}

inline void MeanUInt8(const tflite::MeanParams& op_params, OpData* op_data,
                 const RuntimeShape& unextended_input_shape,
                 const uint8_t* input_data,  const RuntimeShape& unextended_output_shape,
                 uint8_t* output_data) {

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  int32_t input_zp = op_data->input_zp;
  int32_t output_zp = op_data->output_zp;

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  constexpr int32_t kMinValue = std::numeric_limits<uint8_t>::min();
  constexpr int32_t kMaxValue = std::numeric_limits<uint8_t>::max();

  int32_t bias =
      output_zp -
      MultiplyByQuantizedMultiplier(input_zp, op_data->multiplier, op_data->shift);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      int32_t acc = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          acc += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }
      acc = MultiplyByQuantizedMultiplier(acc, op_data->mean_multiplier, op_data->mean_shift);
      acc += bias;
      acc = std::min(std::max(acc, kMinValue), kMaxValue);
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          static_cast<uint8_t>(acc);
    }
  }
}

// Computes the mean of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis for quantized values.
template <typename T, typename U>
inline bool QuantizedMeanOrSum(OpData* op_data, const T* input_data,  const int* input_dims,
                               const int input_num_dims, T* output_data,
                               const int* output_dims,
                               const int output_num_dims, const int* axis,
                               const int num_axis_dimensions, bool keep_dims,
                               int* temp_index, int* resolved_axis, U* temp_sum,
                               bool compute_sum) {

  int32_t input_zp = op_data->input_zp;
  int32_t output_zp = op_data->output_zp;
  // Reset output data.
  size_t num_outputs = 1;
  for (int idx = 0; idx < output_num_dims; ++idx) {
    size_t current = static_cast<size_t>(output_dims[idx]);
    // Overflow prevention.
    if (num_outputs > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_outputs *= current;
  }
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    output_data[idx] = T();
    temp_sum[idx] = U();
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  if (!ReduceSumImpl<T, U>(input_data, input_dims, output_dims, input_num_dims,
                           output_num_dims, resolved_axis, num_resolved_axis,
                           temp_index, temp_sum)) {
    return false;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  size_t num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx) {
    size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
    // Overflow prevention.
    if (current > (std::numeric_limits<size_t>::max() / num_elements_in_axis)) {
      return false;
    }
    num_elements_in_axis *= current;
  }

  if (num_elements_in_axis > 0) {

    if (compute_sum) {
      // TODO(b/116341117): Eliminate float and do this completely in 8bit.
      const int32_t bias = MultiplyByQuantizedMultiplier(
          static_cast<int32_t>(-input_zp*num_elements_in_axis), op_data->multiplier, op_data->shift);
      for (size_t idx = 0; idx < num_outputs; ++idx) {
        const int32_t value =
            static_cast<int32_t>(MultiplyByQuantizedMultiplier(temp_sum[idx] , op_data->multiplier, op_data->shift) + bias) +
            output_zp;
        output_data[idx] = static_cast<T>(value);
      }
    } else {
      const int32_t bias = MultiplyByQuantizedMultiplier(
                static_cast<int32_t>(-input_zp), op_data->multiplier, op_data->shift);
      for (size_t idx = 0; idx < num_outputs; ++idx) {
        int32_t result = MIN((MultiplyByQuantizedMultiplier(temp_sum[idx],op_data->mean_multiplier, op_data->mean_shift) + bias) + output_zp,
            std::numeric_limits<T>::max());
        result = MAX(result, std::numeric_limits<T>::min());
        output_data[idx] = static_cast<T>(result);
      }
    }
  }
  return true;
}

TfLiteStatus ReduceFloatKeepDims(TfLiteContext* context, OpData* op_data,
                         TfLiteReducerParams* params,
                         const TfLiteEvalTensor* input,
                         const TfLiteEvalTensor* axis,
                         TfLiteEvalTensor* output) {
  tflite::MeanParams op_params;
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  ResolveAxis(tflite::micro::GetTensorData<int>(axis), num_axis, &op_params);
  Mean(op_params, op_data, tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<float>(input),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus ReduceFloatChangeDims(TfLiteContext* context, OpData* op_data,
                           TfLiteReducerParams* params,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* axis,
                           TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  bool return_status = Mean(
      tflite::micro::GetTensorData<float>(input), input->dims->data,
      input->dims->size, tflite::micro::GetTensorData<float>(output),
      output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, tflite::micro::GetTensorData<float>(output));
  return return_status == true ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ReduceInt8KeepDims(TfLiteContext* context, OpData* op_data,
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
  return kTfLiteOk;
}

TfLiteStatus ReduceInt8ChangeDims(TfLiteContext* context, OpData* op_data,
                          TfLiteReducerParams* params,
                          const TfLiteEvalTensor* input,
                          const TfLiteEvalTensor* axis,
                          TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  int32_t* temp_buffer = op_data->temp_buffer;
  bool return_status = Mean(
      tflite::micro::GetTensorData<int8_t>(input), input->dims->data,
      input->dims->size, tflite::micro::GetTensorData<int8_t>(output),
      output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, temp_buffer);
  return return_status == true ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ReduceInt8ChangeDimsAndQuant(TfLiteContext* context, OpData* op_data,
                                  TfLiteReducerParams* params,
                                  const TfLiteEvalTensor* input,
                                  const TfLiteEvalTensor* axis,
                                  TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  int32_t* temp_buffer = op_data->temp_buffer;
  bool return_status = QuantizedMeanOrSum(
      op_data, tflite::micro::GetTensorData<int8_t>(input),
      input->dims->data, input->dims->size,
      tflite::micro::GetTensorData<int8_t>(output),
      output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, temp_buffer, false);
  return return_status == true ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ReduceUInt8KeepDims(TfLiteContext* context, OpData* op_data,
                         TfLiteReducerParams* params,
                         const TfLiteEvalTensor* input,
                         const TfLiteEvalTensor* axis,
                         TfLiteEvalTensor* output) {
  tflite::MeanParams op_params;
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  ResolveAxis(tflite::micro::GetTensorData<int>(axis), num_axis, &op_params);
  MeanUInt8(op_params, op_data, tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<uint8_t>(input),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<uint8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus ReduceUInt8ChangeDims(TfLiteContext* context, OpData* op_data,
                           TfLiteReducerParams* params,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* axis,
                           TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  int32_t* temp_buffer = op_data->temp_buffer;
  bool return_status = Mean(
      tflite::micro::GetTensorData<uint8_t>(input), input->dims->data,
      input->dims->size, tflite::micro::GetTensorData<uint8_t>(output),
      output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, temp_buffer);
  return return_status == true ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ReduceUInt8ChangeDimsAndQuant(TfLiteContext* context, OpData* op_data,
                                   TfLiteReducerParams* params,
                                   const TfLiteEvalTensor* input,
                                   const TfLiteEvalTensor* axis,
                                   TfLiteEvalTensor* output) {
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];
  int32_t* temp_buffer = op_data->temp_buffer;
  bool return_status = QuantizedMeanOrSum(
      op_data, tflite::micro::GetTensorData<uint8_t>(input), input->dims->data, input->dims->size,
      tflite::micro::GetTensorData<uint8_t>(output), output->dims->data, output->dims->size,
      tflite::micro::GetTensorData<int>(axis), num_axis, params->keep_dims,
      temp_index, resolved_axis, temp_buffer, false);
  return return_status == true ? kTfLiteOk : kTfLiteError;
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
#if ! TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
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
    void* raw = context->AllocatePersistentBuffer(context, sizeof(int32_t) * output_size);
    op_data->temp_buffer = reinterpret_cast<int32_t*>(raw);
    op_data->input_zp = input->params.zero_point;
    auto input_shape = tflite::micro::GetTensorShape(tflite::micro::GetEvalInput(context, node, 0));
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const float num_elements_in_axis = input_width * input_height;
    double real_scale =
          static_cast<double>(input->params.scale / (num_elements_in_axis * output->params.scale));
    QuantizeMultiplier(real_scale, &op_data->mean_multiplier, &op_data->mean_shift);
    op_data->output_zp = output->params.zero_point;
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
            TFLM_SET_KERNEL_VARIANT(ReduceFloatKeepDims);
      } else {
        op_data->eval_function =
            TFLM_SET_KERNEL_VARIANT(ReduceFloatChangeDims);
      }
    } break;
    case kTfLiteInt8: {
      if (params->keep_dims) {
        op_data->eval_function =
            TFLM_SET_KERNEL_VARIANT(ReduceInt8KeepDims);
      } else if (op_data->input_zp == op_data->output_zp &&
          input->params.scale == output->params.scale) {
        op_data->eval_function =
            TFLM_SET_KERNEL_VARIANT(ReduceInt8ChangeDims);
      } else {
        op_data->eval_function =
            TFLM_SET_KERNEL_VARIANT(ReduceInt8ChangeDimsAndQuant);
      }
    } break;
    case kTfLiteUInt8: {
      if (params->keep_dims) {
        op_data->eval_function =
            TFLM_SET_KERNEL_VARIANT(ReduceUInt8KeepDims);
      } else if (op_data->input_zp == op_data->output_zp &&
          input->params.scale == output->params.scale) {
        op_data->eval_function =
            TFLM_SET_KERNEL_VARIANT(ReduceUInt8ChangeDims);
      } else {
        op_data->eval_function =
            TFLM_SET_KERNEL_VARIANT(ReduceInt8ChangeDimsAndQuant);
      }
    } break;
    default:
      // TODO(b/144955155): Support uint8_t(b/144955155) and int8_t(b/144955018)
      TF_LITE_ENSURE_MSG(context, false,
                         "Currently, only float32, int8 or uint8 input type "
                         "is supported.");
  }

  TFLM_RECORD_OP_USER_DATA("reduce", static_opdata(*op_data, static_cast<size_t>(output_size)));
#endif
  return kTfLiteOk;
}


TfLiteStatus EvalMean(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TfLiteReducerParams* params =
      reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  return op_data->eval_function(context, op_data, params, input, axis, output);
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
