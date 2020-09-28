/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <numeric>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
//#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
//#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"

#include "tensorflow/lite/kernels/internal/spectrogram.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/mem_func_helpers.cc"

// Under construction!!!!
// Check correct new tf usage 
// Apply deque and internal Custom allocator use.
// Optimize memory sizes.

namespace tflite {
namespace ops {
namespace custom {  // can change later to correct path for micro
namespace audio_spectrogram {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

enum KernelType {
  kReference,
};

typedef struct {
  int window_size;
  int stride;
  bool magnitude_squared;
  int output_height;
  internal::Spectrogram* spectrogram;
} TfLiteAudioSpectrogramParams;

struct OpData {
  TfLiteAudioSpectrogramParams* params;
  int input_for_channel_idx;
  int spectrogram_output_idx;

  // float* input_for_channel;
  // float* spectrogram_output;  // Check this because of use persis Buffer
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* raw;
  OpData* op_data = nullptr;

  auto* data = new TfLiteAudioSpectrogramParams;
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  context->AllocatePersistentBuffer(context, sizeof(OpData), &raw);
  op_data = reinterpret_cast<OpData*>(raw);
  op_data->params = data;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->window_size = m["window_size"].AsInt64();
  data->stride = m["stride"].AsInt64();
  data->magnitude_squared = m["magnitude_squared"].AsBool();
  data->spectrogram = new internal::Spectrogram(context);
  context->ReportError(
      context, "Window size: %d\nStride: %d\nMagnitude(bool) %d \n",
      data->window_size, data->stride, data->magnitude_squared);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  auto* params = reinterpret_cast<TfLiteAudioSpectrogramParams*>(buffer);
  delete params->spectrogram;
  delete params;
}

TfLiteStatus AllocateOutDimensions(TfLiteContext* context,
                                   TfLiteIntArray** dims, int x, int y = 0,
                                   int z = 0) {
  int size = 1;

  size = size * x;
  size = (y > 0) ? size * y : size;
  size = (z > 0) ? size * z : size;

  TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
      context, TfLiteIntArrayGetSizeInBytes(size),
      reinterpret_cast<void**>(dims)));
  (*dims)->size = size;
  (*dims)->data[0] = x;
  if (y > 0) {
    (*dims)->data[1] = y;
  }
  if (z > 0) {
    (*dims)->data[2] = z;
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = static_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input =
      GetInput(context, node, kInputTensor);  // new tflite now has GetInputSafe
  TfLiteTensor* output = GetOutput(
      context, node, kOutputTensor);  // new tflite now has GetOutputSafe

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 2);

  context->ReportError(context, "Output dim size: %d: ", output->dims->size);
  if (output->dims->size != 3) {
    TF_LITE_ENSURE_STATUS(AllocateOutDimensions(context, &output->dims, 3, 1));
  }

  TF_LITE_ENSURE_EQ(context, output->dims->size,
                    3);  // Check if previous step worked

  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  // can be transfered in CalculateOpData (A new function)
  TF_LITE_ENSURE(context, op_data->params->spectrogram->Initialize(
                              op_data->params->window_size,
                              op_data->params->stride, context));
  const int64_t sample_count = input->dims->data[0];
  const int64_t length_minus_window =
      (sample_count - op_data->params->window_size);
  if (length_minus_window < 0) {
    op_data->params->output_height = 0;
  } else {
    op_data->params->output_height =
        1 + (length_minus_window / op_data->params->stride);
  }

  output->dims->data[0] = input->dims->data[1];
  output->dims->data[1] = op_data->params->output_height;
  output->dims->data[2] =
      op_data->params->spectrogram->output_frequency_channels();

  context->ReportError(context, "Byes for output: %d",
                       output->bytes);  // Debug report
  context->ReportError(context, "dims of output: %d", NumDimensions(output));
  context->ReportError(context, "Elements of output: %d", NumElements(output));

  // ScratchBuffers
  context->RequestScratchBufferInArena(context, input->dims->data[0],
                                       &op_data->input_for_channel_idx);

  context->RequestScratchBufferInArena(
      context,
      op_data->params->output_height *
          op_data->params->spectrogram->output_frequency_channels(),
      &op_data->spectrogram_output_idx);

  return kTfLiteOk;
}

// template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  void* raw;
  auto* op_data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE(context, op_data->params->spectrogram->Initialize(
                              op_data->params->window_size,
                              op_data->params->stride, context));

  const float* input_data =
      GetTensorData<float>(input);  // Input Flat buffer content

  const int64_t sample_count = input->dims->data[0];
  const int64_t channel_count = input->dims->data[1];

  const int64_t output_width =
      op_data->params->spectrogram->output_frequency_channels();

  float* output_flat =
      GetTensorData<float>(output);  // Output Flat buffer content

  std::vector<float, tflite::ops::micro::ArenaBufferAllocator<float>>
      input_for_channel(  // First vector
          0, tflite::ops::micro::ArenaBufferAllocator<float>(
                 op_data->input_for_channel_idx, sample_count, context));
  for (int64_t channel = 0; channel < channel_count; ++channel) {
    float* output_slice =
        output_flat + (channel * op_data->params->output_height * output_width);
    for (int i = 0; i < sample_count; ++i) {
      input_for_channel[i] = input_data[i * channel_count + channel];
    }

    // std::vector<std::vector<float>> spectrogram_output;

    std::vector<  // Vector of vectors of float
        std::vector<float, tflite::ops::micro::ArenaBufferAllocator<float>>,
        tflite::ops::micro::ArenaBufferAllocator<float>>
        spectrogram_output(
            0,
            tflite::ops::micro::ArenaBufferAllocator<float>(
                op_data->spectrogram_output_idx,
                op_data->params->output_height *
                    op_data->params->spectrogram->output_frequency_channels(),
                context));

    TF_LITE_ENSURE(
        context,
        op_data->params->spectrogram->ComputeSquaredMagnitudeSpectrogram(
            input_for_channel, &spectrogram_output));
    TF_LITE_ENSURE_EQ(context, spectrogram_output.size(),
                      op_data->params->output_height);
    TF_LITE_ENSURE(context, spectrogram_output.empty() ||
                                (spectrogram_output[0].size() == output_width));
    for (int row_index = 0; row_index < op_data->params->output_height;
         ++row_index) {
      const std::vector<float, tflite::ops::micro::ArenaBufferAllocator<float>>&
          spectrogram_row = spectrogram_output[row_index];
      TF_LITE_ENSURE_EQ(context, spectrogram_row.size(), output_width);
      float* output_row = output_slice + (row_index * output_width);
      if (op_data->params->magnitude_squared) {
        for (int i = 0; i < output_width; ++i) {
          output_row[i] = spectrogram_row[i];
        }
      } else {
        for (int i = 0; i < output_width; ++i) {
          output_row[i] = sqrtf(spectrogram_row[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace audio_spectrogram

TfLiteRegistration* Register_AUDIO_SPECTROGRAM() {
  static TfLiteRegistration r = {
      audio_spectrogram::Init, audio_spectrogram::Free,
      audio_spectrogram::Prepare, audio_spectrogram::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
