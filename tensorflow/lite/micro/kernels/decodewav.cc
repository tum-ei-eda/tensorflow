#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/op_macros.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include <cstring>
#include <memory>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/micro/micro_string.h"

#define WAV_STRING_SIZE \
  32000 * 1 * 1 + 48  // Size of char is 1 Byte(This is max expected)
#define DATA_SIZE WAV_STRING_SIZE - 44  // The rest of WAV_STRING WITHOUT HEADER
#define WAV_HEADER_SIZE 48  // HEADER ACCORDING TO THE WAV/RIFF protocol
#define TF_PACKED __attribute__((packed))
constexpr bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;

// max supports:  Bit Sample : 8 Bits
//               Bit Rate : 0,507 Mbps
//               1 Sec WAV files as max : ~ 30 Kilobytes
//               Bytes per frame : 2
//               Channel count : 1(mono) || 2(strero)
//               Total number of samples 160000 Samples (max)
//               Output: float_values of size: #samples*channels = 16000(mono)
//               values(per sec)

namespace tflite {
namespace ops {
namespace micro {
namespace decodewav {
struct TF_PACKED RiffChunk {
  char chunk_id[4];
  char chunk_data_size[4];
  char riff_type[4];
};
static_assert(sizeof(RiffChunk) == 12, "TF_PACKED does not work.");

struct TF_PACKED FormatChunk {
  char chunk_id[4];
  char chunk_data_size[4];
  char compression_code[2];
  char channel_numbers[2];
  char sample_rate[4];
  char bytes_per_second[4];
  char bytes_per_frame[2];
  char bits_per_sample[2];
};
static_assert(sizeof(FormatChunk) == 24, "TF_PACKED does not work.");

struct TF_PACKED DataChunk {
  char chunk_id[4];
  char chunk_data_size[4];
};
static_assert(sizeof(DataChunk) == 8, "TF_PACKED does not work.");

struct TF_PACKED WavHeader {
  RiffChunk riff_chunk;
  FormatChunk format_chunk;
  DataChunk data_chunk;
};
static_assert(sizeof(WavHeader) ==
                  sizeof(RiffChunk) + sizeof(FormatChunk) + sizeof(DataChunk),
              "TF_PACKED does not work.");

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr char kRiffChunkId[5] = "RIFF";
constexpr char kRiffType[5] = "WAVE";
constexpr char kFormatChunkId[5] = "fmt ";
constexpr char kDataChunkId[5] = "data";

struct OpData {
  // Some Data used during Compute
  int16 desired_channels;
  int32 desired_samples;
  uint32 decoded_sample_count;
  uint16 decoded_channel_count;
  uint32 decoded_sample_rate;

  int max_number_samples;
  const int wav_string_size = WAV_STRING_SIZE;

  int decoded_samples_idx;
  int wav_string_idx;
  int found_text_idx;
  int wav_length;
  int outputmatrix_idx;

  float* decoded_samples;
  const char* wav_string;
  char* found_text;
  float* outputmatrix;
};

TfLiteStatus IncrementOffset(int old_offset, size_t increment, size_t max_size,
                             int* new_offset) {
  if (old_offset < 0) {
    return kTfLiteError;
  }
  if (old_offset > max_size) {
    return kTfLiteError;
  }
  *new_offset = old_offset + increment;
  if (*new_offset > max_size) {
    return kTfLiteError;
  }
  if (*new_offset < 0) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus ExpectText(const char* data, const char* expected_text,
                        int* offset, char* found_text, int data_length) {
  int new_offset;
  TF_LITE_ENSURE_STATUS(IncrementOffset(*offset, 4, data_length, &new_offset));
  for (int i = 0; i < 4; i++) {
    found_text[i] = data[*offset + i];
  }
  if (strcmp(found_text, expected_text) != 0) {
    return kTfLiteError;
  }
  *offset = new_offset;
  return kTfLiteOk;
}

template <class T>
TfLiteStatus ReadValue(const char* data, T* value, int* offset) {
  int new_offset;
  TF_LITE_ENSURE_STATUS(
      IncrementOffset(*offset, sizeof(T), WAV_STRING_SIZE, &new_offset));
  if (kLittleEndian) {
    memcpy(value, data + *offset, sizeof(T));
  } else {
    *value = 0;
    const uint8* data_buf = reinterpret_cast<const uint8*>(data + *offset);
    int shift = 0;
    for (int i = 0; i < sizeof(T); ++i, shift += 8) {
      *value = *value | (data_buf[i] << shift);
    }
  }
  *offset = new_offset;
  return kTfLiteOk;
}

TfLiteStatus ReadString(const char* data, int expected_length, char* value,
                        int* offset, int data_length) {
  int new_offset;
  TF_LITE_ENSURE_STATUS(
      IncrementOffset(*offset, expected_length, data_length, &new_offset));
  for (int i = 0; i < 4; i++) {
    value[i] = data[*offset + i];
  }
  *offset = new_offset;
  return kTfLiteOk;
}

inline float Int16SampleToFloat(int16 data) {
  constexpr float kMultiplier = 1.0f / (1 << 15);
  return data * kMultiplier;
}

// Decodes the little-endian signed 16-bit PCM WAV file data (aka LIN16
// encoding) into a float Tensor. The channels are encoded as the lowest
// dimension of the tensor, with the number of frames as the second. This means
// that a four frame stereo signal will have the shape [4, 2]. The sample rate
// is read from the file header, and an error is returned if the format is not
// supported.
// The results are output as floats within the range -1 to 1,
TfLiteStatus DecodeLin16WaveAsFloatVector(
    TfLiteContext* context, const char* wav_string, float* float_values,
    uint32* sample_count, uint16* channel_count, uint32* sample_rate,
    OpData* op_data) {
  int offset = 0;
  uint32 total_file_size;
  uint32 format_chunk_size;
  uint16 audio_format;
  uint32 bytes_per_second;
  uint16 bytes_per_sample;

  context->ReportError(context, "Reading RiffChunk");
  TF_LITE_ENSURE_STATUS(ExpectText(wav_string, kRiffChunkId, &offset,
                                   op_data->found_text, op_data->wav_length));
  context->ReportError(context, "Reading file_size");
  TF_LITE_ENSURE_STATUS(
      ReadValue<uint32>(wav_string, &total_file_size, &offset));
  context->ReportError(context, "--> %d", total_file_size);
  op_data->wav_length = total_file_size + 8;
  context->ReportError(context, "Reading RiffType");
  TF_LITE_ENSURE_STATUS(ExpectText(wav_string, kRiffType, &offset,
                                   op_data->found_text, op_data->wav_length));
  context->ReportError(context, "Reading FormatChunkId");
  TF_LITE_ENSURE_STATUS(ExpectText(wav_string, kFormatChunkId, &offset,
                                   op_data->found_text, op_data->wav_length));
  context->ReportError(context, "Reading format_chunk_size");
  TF_LITE_ENSURE_STATUS(
      ReadValue<uint32>(wav_string, &format_chunk_size, &offset));
  context->ReportError(context, "--> %d", format_chunk_size);
  if ((format_chunk_size != 16) && (format_chunk_size != 18)) {
    context->ReportError(
        context, "Bad format chunk size for WAV: Expected 16 or 18, but got %d",
        format_chunk_size);
    return kTfLiteError;
  }
  context->ReportError(context, "Reading audio_format");
  TF_LITE_ENSURE_STATUS(ReadValue<uint16>(wav_string, &audio_format, &offset));
  context->ReportError(context, "--> %d", audio_format);
  if (audio_format != 1) {
    context->ReportError(
        context, "Bad audio format for WAV: Expected 1 (PCM), but got %d",
        audio_format);
    return kTfLiteError;
  }
  context->ReportError(context, "Reading channel_count");
  TF_LITE_ENSURE_STATUS(ReadValue<uint16>(wav_string, channel_count, &offset));
  context->ReportError(context, "--> %d", *channel_count);
  if (*channel_count < 1) {
    context->ReportError(
        context,
        "Bad number of channels for WAV: Expected at least 1, but got %d",
        channel_count);
    return kTfLiteError;
  }
  context->ReportError(context, "Reading sample_rate");
  TF_LITE_ENSURE_STATUS(ReadValue<uint32>(wav_string, sample_rate, &offset));
  context->ReportError(context, "--> %d", *sample_rate);
  context->ReportError(context, "Reading bytes_per_second");
  TF_LITE_ENSURE_STATUS(
      ReadValue<uint32>(wav_string, &bytes_per_second, &offset));
  context->ReportError(context, "--> %d", bytes_per_second);
  context->ReportError(context, "Reading bytes_per_sample");
  TF_LITE_ENSURE_STATUS(
      ReadValue<uint16>(wav_string, &bytes_per_sample, &offset));
  context->ReportError(context, "--> %d", bytes_per_sample);

  uint16 bits_per_sample;
  context->ReportError(context, "Reading bits_per_sample");
  TF_LITE_ENSURE_STATUS(
      ReadValue<uint16>(wav_string, &bits_per_sample, &offset));
  context->ReportError(context, "--> %d", bytes_per_sample);
  if (bits_per_sample != 16) {
    context->ReportError(context, "Can only read 16-bit WAV files");
    return kTfLiteError;
  }

  const uint32 expected_bytes_per_sample =
      ((bits_per_sample * *channel_count) + 7) / 8;
  if (bytes_per_sample != expected_bytes_per_sample) {
    context->ReportError(context, "Bad bytes per sample in WAV header");
    return kTfLiteError;
  }
  const uint32 expected_bytes_per_second = bytes_per_sample * *sample_rate;
  if (bytes_per_second != expected_bytes_per_second) {
    context->ReportError(context, "Bad bytes per second in WAV header");
    return kTfLiteError;
  }
  if (format_chunk_size == 18) {
    // Skip over this unused section.
    offset += 2;
  }
  bool was_data_found = false;
  while (offset < op_data->wav_length) {
    TF_LITE_ENSURE_STATUS(ReadString(wav_string, 4, op_data->found_text,
                                     &offset, op_data->wav_length));
    uint32 chunk_size;
    TF_LITE_ENSURE_STATUS(ReadValue<uint32>(wav_string, &chunk_size, &offset));
    if (chunk_size > std::numeric_limits<int32>::max()) {
      context->ReportError(context, "WAV data chunk is too large");
      return kTfLiteError;
    }
    if (strcmp(op_data->found_text, kDataChunkId) == 0) {
      if (was_data_found) {
        context->ReportError(context, "More than one data chunk found in WAV");
        return kTfLiteError;
      }
      was_data_found = true;
      *sample_count = chunk_size / bytes_per_sample;
      const uint32 data_count = *sample_count * *channel_count;
      int unused_new_offset = 0;
      // Validate that the data exists before allocating space for it
      // (prevent easy OOM errors).
      TF_LITE_ENSURE_STATUS(IncrementOffset(offset, sizeof(int16) * data_count,
                                            op_data->wav_length,
                                            &unused_new_offset));
      TF_LITE_ENSURE(
          context,
          (data_count <= op_data->desired_samples *
                             op_data->desired_channels));  // float_values_max ~
                                                           // 16000*2 for e.g.
      for (int i = 0; i < data_count; i++) {
        int16 single_channel_value = 0;
        TF_LITE_ENSURE_STATUS(
            ReadValue<int16>(wav_string, &single_channel_value, &offset));
        float_values[i] = Int16SampleToFloat(single_channel_value);
      }
    } else {
      offset += chunk_size;
    }
    if (!was_data_found) {
      return kTfLiteError;
    }
    return kTfLiteOk;
  }
}

// Allocate enough required dims for a given tensor.
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

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  context->ReportError(context, "Inside Init");
  void* raw;
  OpData* op_data = nullptr;
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  context->AllocatePersistentBuffer(context, sizeof(OpData), &raw);
  op_data = reinterpret_cast<OpData*>(raw);

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  op_data->desired_channels = m["desired_channels"].AsInt8();
  op_data->desired_samples = m["desired_samples"].AsInt64();

  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "Inside Prepare");
  auto* op_data = static_cast<OpData*>(node->user_data);
  const int max_number_samples =
      op_data->desired_samples * 1;  // 1 secs supported
  const int max_float_values =
      op_data->desired_samples * op_data->desired_channels * 1;

  // Request the buffers needed for Prepare and decoding
  // Mostly here the params are staic_casted from the used internal function
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  TFLITE_DCHECK(context != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output1 = GetOutput(context, node, kOutputTensor);
  TfLiteTensor* output2 = GetOutput(context, node, 1);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);  // Must be scalar.
  // Allocatiion in ScratchBuffer
  context->RequestScratchBufferInArena(context, WAV_STRING_SIZE,
                                       &op_data->wav_string_idx);
  context->ReportError(context, "decoded samples: %d",
                       max_number_samples * sizeof(float));
  context->RequestScratchBufferInArena(
      context, max_number_samples * sizeof(float),
      &op_data->decoded_samples_idx);  // TOO large???
  context->RequestScratchBufferInArena(
      context, 5,
      &op_data->found_text_idx);  // ALL texts have max length of 4 Bytes so 1
                                  // more extra Byte overhead.
  context->ReportError(
      context, "allocated %d ",
      op_data->desired_samples * op_data->desired_channels * sizeof(float));
  context->RequestScratchBufferInArena(
      context,
      op_data->desired_samples * op_data->desired_channels * sizeof(float),
      &op_data->outputmatrix_idx);
  // Tensor Allocation:
  context->ReportError(context, "Inside Prepare 2");
  TF_LITE_ENSURE_STATUS(AllocateOutDimensions(context, &output1->dims,
                                              op_data->desired_samples,
                                              op_data->desired_channels));
  TF_LITE_ENSURE_STATUS(AllocateOutDimensions(context, &output2->dims, 1));
  context->ReportError(context, "End Prepare.");
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "Inside Eval");
  // Where tensors are reinterpreted and functions are called, depending on the
  // input type or value

  // Usage is okay. But why when it's obviously a const along all the program.
  // Can be predefined.
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output1 = GetOutput(context, node, kOutputTensor);
  TfLiteTensor* output2 = GetOutput(context, node, 1);

  void* raw;
  auto* op_data = static_cast<OpData*>(node->user_data);

  // Get the buffers.
  raw = context->GetScratchBuffer(context, op_data->wav_string_idx);
  op_data->wav_string = reinterpret_cast<char*>(raw);
  raw = context->GetScratchBuffer(context, op_data->decoded_samples_idx);
  op_data->decoded_samples = reinterpret_cast<float*>(raw);
  raw = context->GetScratchBuffer(context, op_data->found_text_idx);
  op_data->found_text = reinterpret_cast<char*>(raw);

  raw = context->GetScratchBuffer(context, op_data->outputmatrix_idx);
  op_data->outputmatrix = reinterpret_cast<float*>(raw);
  output1->data.f = op_data->outputmatrix;

  op_data->wav_string = GetTensorData<char>(input);
  op_data->wav_length = WAV_STRING_SIZE;

  if (op_data->wav_length > WAV_STRING_SIZE) {
    context->ReportError(context, "WAV contents are too large");
    return kTfLiteError;
  }
  TF_LITE_ENSURE_STATUS(DecodeLin16WaveAsFloatVector(
      context, op_data->wav_string, op_data->decoded_samples,
      &op_data->decoded_sample_count, &op_data->decoded_channel_count,
      &op_data->decoded_sample_rate, op_data));
  context->ReportError(context, "Eval Phase 2\n ");
  int32 output_sample_count;
  if (op_data->desired_samples == -1) {
    output_sample_count = op_data->decoded_sample_count;
  } else {
    output_sample_count = op_data->desired_samples;
  }
  int32 output_channel_count;
  if (op_data->desired_channels == -1) {
    output_channel_count = op_data->decoded_channel_count;
  } else {
    output_channel_count = op_data->desired_channels;
  }
  context->ReportError(context, "Output sample count: %d", output_sample_count);
  context->ReportError(context, "Output channel : %d", output_channel_count);
  for (int sample = 0; sample < output_sample_count; ++sample) {
    for (int channel = 0; channel < output_channel_count; ++channel) {
      float output_value;
      if (sample >= op_data->decoded_sample_count) {
        output_value = 0.0f;
      } else {
        int source_channel;
        if (channel < op_data->decoded_channel_count) {
          source_channel = channel;
        } else {
          source_channel = op_data->decoded_channel_count - 1;
        }
        const int decoded_index =
            (sample * op_data->decoded_channel_count) + source_channel;
        output_value = op_data->decoded_samples[decoded_index];
      }
      op_data->outputmatrix[sample * op_data->decoded_channel_count + channel] =
          output_value;

      // context->ReportError(context, "output_value: %f", output_value);
    }
  }
  context->ReportError(context, "out");
  context->ReportError(context, "decoded:%d",op_data->decoded_sample_rate);
  output2->data.i32 =
      reinterpret_cast<int32*>(op_data->decoded_sample_rate);  // Questionable?.
  context->ReportError(context,"output2data: %d",output2->data.i32);
  context->ReportError(context, "End of phase 2 Invoke ended");

}
}  // namespace decodewav
TfLiteRegistration* Register_DECODE_WAV() {
  static TfLiteRegistration r = {

      decodewav::Init, decodewav::Free, decodewav::Prepare, decodewav::Eval};
  return &r;
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite