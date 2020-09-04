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

#include <cstddef>
#include <cstdint>
#include <iostream>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

MockAllocator* mock_allocator;
TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ctx, size_t bytes,
                                      void** ptr) {
  return mock_allocator->AllocatePersistentBuffer(ctx, bytes, ptr);
}

// TODO Factor into shared support library
template <typename CONTAINER_T>
static std::vector<CONTAINER_T> PackedSub8BitCustomQuantization(
    const uint8_t* data, size_t elts, size_t minor_dim_size,
    TfLiteCustomSub8BitPackingDetails* format) {
  unsigned int container_bits = format->container_bits;
  size_t bits_per_item = format->bits_per_item;
  assert(container_bits <= 32u);
  CONTAINER_T mask = (static_cast<CONTAINER_T>(1) << bits_per_item) -
                     static_cast<CONTAINER_T>(1);
  uint32_t container_buf = 0;
  // Lazy way of getting sufficient CONTAINER_T aligned storage...
  std::vector<CONTAINER_T> packed_data(elts / (container_bits / 8));

  uint8_t* packed_data_byte = reinterpret_cast<uint8_t*>(packed_data.data());
  CONTAINER_T bits_in_container = 0;
  for (size_t i = 0; i < elts; ++i) {
    // Little-endian packing...
    container_buf |= (static_cast<CONTAINER_T>(data[i]) & mask) << bits_in_container;
    bits_in_container += bits_per_item;
    // Flush container when insufficient space for another item
    // Start of each minor dimension to ensure CONTAINER_T aligned...
    // ToDO IFX_PATCH: probably more efficient to align on selected dimension
    // (ideally: dependent on op) to handle depthwise conv / inner loop 2D conv
    if (bits_in_container + bits_per_item > container_bits ||
        (i % minor_dim_size == (minor_dim_size - 1))) {
      // Flatbuffers are stored little-endian
      for (size_t bits = 0; bits < container_bits; bits += 8) {
        uint8_t byte = (container_buf & 0xff);
        *packed_data_byte = byte;
        ++packed_data_byte;
        container_buf >>= 8;
      }
      bits_in_container = 0;
      container_buf = 0;
    }
  }

  assert(bits_in_container == 0);
  return packed_data;
}



static void SetPackingParams(TfLiteTensor& tensor, float min, float max,
                             TfLiteCustomSub8BitPackingDetails* format) {
  tensor.params.scale = ScaleFromMinMaxPacked(min, max, format->bits_per_item);
  tensor.params.zero_point =
      ZeroPointFromMinMaxPacked(min, max, format->bits_per_item);
  tensor.quantization.details.type = kTfLiteSub8BitPackedUniformDetail;
  tensor.quantization.details.data.custom_sub8bit_packing = format;
}

TfLiteStatus TestFullyConnectedFloat(
    const int* input_dims_data, const float* input_data,
    const int* weights_dims_data, const float* weights_data,
    const int* bias_dims_data, const float* bias_data,
    const float* expected_output_data, const int* output_dims_data,
    TfLiteFusedActivation activation, float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(weights_data, weights_dims),
      CreateFloatTensor(bias_data, bias_dims),
      CreateFloatTensor(output_data, output_dims),
  };

  TfLiteFullyConnectedParams builtin_data = {
      activation, kTfLiteFullyConnectedWeightsFormatDefault, false, false};

  int inputs_array_data[] = {inputs_size, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {outputs_size, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      ops::micro::Register_FULLY_CONNECTED();
  micro::KernelRunner runner(
      registration, tensors, tensors_size, inputs_array, outputs_array,
      reinterpret_cast<void*>(&builtin_data), micro_test::reporter);

  TfLiteStatus status = runner.InitAndPrepare();
  if (status != kTfLiteOk) {
    return status;
  }

  status = runner.Invoke();
  if (status != kTfLiteOk) {
    return status;
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus TestFullyConnectedQuantized(
    const int* input_dims_data, const T* input_data, const float input_min,
    const float input_max, const int* weights_dims_data, const T* weights_data,
    const float weights_min, const float weights_max,
    TfLiteCustomSub8BitPackingDetails* packing, const int* bias_dims_data,
    const int32_t* bias_data, const float bias_scale,
    const T* expected_output_data, const int* output_dims_data,
    const float output_min, const float output_max,
    TfLiteFusedActivation activation, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, input_min, input_max),
      CreateQuantizedTensor(weights_data, weights_dims, weights_min,
                            weights_max),
      CreateQuantized32Tensor(bias_data, bias_dims, bias_scale),
      CreateQuantizedTensor(output_data, output_dims, output_min, output_max),
  };

  if (packing) {
    SetPackingParams(tensors[1], weights_min, weights_max, packing);
  }

  TfLiteFullyConnectedParams builtin_data = {
      activation, kTfLiteFullyConnectedWeightsFormatDefault, false, false};

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      ops::micro::Register_FULLY_CONNECTED();
  micro::KernelRunner runner(
      registration, tensors, tensors_size, inputs_array, outputs_array,
      reinterpret_cast<void*>(&builtin_data), micro_test::reporter);

  TfLiteStatus status = runner.InitAndPrepare();
  if (status != kTfLiteOk) {
    return status;
  }

  status = runner.Invoke();
  if (status != kTfLiteOk) {
    return status;
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTest) {
  const int input_dims_data[] = {2, 2, 10};
  const float input_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  };
  const int weights_dims_data[] = {2, 3, 10};
  const float weights_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  };
  const int bias_dims_data[] = {1, 3};
  const float bias_data[] = {1, 2, 3};
  const float expected_output_data[] = {
      24, 25, 26, 58, 59, 60,
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  float output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          input_dims_data, input_data, weights_dims_data, weights_data,
          bias_dims_data, bias_data, expected_output_data, output_dims_data,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest2) {
  const int input_dims_data[] = {2, 2, 2};
  const float input_data[] = {
      1, 2,  // b = 0
      2, 1,  // b = 1
  };
  const int weights_dims_data[] = {2, 1, 2};
  const float weights_data[] = {
      2, 4,  // u = 0
  };
  const int bias_dims_data[] = {1, 1};
  const float bias_data[] = {1};
  const float expected_output_data[] = {
      11,
      9,
  };
  const int output_dims_data[] = {2, 2, 1};

  const int output_dims_count = 6;
  float output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          input_dims_data, input_data, weights_dims_data, weights_data,
          bias_dims_data, bias_data, expected_output_data, output_dims_data,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestRelu) {
  const int input_dims_data[] = {2, 2, 10};
  const float input_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  };
  const int weights_dims_data[] = {2, 3, 10};
  const float weights_data[] = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 0
      -1, -2, -3, -4, -5, -6, -7, -8, -9, -10,  // u = 1
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 2
  };
  const int bias_dims_data[] = {1, 3};
  const float bias_data[] = {1, -2, 3};
  const float expected_output_data[] = {
      24, 0, 26, 58, 0, 60,
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  float output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          input_dims_data, input_data, weights_dims_data, weights_data,
          bias_dims_data, bias_data, expected_output_data, output_dims_data,
          kTfLiteActRelu, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {2, 2, 10};
  const uint8_t input_data[] = {
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
      F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
      F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(25, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(59, output_min, output_max), F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=packing*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

// IFX_PATCH
// TODO eliminate code-duplication, sane names for arguments patterns for
// testing utils.  Move some functinoality to commom lite?
//
TF_LITE_MICRO_TEST(SmokeTestPackedQuantizedUInt8_4) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;
  using tflite::testing::F2QB;
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_sf = 2.0;
  const float weights_sf = 32.0;
  const float input_min = -128.0f / input_sf;
  const float input_max = 127.0f / input_sf;
  const float weights_min = -8.0f / weights_sf;
  const float weights_max = 7.0f / weights_sf;
  const float bias_scale = 1.0f / (input_sf * weights_sf);
  const float output_min = -128.0f / 32.0f;
  const float output_max = 127.0f / 32.0f;

  const size_t num_weights = 8;
  const int input_dims_data[] = {1, 1, num_weights};
  const uint8_t input_data[] = {
      F2Q(1 / input_sf, input_min, input_max),
      F2Q(-2 / input_sf, input_min, input_max),
      F2Q(3 / input_sf, input_min, input_max),
      F2Q(-4 / input_sf, input_min, input_max),
      F2Q(5 / input_sf, input_min, input_max),
      F2Q(-6 / input_sf, input_min, input_max),
      F2Q(7 / input_sf, input_min, input_max),
      F2Q(-8 / input_sf, input_min, input_max),

  };


  const int weights_dims_data[] = {2, 1, num_weights};
  const uint8_t weights_data[] = {
      F2QB<4>(1 / weights_sf, weights_min, weights_max),
      F2QB<4>(2 / weights_sf, weights_min, weights_max),
      F2QB<4>(3 / weights_sf, weights_min, weights_max),
      F2QB<4>(-4 / weights_sf, weights_min, weights_max),
      F2QB<4>(5 / weights_sf, weights_min, weights_max),
      F2QB<4>(6 / weights_sf, weights_min, weights_max),
      F2QB<4>(7 / weights_sf, weights_min, weights_max),
      F2QB<4>(7 / weights_sf, weights_min, weights_max),
  };

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1, {}};

  auto packed_weights =
      tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
          weights_data, 1u * num_weights, num_weights, &packing);
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(0, bias_scale),
      F2Q32(0, bias_scale),
      F2Q32(0, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q((1 * 1 + -2 * 2 + 3 * 3 + -4 * -4 + 5 * 5 + -6 * 6 + 7 * 7 + -8 * 7) 
             / (input_sf * weights_sf),
          output_min, output_max),

  };
  const int output_dims_data[] = {2, 1, 1};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          packed_weights.data(), weights_min, weights_max, &packing,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SmokeTestPackedQuantizedUInt8_5) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;
  using tflite::testing::F2QB;
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_sf = 2.0;
  const float weights_sf = 32.0;
  const float input_min = -128.0f / input_sf;
  const float input_max = 127.0f / input_sf;
  const float weights_min = -16.0f / weights_sf;
  const float weights_max = 15.0f / weights_sf;
  const float bias_scale = 1.0f / (input_sf * weights_sf);
  const float output_min = -128.0f / 32.0f;
  const float output_max = 127.0f / 32.0f;

  const size_t num_weights = 8;
  const int input_dims_data[] = {1, 1, num_weights};
  const uint8_t input_data[] = {
      F2Q(1 / input_sf, input_min, input_max),
      F2Q(-2 / input_sf, input_min, input_max),
      F2Q(3 / input_sf, input_min, input_max),
      F2Q(-4 / input_sf, input_min, input_max),
      F2Q(5 / input_sf, input_min, input_max),
      F2Q(-6 / input_sf, input_min, input_max),
      F2Q(7 / input_sf, input_min, input_max),
      F2Q(-8 / input_sf, input_min, input_max),

  };

  const int weights_dims_data[] = {2, 1, num_weights};
  const uint8_t weights_data[] = {
      F2QB<5>(1 / weights_sf, weights_min, weights_max),
      F2QB<5>(2 / weights_sf, weights_min, weights_max),
      F2QB<5>(3 / weights_sf, weights_min, weights_max),
      F2QB<5>(-4 / weights_sf, weights_min, weights_max),
      F2QB<5>(5 / weights_sf, weights_min, weights_max),
      F2QB<5>(6 / weights_sf, weights_min, weights_max),
      F2QB<5>(7 / weights_sf, weights_min, weights_max),
      F2QB<5>(7 / weights_sf, weights_min, weights_max),
  };

  TfLiteCustomSub8BitPackingDetails packing = {5, 16, 1, {}};

  auto packed_weights =
      tflite::testing::PackedSub8BitCustomQuantization<uint16_t>(
          weights_data, 1u * num_weights, num_weights, &packing);
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(0, bias_scale),
      F2Q32(0, bias_scale),
      F2Q32(0, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q((1 * 1 + -2 * 2 + 3 * 3 + -4 * -4 + 5 * 5 + -6 * 6 + 7 * 7 + -8 * 7) /
              (input_sf * weights_sf),
          output_min, output_max),

  };

#if 0
  std::cout << "EXPECTED RAW PRODUCTS ";
  for (size_t i = 0; i < num_weights; ++i) {
    std::cout << (int32_t)(input_data[i] - input_zero_point) << "*"
              << (int32_t)(weights_data[i] - weight_zero_point) << ", ";
  }
  std::cout << std::endl;
  std::cout << "EXPECTED RAW ACCUM " << (int32_t)expected_output_data[0]
            << std::endl;
#endif
  const int output_dims_data[] = {2, 1, 1};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          reinterpret_cast<const uint8_t*>(packed_weights.data()), weights_min,
          weights_max, &packing, bias_dims_data, bias_data, bias_scale,
          expected_output_data, output_dims_data, output_min, output_max,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}


TF_LITE_MICRO_TEST(SmokeTestPackedQuantizedUInt8_6) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;
  using tflite::testing::F2QB;
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_sf = 2.0;
  const float weights_sf = 32.0;
  const float input_min = -128.0f / input_sf;
  const float input_max = 127.0f / input_sf;
  const float weights_min = -32.0f / weights_sf;
  const float weights_max = 31.0f / weights_sf;
  const float bias_scale = 1.0f / (input_sf * weights_sf);
  const float output_min = -128.0f / 32.0f;
  const float output_max = 127.0f / 32.0f;

  const size_t num_weights = 8;
  const int input_dims_data[] = {1, 1, num_weights};
  const uint8_t input_data[] = {
      F2Q(1 / input_sf, input_min, input_max),
      F2Q(-2 / input_sf, input_min, input_max),
      F2Q(3 / input_sf, input_min, input_max),
      F2Q(-4 / input_sf, input_min, input_max),
      F2Q(5 / input_sf, input_min, input_max),
      F2Q(-6 / input_sf, input_min, input_max),
      F2Q(7 / input_sf, input_min, input_max),
      F2Q(-8 / input_sf, input_min, input_max),

  };

  const int weights_dims_data[] = {2, 1, num_weights};
  const uint8_t weights_data[] = {
      F2QB<6>(1 / weights_sf, weights_min, weights_max),
      F2QB<6>(2 / weights_sf, weights_min, weights_max),
      F2QB<6>(3 / weights_sf, weights_min, weights_max),
      F2QB<6>(-4 / weights_sf, weights_min, weights_max),
      F2QB<6>(5 / weights_sf, weights_min, weights_max),
      F2QB<6>(6 / weights_sf, weights_min, weights_max),
      F2QB<6>(7 / weights_sf, weights_min, weights_max),
      F2QB<6>(7 / weights_sf, weights_min, weights_max),
  };

  TfLiteCustomSub8BitPackingDetails packing = {6, 32, 1, {}};

  auto packed_weights =
      tflite::testing::PackedSub8BitCustomQuantization<uint32_t>(
          weights_data, 1u * num_weights, num_weights, &packing);
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(0, bias_scale),
      F2Q32(0, bias_scale),
      F2Q32(0, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q((1 * 1 + -2 * 2 + 3 * 3 + -4 * -4 + 5 * 5 + -6 * 6 + 7 * 7 + -8 * 7) /
              (input_sf * weights_sf),
          output_min, output_max),

  };

#if 0
  std::cout << "EXPECTED RAW PRODUCTS ";
  for (size_t i = 0; i < num_weights; ++i) {
    std::cout << (int32_t)(input_data[i] - input_zero_point) << "*"
              << (int32_t)(weights_data[i] - weight_zero_point) << ", ";
  }
  std::cout << std::endl;
  std::cout << "EXPECTED RAW ACCUM " << (int32_t)expected_output_data[0]
            << std::endl;
#endif
  const int output_dims_data[] = {2, 1, 1};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          reinterpret_cast<const uint8_t*>(packed_weights.data()), weights_min,
          weights_max, &packing, bias_dims_data, bias_data, bias_scale,
          expected_output_data, output_dims_data, output_min, output_max,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

// TODO(b/138811455): Fix code duplication in micro tests
TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -64.0f;
  const float weights_max = 63.5f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {2, 2, 10};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(25, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(59, output_min, output_max), F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<int8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=bits_per_item*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8Relu) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {2, 2, 10};
  const uint8_t input_data[] = {
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
      F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
      F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
      F2Q(1, weights_min, weights_max),  F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max),  F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max),  F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max),  F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max),  F2Q(10, weights_min, weights_max),
      F2Q(-1, weights_min, weights_max), F2Q(-2, weights_min, weights_max),
      F2Q(-3, weights_min, weights_max), F2Q(-4, weights_min, weights_max),
      F2Q(-5, weights_min, weights_max), F2Q(-6, weights_min, weights_max),
      F2Q(-7, weights_min, weights_max), F2Q(-8, weights_min, weights_max),
      F2Q(-9, weights_min, weights_max), F2Q(-10, weights_min, weights_max),
      F2Q(1, weights_min, weights_max),  F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max),  F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max),  F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max),  F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max),  F2Q(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(0, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(0, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(0, output_min, output_max),  F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=bits_per_item*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActRelu,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8Relu) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -64.0f;
  const float weights_max = 63.5f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {2, 2, 10};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max),  F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max),  F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max),  F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max),  F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max),  F2QS(10, weights_min, weights_max),
      F2QS(-1, weights_min, weights_max), F2QS(-2, weights_min, weights_max),
      F2QS(-3, weights_min, weights_max), F2QS(-4, weights_min, weights_max),
      F2QS(-5, weights_min, weights_max), F2QS(-6, weights_min, weights_max),
      F2QS(-7, weights_min, weights_max), F2QS(-8, weights_min, weights_max),
      F2QS(-9, weights_min, weights_max), F2QS(-10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max),  F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max),  F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max),  F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max),  F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max),  F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(0, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(0, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(0, output_min, output_max),  F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<int8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=bits_per_item*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActRelu,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8OutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -127.0f;
  const float weights_max = 128.0f;
  const float bias_scale = 1.0f;
  const float output_min = -63.5f;
  const float output_max = 64.0f;

  const int input_dims_data[] = {2, 2, 10};
  const uint8_t input_data[] = {
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
      F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
      F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(25, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(59, output_min, output_max), F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=bits_per_item*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8OutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -63.5f;
  const float output_max = 64.0f;

  const int input_dims_data[] = {2, 2, 10};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(25, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(59, output_min, output_max), F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<int8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=bits_per_item*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInput) {
  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const float input_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  };
  const int weights_dims_data[] = {2, 3, 10};
  const float weights_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  };
  const int bias_dims_data[] = {1, 3};
  const float bias_data[] = {1, 2, 3};
  const float expected_output_data[] = {
      24, 25, 26, 58, 59, 60,  // Expected results.
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  float output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          input_dims_data, input_data, weights_dims_data, weights_data,
          bias_dims_data, bias_data, expected_output_data, output_dims_data,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedUInt8) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const uint8_t input_data[] = {
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
      F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
      F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(25, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(59, output_min, output_max), F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=bits_per_item*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedInt8) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -64.0f;
  const float weights_max = 63.5f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(25, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(59, output_min, output_max), F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<int8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=bits_per_item*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(
    SimpleTest4DInputQuantizedUInt8OutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -127.0f;
  const float weights_max = 128.0f;
  const float bias_scale = 1.0f;
  const float output_min = -63.5f;
  const float output_max = 64.0f;

  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const uint8_t input_data[] = {
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
      F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
      F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
      F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
      F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
      F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
      F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
      F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
      F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
      F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(25, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(59, output_min, output_max), F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=bits_per_item*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedInt8OutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -63.5f;
  const float output_max = 64.0f;

  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(25, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(59, output_min, output_max), F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized<int8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, 0 /*=bits_per_item*/,
          bias_dims_data, bias_data, bias_scale, expected_output_data,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TESTS_END
