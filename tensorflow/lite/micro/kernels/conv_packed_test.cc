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

#include <vector>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

// Common inputs and outputs.
static const int kInputElements = 64;
static const int kInputShape[] = {4, 2, 2, 4, 4};
static const float kInputData[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                   1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                                   1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,};
static const int kFilterElements = 48;
static const int kFilterShape[] = {4, 3, 2, 2, 4};
static const float kFilterData[] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                                    -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1};
static const int kBiasElements = 3;
static const int kBiasShape[] = {1, 3};
static const float kBiasData[] = {1, 2, 3};
static const int kOutputElements = 12;
static const int kOutputShape[] = {4, 2, 1, 2, 3};
static const float kGoldenData[] = {69, 2, 11, 69, 2, 11, 65, 10, 3, 145, 10, 3};

static TfLiteConvParams common_conv_params = {
    kTfLitePaddingValid,  // padding
    2,                    // stride_width
    2,                    // stride_height
    kTfLiteActNone,       // activation
    1,                    // dilation_width_factor
    1,                    // dilation_height_factor
};

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
  uint32_t items_per_container = std::floor((float)(container_bits)/bits_per_item);
  uint32_t cont_per_minor = std::ceil((float)(minor_dim_size)/items_per_container);
  uint32_t number_containers = (elts / minor_dim_size) * cont_per_minor;
  std::vector<CONTAINER_T> packed_data(number_containers);

  uint8_t* packed_data_byte = reinterpret_cast<uint8_t*>(packed_data.data());
  int bits_in_container = 0;
  for (size_t dim = 0; dim < elts / minor_dim_size; dim++) {
    uint32_t data_index = dim * minor_dim_size;
    for (size_t i = 0; i < minor_dim_size; ++i) {
      // Little-endian packing...
      container_buf |= (static_cast<CONTAINER_T>(data[data_index + i]) & mask) << bits_in_container;
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


template <typename T>
TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const T* expected_output_data, T* output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 float tolerance = 1e-5) {
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_CONV_2D();
  micro::KernelRunner runner(
      registration, tensors, tensors_size, inputs_array, outputs_array,
      reinterpret_cast<void*>(conv_params), micro_test::reporter);

  const char* init_data = reinterpret_cast<const char*>(conv_params);

  // TODO(b/154240825): Use a test macro here which fails and returns.
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}


void TestConvQuantizedPerLayer(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, const int* filter_dims_data,
    const uint8_t* filter_quantized, float filter_scale, int filter_zero_point,
    float weights_min, float weights_max, TfLiteCustomSub8BitPackingDetails* packing,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const int* output_dims_data, const float* expected_output_data,
    uint8_t* expected_output_quantized, uint8_t* output_data,
    float output_scale, TfLiteConvParams* conv_params) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  tflite::AsymmetricQuantize(expected_output_data, expected_output_quantized,
                             output_dims_count, output_scale, 128);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, 128),
      CreateQuantizedTensor(filter_quantized, filter_dims,
                            filter_scale, filter_zero_point),
      CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                input_scale, filter_scale),
      CreateQuantizedTensor(output_data, output_dims, output_scale, 128)};

  float filter_scales[] = {1, filter_scale};
  int filter_zero_points[] = {1, filter_zero_point};
  TfLiteAffineQuantization filter_quant = {
      FloatArrayFromFloats(filter_scales),
      IntArrayFromInts(filter_zero_points), 
      0};
  // TF_LITE_PACKED_QUANTIZED_DATA support
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant, {}};

  if (packing) {
        SetPackingParams(tensors[1], weights_min, weights_max, packing);
  }

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateConvGoldens(tensors, tensors_size, expected_output_quantized,
                          output_data, output_dims_count, conv_params));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTestQuantizedPackedWeights4Bit) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const int output_dims_count = 12;
  uint8_t output_data[output_dims_count];

  const float weights_min = -3.5f;
  const float weights_max = 4.0f;

  const int weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);

  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 3 /* Packed dimension needs to be 3 */, {}};

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;

  uint8_t input_quantized[tflite::testing::kInputElements];
  uint8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  uint8_t golden_quantized[tflite::testing::kOutputElements];


  tflite::AsymmetricQuantize(tflite::testing::kFilterData, filter_quantized,
                             tflite::testing::kFilterElements, filter_scale, weight_zero_point);

  auto packed_weights =
        tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
            filter_quantized, 1u * tflite::testing::kFilterElements, 4, &packing);


  tflite::testing::TestConvQuantizedPerLayer(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, tflite::testing::kFilterShape,
      reinterpret_cast<const uint8_t*>(packed_weights.data()),
      filter_scale, weight_zero_point, weights_min, weights_max, &packing,
      tflite::testing::kBiasShape, tflite::testing::kBiasData, bias_quantized,
      tflite::testing::kOutputShape, tflite::testing::kGoldenData,
      golden_quantized, output_data, output_scale,
      &tflite::testing::common_conv_params);

}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPackedWeights5Bit) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const int output_dims_count = 12;
  uint8_t output_data[output_dims_count];
  
  const float weights_min = -8.f;
  const float weights_max = 7.5f;

  const int weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 5);

  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 5);

  TfLiteCustomSub8BitPackingDetails packing = {5, 16, 3 /* Packed dimension needs to be 3 */, {}};

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;

  uint8_t input_quantized[tflite::testing::kInputElements];
  uint8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  uint8_t golden_quantized[tflite::testing::kOutputElements];


  tflite::AsymmetricQuantize(tflite::testing::kFilterData, filter_quantized,
                             tflite::testing::kFilterElements, filter_scale, weight_zero_point);

  auto packed_weights =
        tflite::testing::PackedSub8BitCustomQuantization<uint16_t>(
            filter_quantized, 1u * tflite::testing::kFilterElements, 4, &packing);


  tflite::testing::TestConvQuantizedPerLayer(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, tflite::testing::kFilterShape,
      reinterpret_cast<const uint8_t*>(packed_weights.data()),
      filter_scale, weight_zero_point, weights_min, weights_max, &packing,
      tflite::testing::kBiasShape, tflite::testing::kBiasData, bias_quantized,
      tflite::testing::kOutputShape, tflite::testing::kGoldenData,
      golden_quantized, output_data, output_scale,
      &tflite::testing::common_conv_params);

}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPackedWeights6Bit) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const int output_dims_count = 12;
  uint8_t output_data[output_dims_count];

  const float weights_min = -16.f;
  const float weights_max = 15.5f;

  const int weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 6);

  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 6);

  TfLiteCustomSub8BitPackingDetails packing = {6, 32, 3 /* Packed dimension needs to be 3 */, {}};

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;

  uint8_t input_quantized[tflite::testing::kInputElements];
  uint8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  uint8_t golden_quantized[tflite::testing::kOutputElements];


  tflite::AsymmetricQuantize(tflite::testing::kFilterData, filter_quantized,
                             tflite::testing::kFilterElements, filter_scale, weight_zero_point);

  auto packed_weights =
        tflite::testing::PackedSub8BitCustomQuantization<uint32_t>(
            filter_quantized, 1u * tflite::testing::kFilterElements, 4, &packing);


  tflite::testing::TestConvQuantizedPerLayer(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, tflite::testing::kFilterShape,
      reinterpret_cast<const uint8_t*>(packed_weights.data()),
      filter_scale, weight_zero_point, weights_min, weights_max, &packing,
      tflite::testing::kBiasShape, tflite::testing::kBiasData, bias_quantized,
      tflite::testing::kOutputShape, tflite::testing::kGoldenData,
      golden_quantized, output_data, output_scale,
      &tflite::testing::common_conv_params);

}

TF_LITE_MICRO_TEST(TestConv4BitWithPadding) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float weights_min = -3.5f;
  const float weights_max = 4.0f;

  const int weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);

  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 3 /* Packed dimension needs to be 3 */, {}};

  TfLiteConvParams padding_conv_params = {
        kTfLitePaddingSame,   // padding
        1,                    // stride_width
        1,                    // stride_height
        kTfLiteActNone,       // activation
        1,                    // dilation_width_factor
        1,                    // dilation_height_factor
    };

    const int kInputElements = 9;
    const int kInputShape[] = {4, 1, 3, 3, 1};
    const float kInputData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    const int kFilterElements = 9;
    const int kFilterShape[] = {4, 1, 3, 3, 1};
    const float kFilterData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    const int kBiasElements = 1;
    const int kBiasShape[] = {1, 1};
    const float kBiasData[] = {1};
    const int kOutputShape[] = {4, 1, 3, 3, 1};
    const float kGoldenData[] = {17, 25, 17, 29, 43, 29, 17, 25, 17};

    const int output_dims_count = 9;
    uint8_t output_data[output_dims_count];

    const float input_scale = 0.5f;
    const float output_scale = 1.0f;

    uint8_t input_quantized[kInputElements];
    uint8_t filter_quantized[kFilterElements];
    int32_t bias_quantized[kBiasElements];
    uint8_t golden_quantized[output_dims_count];

    tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                                 kFilterElements, filter_scale, weight_zero_point);

    auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
              filter_quantized, 1u * kFilterElements, 1, &packing);


    tflite::testing::TestConvQuantizedPerLayer(
        kInputShape, kInputData,
        input_quantized, input_scale, kFilterShape,
        reinterpret_cast<const uint8_t*>(packed_weights.data()),
        filter_scale, weight_zero_point, weights_min, weights_max, &packing,
        kBiasShape, kBiasData, bias_quantized,
        kOutputShape, kGoldenData,
        golden_quantized, output_data, output_scale,
        &padding_conv_params);
}

TF_LITE_MICRO_TEST(TestConv5BitWithPadding) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float weights_min = -1.5f;
  const float weights_max = 14.0f;  // Test weird values here

  const int weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 5);

  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 5);

  TfLiteCustomSub8BitPackingDetails packing = {5, 16, 3 /* Packed dimension needs to be 3 */, {}};

  TfLiteConvParams padding_conv_params = {
        kTfLitePaddingSame,   // padding
        1,                    // stride_width
        1,                    // stride_height
        kTfLiteActNone,       // activation
        1,                    // dilation_width_factor
        1,                    // dilation_height_factor
    };

    const int kInputElements = 9;
    const int kInputShape[] = {4, 1, 3, 3, 1};
    const float kInputData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    const int kFilterElements = 9;
    const int kFilterShape[] = {4, 1, 3, 3, 1};
    const float kFilterData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    const int kBiasElements = 1;
    const int kBiasShape[] = {1, 1};
    const float kBiasData[] = {1};
    const int kOutputShape[] = {4, 1, 3, 3, 1};
    const float kGoldenData[] = {17, 25, 17, 29, 43, 29, 17, 25, 17};

    const int output_dims_count = 9;
    uint8_t output_data[output_dims_count];

    const float input_scale = 0.5f;
    const float output_scale = 1.0f;

    uint8_t input_quantized[kInputElements];
    uint8_t filter_quantized[kFilterElements];
    int32_t bias_quantized[kBiasElements];
    uint8_t golden_quantized[output_dims_count];

    tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                                 kFilterElements, filter_scale, weight_zero_point);

    auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint16_t>(
              filter_quantized, 1u * kFilterElements, 1, &packing);


    tflite::testing::TestConvQuantizedPerLayer(
        kInputShape, kInputData,
        input_quantized, input_scale, kFilterShape,
        reinterpret_cast<const uint8_t*>(packed_weights.data()),
        filter_scale, weight_zero_point, weights_min, weights_max, &packing,
        kBiasShape, kBiasData, bias_quantized,
        kOutputShape, kGoldenData,
        golden_quantized, output_data, output_scale,
        &padding_conv_params);
}

TF_LITE_MICRO_TEST(TestConv6BitWithPadding) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float weights_min = -15.5f;
  const float weights_max = 16.0f;

  const int weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 6);

  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 6);

  TfLiteCustomSub8BitPackingDetails packing = {6, 32, 3 /* Packed dimension needs to be 3 */, {}};

  TfLiteConvParams padding_conv_params = {
        kTfLitePaddingSame,   // padding
        1,                    // stride_width
        1,                    // stride_height
        kTfLiteActNone,       // activation
        1,                    // dilation_width_factor
        1,                    // dilation_height_factor
    };

    const int kInputElements = 9;
    const int kInputShape[] = {4, 1, 3, 3, 1};
    const float kInputData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    const int kFilterElements = 9;
    const int kFilterShape[] = {4, 1, 3, 3, 1};
    const float kFilterData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    const int kBiasElements = 1;
    const int kBiasShape[] = {1, 1};
    const float kBiasData[] = {1};
    const int kOutputShape[] = {4, 1, 3, 3, 1};
    const float kGoldenData[] = {17, 25, 17, 29, 43, 29, 17, 25, 17};

    const int output_dims_count = 9;
    uint8_t output_data[output_dims_count];

    const float input_scale = 0.5f;
    const float output_scale = 1.0f;

    uint8_t input_quantized[kInputElements];
    uint8_t filter_quantized[kFilterElements];
    int32_t bias_quantized[kBiasElements];
    uint8_t golden_quantized[output_dims_count];

    tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                                 kFilterElements, filter_scale, weight_zero_point);

    auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint32_t>(
              filter_quantized, 1u * kFilterElements, 1, &packing);


    tflite::testing::TestConvQuantizedPerLayer(
        kInputShape, kInputData,
        input_quantized, input_scale, kFilterShape,
        reinterpret_cast<const uint8_t*>(packed_weights.data()),
        filter_scale, weight_zero_point, weights_min, weights_max, &packing,
        kBiasShape, kBiasData, bias_quantized,
        kOutputShape, kGoldenData,
        golden_quantized, output_data, output_scale,
        &padding_conv_params);
}

TF_LITE_MICRO_TEST(TestConv4BitWithReluActivation) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float weights_min = -4.0f;
  const float weights_max = 3.5f;

  const int weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);

  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 3 /* Packed dimension needs to be 3 */, {}};

  TfLiteConvParams relu_conv_params = {
        kTfLitePaddingValid,   // padding
        1,                    // stride_width
        1,                    // stride_height
        kTfLiteActRelu,       // activation
        1,                    // dilation_width_factor
        1,                    // dilation_height_factor
    };

    const int kInputElements = 18;
    const int kInputShape[] = {4, 1, 3, 3, 2};
    const float kInputData[] = {1, 1, 1, 1, 1, 1,
                                2, -4, 2, -4, 2, -4,
                                3, 3, 3, 1, 1, 1};
    const int kFilterElements = 2;
    const int kFilterShape[] = {4, 1, 1, 1, 2};
    const float kFilterData[] = {1, 1};
    const int kBiasElements = 1;
    const int kBiasShape[] = {1, 1,};
    const float kBiasData[] = {1};
    const int kOutputShape[] = {4, 1, 3, 3, 1};
    const float kGoldenData[] = {3, 3, 3, 0, 0, 0, 7, 5, 3}; // Three zeros because of relu

    const int output_dims_count = 9;
    uint8_t output_data[output_dims_count];

    const float input_scale = 0.5f;
    const float output_scale = 1.0f;

    uint8_t input_quantized[kInputElements];
    uint8_t filter_quantized[kFilterElements];
    int32_t bias_quantized[kBiasElements];
    uint8_t golden_quantized[output_dims_count];

    tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                                 kFilterElements, filter_scale, weight_zero_point);

    auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
              filter_quantized, 1u * kFilterElements, 2, &packing);


    tflite::testing::TestConvQuantizedPerLayer(
        kInputShape, kInputData,
        input_quantized, input_scale, kFilterShape,
        reinterpret_cast<const uint8_t*>(packed_weights.data()),
        filter_scale, weight_zero_point, weights_min, weights_max, &packing,
        kBiasShape, kBiasData, bias_quantized,
        kOutputShape, kGoldenData,
        golden_quantized, output_data, output_scale,
        &relu_conv_params);
}

TF_LITE_MICRO_TEST(TestConv4BitWithDilation) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float weights_min = -4.0f;
  const float weights_max = 3.5f;

  const int weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);

  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 3 /* Packed dimension needs to be 3 */, {}};

  TfLiteConvParams relu_conv_params = {
        kTfLitePaddingValid,   // padding
        1,                    // stride_width
        1,                    // stride_height
        kTfLiteActRelu,       // activation
        2,                    // dilation_width_factor
        1,                    // dilation_height_factor
    };

    const int kInputElements = 16;
    const int kInputShape[] = {4, 1, 2, 4, 2};
    const float kInputData[] = {1, 1, 2, 2, -1, -1, 1, 3,
                                1, 3, -2, -2, 2, 2, -3, 0};
    const int kFilterElements = 8;
    const int kFilterShape[] = {4, 1, 2, 2, 2};
    const float kFilterData[] = {1, 2, 3, -1,
                                 -3, 2, 1, 1};
    const int kBiasElements = 1;
    const int kBiasShape[] = {1, 1,};
    const float kBiasData[] = {1};
    const int kOutputShape[] = {4, 1, 1, 2, 1};
    const float kGoldenData[] = {9, 6};

    const int output_dims_count = 2;
    uint8_t output_data[output_dims_count];

    const float input_scale = 0.5f;
    const float output_scale = 1.0f;

    uint8_t input_quantized[kInputElements];
    uint8_t filter_quantized[kFilterElements];
    int32_t bias_quantized[kBiasElements];
    uint8_t golden_quantized[output_dims_count];

    tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                                 kFilterElements, filter_scale, weight_zero_point);

    auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
              filter_quantized, 1u * kFilterElements, 2, &packing);


    tflite::testing::TestConvQuantizedPerLayer(
        kInputShape, kInputData,
        input_quantized, input_scale, kFilterShape,
        reinterpret_cast<const uint8_t*>(packed_weights.data()),
        filter_scale, weight_zero_point, weights_min, weights_max, &packing,
        kBiasShape, kBiasData, bias_quantized,
        kOutputShape, kGoldenData,
        golden_quantized, output_data, output_scale,
        &relu_conv_params);
}

TF_LITE_MICRO_TESTS_END
