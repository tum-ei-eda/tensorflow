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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

#include <iostream>

namespace tflite {
namespace testing {
namespace {

// Common inputs and outputs.
static const int kInputElements = 64;
static const int kInputShape[] = {4, 2, 2, 4, 4};
static const float kInputData[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                   1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                                   1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
static const int kFilterElements = 32;
static const int kFilterShape[] = {4, 1, 2, 2, 8};
static const float kFilterData[] = {
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4,
    -1, -1, -1, -1, 1, 1, 1, 1,
    -1, -1, -1, -1, 1, 1, 1, 1};
static const int kBiasElements = 8;
static const int kBiasShape[] = {1, 8};
static const float kBiasData[] = {1, 1, 1, 2, 2, 2, 3, 3};
static const int kOutputElements = 48;
static const int kOutputShape[] = {4, 2, 1, 3, 8};
static const float kGoldenData[] = {
    1, 1, 1, 2, 12, 12, 13, 13,
    1, 1, 1, 2, 12, 12, 13, 13,
    1, 1, 1, 2, 12, 12, 13, 13,
    5, 5, 5, 6, 15, 15, 16, 16,
    7, 7, 7, 8, 23, 23, 24, 24,
    9, 9, 9, 10, 31, 31, 32, 32
};



constexpr int kMaxFilterChannels = 64;
constexpr int kMaxBiasChannels = 64;

// Index of the output tensor in context->tensors, specific to
// DepthwiseConv.
constexpr int kOutputTensorIndex = 3;

static TfLiteDepthwiseConvParams common_depthwise_conv_params = {
    kTfLitePaddingValid,  /* Padding */
    1,                    /* Stride Width */
    1,                    /* Stride Height */
    2,                    /* Depth Multiplier */
    kTfLiteActNone,       /* Activation*/
    1,                    /* Dilation Width*/
    1                     /* Dilation Height */
};

// TODO Factor out into support library
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
      // TF_LITE_PACKED_QUANTIZED_DATA support
      // TODO: probably more efficient to align on selected dimension
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


// Creates a DepthwiseConv opeerator, calls it with the provided input tensors
// and some defaults parameters, and compares the output with
// expected_output_data.
//
// The tensors parameter contains both the input tensors as well as a
// preallocated output tensor into which the output is stored.
template <typename T>
TfLiteStatus ValidateDepthwiseConvGoldens(
    const T* expected_output_data, int output_length,
    TfLiteDepthwiseConvParams* conv_params, float tolerance, int tensors_size,
                                          TfLiteTensor* tensors) {
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D();
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

  const T* output_data = tflite::GetTensorData<T>(&tensors[kOutputTensorIndex]);
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}




void TestDepthwiseConvQuantizedPerLayer(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data,
    const uint8_t* filter_quantized, float filter_scale, int filter_zero_point,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const float* golden, uint8_t* golden_quantized, const int* output_dims_data,
    uint8_t* output_data, float output_scale, int output_zero_point,
    TfLiteDepthwiseConvParams params, TfLiteCustomSub8BitPackingDetails* packing,
    float weights_min, float weights_max) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input_data, input_quantized,
                                             input_dims, input_scale,
                                             input_zero_point),
      tflite::testing::CreateQuantizedTensor(filter_quantized,
                                             filter_dims, filter_scale,
                                             filter_zero_point),
      tflite::testing::CreateQuantizedBiasTensor(
          bias_data, bias_quantized, bias_dims, input_scale, filter_scale),
      tflite::testing::CreateQuantizedTensor(output_data, output_dims,
                                             output_scale, output_zero_point),
  };

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
  float filter_scales[] = {1, filter_scale};
  int filter_zero_points[] = {1, 128};
  TfLiteAffineQuantization filter_quant = {
      FloatArrayFromFloats(filter_scales),
      IntArrayFromInts(filter_zero_points),
      0};
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant, {kTfLiteNoDetails, {}}};

  float bias_scales[] = {1, filter_scale * input_scale};
  int bias_zero_points[] = {1, 128};
  TfLiteAffineQuantization bias_quant = {FloatArrayFromFloats(bias_scales),
                                         IntArrayFromInts(bias_zero_points), 0};
  tensors[2].quantization = {kTfLiteAffineQuantization, &bias_quant, {kTfLiteNoDetails, {}}};

  if (packing) {
          SetPackingParams(tensors[1], weights_min, weights_max, packing);
  }

  AsymmetricQuantize(golden, golden_quantized, output_dims_count, output_scale,
                     output_zero_point);
  ValidateDepthwiseConvGoldens(golden_quantized, output_dims_count, &params,
                               1.0, tensors_size, tensors);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(DepthwiseConvQuantizedPackedWeights4Bit) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;
  float weights_min = -3.5f;
  float weights_max = 4.0f;

  const int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);
  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1 /* Packed dimension needs to be 1 */, {}};

  uint8_t input_quantized[tflite::testing::kInputElements];
  uint8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  uint8_t golden_quantized[tflite::testing::kOutputElements];
  uint8_t output_data[tflite::testing::kOutputElements];

  tflite::AsymmetricQuantize(tflite::testing::kFilterData, filter_quantized,
                               tflite::testing::kFilterElements, filter_scale, filter_zero_point);

  auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
              filter_quantized, 1u * tflite::testing::kFilterElements, 8, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      tflite::testing::kInputShape, tflite::testing::kInputData, input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, reinterpret_cast<const uint8_t*>(packed_weights.data()), filter_scale,
      filter_zero_point, tflite::testing::kBiasShape, tflite::testing::kBiasData, bias_quantized, tflite::testing::kGoldenData,
      golden_quantized, tflite::testing::kOutputShape, output_data, output_scale,
      output_zero_point, tflite::testing::common_depthwise_conv_params, &packing, weights_min, weights_max);
}

TF_LITE_MICRO_TEST(DepthwiseConvQuantizedPackedWeights5Bit) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;
  float weights_min = -7.5f;
  float weights_max = 8.0f;

  const int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max,5);
  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 5);

  TfLiteCustomSub8BitPackingDetails packing = {5, 16, 1 /* Packed dimension needs to be 1 */, {}};

  uint8_t input_quantized[tflite::testing::kInputElements];
  uint8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  uint8_t golden_quantized[tflite::testing::kOutputElements];
  uint8_t output_data[tflite::testing::kOutputElements];

  tflite::AsymmetricQuantize(tflite::testing::kFilterData, filter_quantized,
                               tflite::testing::kFilterElements, filter_scale, filter_zero_point);

  auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint16_t>(
              filter_quantized, 1u * tflite::testing::kFilterElements, 8, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      tflite::testing::kInputShape, tflite::testing::kInputData, input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, reinterpret_cast<const uint8_t*>(packed_weights.data()), filter_scale,
      filter_zero_point, tflite::testing::kBiasShape, tflite::testing::kBiasData, bias_quantized, tflite::testing::kGoldenData,
      golden_quantized, tflite::testing::kOutputShape, output_data, output_scale,
      output_zero_point, tflite::testing::common_depthwise_conv_params, &packing, weights_min, weights_max);
}

TF_LITE_MICRO_TEST(DepthwiseConvQuantizedPackedWeights6Bit) {
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;
  float weights_min = -15.5f;
  float weights_max = 16.0f;

  const int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 6);
  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 6);

  TfLiteCustomSub8BitPackingDetails packing = {6, 32, 1 /* Packed dimension needs to be 1 */, {}};

  uint8_t input_quantized[tflite::testing::kInputElements];
  uint8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  uint8_t golden_quantized[tflite::testing::kOutputElements];
  uint8_t output_data[tflite::testing::kOutputElements];

  tflite::AsymmetricQuantize(tflite::testing::kFilterData, filter_quantized,
                               tflite::testing::kFilterElements, filter_scale, filter_zero_point);

  auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint32_t>(
              filter_quantized, 1u * tflite::testing::kFilterElements, 8, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      tflite::testing::kInputShape, tflite::testing::kInputData, input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, reinterpret_cast<const uint8_t*>(packed_weights.data()), filter_scale,
      filter_zero_point, tflite::testing::kBiasShape, tflite::testing::kBiasData, bias_quantized, tflite::testing::kGoldenData,
      golden_quantized, tflite::testing::kOutputShape, output_data, output_scale,
      output_zero_point, tflite::testing::common_depthwise_conv_params, &packing, weights_min, weights_max);
}

TF_LITE_MICRO_TEST(DepthwiseConv4BitWithPadding) {
  const int input_elements = 9;
  const int input_shape[] = {4, 1, 3, 3, 1};
  const float input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
  const int filter_elements = 18;
  const int filter_shape[] = {4, 1, 3, 3, 2};
  const float filter_values[] = {1, 4, 1, 4, 1, 4, 2, 5, 2,
                                5, 2, 5, 3, 6, 3, 6, 3, 6};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 18;
  const float bias_values[] = {1, 2};
  const float golden[] = {17, 36, 25, 53, 17, 36,
                          29, 66, 43, 98, 29, 66,
                          17, 48, 25, 71, 17, 48};

  const int output_shape[] = {4, 1, 3, 3, 2};

  uint8_t input_quantized[input_elements];
  uint8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  uint8_t golden_quantized[output_elements];
  uint8_t output_data[output_elements];

  static TfLiteDepthwiseConvParams padding_depthwise_conv_params = {
      kTfLitePaddingSame,  /* Padding */
      1,                    /* Stride Width */
      1,                    /* Stride Height */
      2,                    /* Depth Multiplier */
      kTfLiteActNone,       /* Activation*/
      1,                    /* Dilation Width*/
      1                     /* Dilation Height */
  };

  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;
  float weights_min = -1.5f;
  float weights_max = 6.0f;

  const int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);
  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1 /* Packed dimension needs to be 1 */, {}};

  tflite::AsymmetricQuantize(filter_values, filter_quantized,
                             filter_elements, filter_scale, filter_zero_point);

  auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
              filter_quantized, 1u * filter_elements, 2, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, reinterpret_cast<const uint8_t*>(packed_weights.data()), filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, padding_depthwise_conv_params, &packing, weights_min, weights_max);
}

TF_LITE_MICRO_TEST(DepthwiseConv5BitWithPadding) {
  const int input_elements = 9;
  const int input_shape[] = {4, 1, 3, 3, 1};
  const float input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
  const int filter_elements = 18;
  const int filter_shape[] = {4, 1, 3, 3, 2};
  const float filter_values[] = {1, 4, 1, 4, 1, 4, 2, 5, 2,
                                5, 2, 5, 3, 6, 3, 6, 3, 6};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 18;
  const float bias_values[] = {1, 2};
  const float golden[] = {17, 36, 25, 53, 17, 36,
                          29, 66, 43, 98, 29, 66,
                          17, 48, 25, 71, 17, 48};

  const int output_shape[] = {4, 1, 3, 3, 2};

  uint8_t input_quantized[input_elements];
  uint8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  uint8_t golden_quantized[output_elements];
  uint8_t output_data[output_elements];

  static TfLiteDepthwiseConvParams padding_depthwise_conv_params = {
      kTfLitePaddingSame,  /* Padding */
      1,                    /* Stride Width */
      1,                    /* Stride Height */
      2,                    /* Depth Multiplier */
      kTfLiteActNone,       /* Activation*/
      1,                    /* Dilation Width*/
      1                     /* Dilation Height */
  };

  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;
  float weights_min = -1.75f;
  float weights_max = 6.0f;

  const int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 5);
  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 5);

  TfLiteCustomSub8BitPackingDetails packing = {5, 16, 1 /* Packed dimension needs to be 1 */, {}};

  tflite::AsymmetricQuantize(filter_values, filter_quantized,
                             filter_elements, filter_scale, filter_zero_point);

  auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint16_t>(
              filter_quantized, 1u * filter_elements, 2, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, reinterpret_cast<const uint8_t*>(packed_weights.data()), filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, padding_depthwise_conv_params, &packing, weights_min, weights_max);
}

TF_LITE_MICRO_TEST(DepthwiseConv6BitWithPadding) {
  const int input_elements = 9;
  const int input_shape[] = {4, 1, 3, 3, 1};
  const float input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
  const int filter_elements = 18;
  const int filter_shape[] = {4, 1, 3, 3, 2};
  const float filter_values[] = {1, 4, 1, 4, 1, 4, 2, 5, 2,
                                5, 2, 5, 3, 6, 3, 6, 3, 6};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 18;
  const float bias_values[] = {1, 2};
  const float golden[] = {17, 36, 25, 53, 17, 36,
                          29, 66, 43, 98, 29, 66,
                          17, 48, 25, 71, 17, 48};

  const int output_shape[] = {4, 1, 3, 3, 2};

  uint8_t input_quantized[input_elements];
  uint8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  uint8_t golden_quantized[output_elements];
  uint8_t output_data[output_elements];

  static TfLiteDepthwiseConvParams padding_depthwise_conv_params = {
      kTfLitePaddingSame,  /* Padding */
      1,                    /* Stride Width */
      1,                    /* Stride Height */
      2,                    /* Depth Multiplier */
      kTfLiteActNone,       /* Activation*/
      1,                    /* Dilation Width*/
      1                     /* Dilation Height */
  };

  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;
  float weights_min = -1.875f;
  float weights_max = 6.0f;

  const int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 6);
  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 6);

  TfLiteCustomSub8BitPackingDetails packing = {6, 32, 1 /* Packed dimension needs to be 1 */, {}};

  tflite::AsymmetricQuantize(filter_values, filter_quantized,
                             filter_elements, filter_scale, filter_zero_point);

  auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint32_t>(
              filter_quantized, 1u * filter_elements, 2, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, reinterpret_cast<const uint8_t*>(packed_weights.data()), filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, padding_depthwise_conv_params, &packing, weights_min, weights_max);
}

TF_LITE_MICRO_TEST(DepthwiseConv4BitWithStride2) {
  const int input_elements = 16;
  const int input_shape[] = {4, 1, 4, 4, 1};
  const float input_values[] = {1, 1, 1, 1,
                                2, 2, 2, 2,
                                3, 3, 3, 3,
                                4, 4, 4, 4};
  const int filter_elements = 8;
  const int filter_shape[] = {4, 1, 2, 2, 2};
  const float filter_values[] = {1, 4, 1, 4,
                                 -3, 2, -3, 3};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 18;
  const float bias_values[] = {1, 2};
  const float golden[] = {-9, 20, -9, 20, -17, 46, -17, 46};

  const int output_shape[] = {4, 1, 2, 2, 2};

  uint8_t input_quantized[input_elements];
  uint8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  uint8_t golden_quantized[output_elements];
  uint8_t output_data[output_elements];

  static TfLiteDepthwiseConvParams stride_depthwise_conv_params = {
      kTfLitePaddingValid,  /* Padding */
      2,                    /* Stride Width */
      2,                    /* Stride Height */
      2,                    /* Depth Multiplier */
      kTfLiteActNone,       /* Activation*/
      1,                    /* Dilation Width*/
      1                     /* Dilation Height */
  };

  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;
  float weights_min = -3.5f;
  float weights_max = 4.0f;

  const int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);
  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1 /* Packed dimension needs to be 1 */, {}};

  tflite::AsymmetricQuantize(filter_values, filter_quantized,
                             filter_elements, filter_scale, filter_zero_point);

  auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
              filter_quantized, 1u * filter_elements, 2, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, reinterpret_cast<const uint8_t*>(packed_weights.data()), filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, stride_depthwise_conv_params, &packing, weights_min, weights_max);
}

TF_LITE_MICRO_TEST(DepthwiseConv4BitWithStride2Relu) {
  const int input_elements = 16;
  const int input_shape[] = {4, 1, 4, 4, 1};
  const float input_values[] = {1, 1, 1, 1,
                                2, 2, 2, 2,
                                3, 3, 3, 3,
                                4, 4, 4, 4};
  const int filter_elements = 8;
  const int filter_shape[] = {4, 1, 2, 2, 2};
  const float filter_values[] = {1, 4, 1, 4,
                                 -3, 2, -3, 3};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 18;
  const float bias_values[] = {1, 2};
  const float golden[] = {0, 20, 0, 20, 0, 46, 0, 46};

  const int output_shape[] = {4, 1, 2, 2, 2};

  uint8_t input_quantized[input_elements];
  uint8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  uint8_t golden_quantized[output_elements];
  uint8_t output_data[output_elements];

  static TfLiteDepthwiseConvParams stride_depthwise_conv_params = {
      kTfLitePaddingValid,  /* Padding */
      2,                    /* Stride Width */
      2,                    /* Stride Height */
      2,                    /* Depth Multiplier */
      kTfLiteActRelu,       /* Activation*/
      1,                    /* Dilation Width*/
      1                     /* Dilation Height */
  };

  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;
  float weights_min = -3.5f;
  float weights_max = 4.0f;

  const int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);
  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1 /* Packed dimension needs to be 1 */, {}};

  tflite::AsymmetricQuantize(filter_values, filter_quantized,
                             filter_elements, filter_scale, filter_zero_point);

  auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
              filter_quantized, 1u * filter_elements, 2, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, reinterpret_cast<const uint8_t*>(packed_weights.data()), filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, stride_depthwise_conv_params, &packing, weights_min, weights_max);
}

TF_LITE_MICRO_TEST(DepthwiseConv4BitWithDilation2) {
  const int input_elements = 16;
  const int input_shape[] = {4, 1, 4, 4, 1};
  const float input_values[] = {1, 1, 1, 1,
                                2, 2, 2, 2,
                                3, 3, 3, 3,
                                4, 4, 4, 4};
  const int filter_elements = 8;
  const int filter_shape[] = {4, 1, 2, 2, 2};
  const float filter_values[] = {1, 4, 1, 4,
                                 -3, 2, -3, 3};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 18;
  const float bias_values[] = {1, 2};
  const float golden[] = {-15, 25, -15, 25, -19, 38, -19, 38};

  const int output_shape[] = {4, 1, 2, 2, 2};

  uint8_t input_quantized[input_elements];
  uint8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  uint8_t golden_quantized[output_elements];
  uint8_t output_data[output_elements];

  static TfLiteDepthwiseConvParams dilation_depthwise_conv_params = {
      kTfLitePaddingValid,  /* Padding */
      1,                    /* Stride Width */
      1,                    /* Stride Height */
      2,                    /* Depth Multiplier */
      kTfLiteActNone,       /* Activation*/
      2,                    /* Dilation Width*/
      2                     /* Dilation Height */
  };

  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;
  float weights_min = -3.5f;
  float weights_max = 4.0f;

  const int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);
  const float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1 /* Packed dimension needs to be 1 */, {}};

  tflite::AsymmetricQuantize(filter_values, filter_quantized,
                             filter_elements, filter_scale, filter_zero_point);

  auto packed_weights =
          tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
              filter_quantized, 1u * filter_elements, 2, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, reinterpret_cast<const uint8_t*>(packed_weights.data()), filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, dilation_depthwise_conv_params, &packing, weights_min, weights_max);
}


TF_LITE_MICRO_TESTS_END
