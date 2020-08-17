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

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void InitInputAndFilterRandomly(float* input, float* weights, int input_size, int weights_size) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(-4,3);
  for (int i = 0; i < weights_size; i++) {
    weights[i] = distribution(generator);
  }
  std::uniform_int_distribution<int> distribution2(-8,7);
  for (int i = 0; i < input_size; i++) {
    input[i] = distribution2(generator);
  }
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
  uint32 cont_per_minor = std::ceil((float)(minor_dim_size * bits_per_item) / container_bits);
  uint32 number_containers = (elts / minor_dim_size) * cont_per_minor;
  std::vector<CONTAINER_T> packed_data(number_containers);

  uint8_t* packed_data_byte = reinterpret_cast<uint8_t*>(packed_data.data());
  int bits_in_container = 0;
  for (size_t dim = 0; dim < elts / minor_dim_size; dim++) {
    uint32 data_index = dim * minor_dim_size;
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
                                 T* output_data, int output_length,
                                 TfLiteConvParams* conv_params, std::string desc,
                                 float tolerance = 1e-5) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;

  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_CONV_2D);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = reinterpret_cast<const char*>(conv_params);
  size_t init_data_size = 0;
  void* user_data = nullptr;

  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(conv_params);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  if (registration->prepare) {
    TfLiteStatus return_val = registration->prepare(&context, &node);
    if (return_val != kTfLiteOk) {
      return return_val;
    }
  }

  // Start main benchmarking loop
  // Increase the variable benchmarking_iterations to make result representative
  const int number_of_invocations = 100;
  auto start = std::chrono::high_resolution_clock::now();

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  for (int j = 0; j < number_of_invocations; j++) {
    TfLiteStatus return_val = registration->invoke(&context, &node);
    if (return_val != kTfLiteOk) {
      return return_val;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  micro_test::reporter->Report("%s %d Invoke run time =  %d us", desc.c_str(), number_of_invocations, duration);

  if (registration->free) {
    registration->free(&context, user_data);
  }

  return kTfLiteOk;
}


// Creates a DepthwiseConv opeerator, calls it with the provided input tensors
// and some defaults parameters, and compares the output with
// expected_output_data.
//
// The tensors parameter contains both the input tensors as well as a
// preallocated output tensor into which the output is stored.

TfLiteStatus ValidateDepthwiseConvGoldens(TfLiteDepthwiseConvParams params,
                                          float tolerance, int tensors_size,
                                          TfLiteTensor* tensors, std::string desc) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = reinterpret_cast<const char*>(&params);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&params);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  if (registration->prepare) {
    TF_LITE_ENSURE_OK(context, registration->prepare(&context, &node));
  }

  const int number_of_invocations = 100;
  auto start = std::chrono::high_resolution_clock::now();

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  for (int j = 0; j < number_of_invocations; j++) {
    TfLiteStatus return_val = registration->invoke(&context, &node);
    if (return_val != kTfLiteOk) {
      return return_val;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  micro_test::reporter->Report("%s %d Invoke run time =  %d us", desc.c_str(), number_of_invocations, duration);

  if (registration->free) {
    registration->free(&context, user_data);
  }
  return kTfLiteOk;
}

void TestDepthwiseConvQuantizedPerLayer(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data,
    const uint8_t* filter_quantized, float filter_scale, int filter_zero_point,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const int* output_dims_data,
    uint8_t* output_data, float output_scale, int output_zero_point,
    TfLiteDepthwiseConvParams params, TfLiteCustomSub8BitPackingDetails* packing,
    float weights_min, float weights_max, std::string desc) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

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

  ValidateDepthwiseConvGoldens(params, 1.0, tensors_size, tensors, desc);
}

void TestConvQuantizedPerLayer(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, const int* filter_dims_data,
    const uint8_t* filter_quantized, float filter_scale, int filter_zero_point,
    float weights_min, float weights_max, TfLiteCustomSub8BitPackingDetails* packing,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const int* output_dims_data, uint8_t* output_data,
    float output_scale, TfLiteConvParams* conv_params, std::string desc) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

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
      ValidateConvGoldens(tensors, tensors_size,
                          output_data, output_dims_count,
                          conv_params, desc, 1e-5));
}

void TestConvQuantizedPerLayerNotPacked(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, const int* filter_dims_data,
    const float* filter_data, uint8_t* filter_quantized, float filter_scale,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const int* output_dims_data, uint8_t* output_data,
    float output_scale, TfLiteConvParams* conv_params, std::string desc) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, 128),
      CreateQuantizedTensor(filter_data, filter_quantized, filter_dims,
                            filter_scale, 128),
      CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                input_scale, filter_scale),
      CreateQuantizedTensor(output_data, output_dims, output_scale, 128)};

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
  float filter_scales[] = {1, filter_scale};
  int filter_zero_points[] = {1, 128};
  TfLiteAffineQuantization filter_quant = {FloatArrayFromFloats(filter_scales),
                                           IntArrayFromInts(filter_zero_points),
                                           0};
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant, {kTfLiteNoDetails, {}}};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateConvGoldens(tensors, tensors_size,
                          output_data, output_dims_count, conv_params, desc));
}

void TestDepthwiseConvQuantizedPerLayerNotPacked(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data, const float* filter_data,
    uint8_t* filter_quantized, float filter_scale, int filter_zero_point,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const int* output_dims_data,
    uint8_t* output_data, float output_scale, int output_zero_point,
    TfLiteDepthwiseConvParams*  conv_params, std::string desc) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input_data, input_quantized,
                                             input_dims, input_scale,
                                             input_zero_point),
      tflite::testing::CreateQuantizedTensor(filter_data, filter_quantized,
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
  TfLiteAffineQuantization filter_quant = {FloatArrayFromFloats(filter_scales),
                                           IntArrayFromInts(filter_zero_points),
                                           0};
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant, {kTfLiteNoDetails, {}}};

  float bias_scales[] = {1, filter_scale * input_scale};
  int bias_zero_points[] = {1, 128};
  TfLiteAffineQuantization bias_quant = {FloatArrayFromFloats(bias_scales),
                                         IntArrayFromInts(bias_zero_points), 0};
  tensors[2].quantization = {kTfLiteAffineQuantization, &bias_quant, {kTfLiteNoDetails, {}}};

  ValidateDepthwiseConvGoldens(*conv_params, 1.0, tensors_size, tensors, desc);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ConvInvokeComparison) {
  /*
   * This test compares the invoke run times of packed vs non-packed implementations
   * of the conv kernel.
   * */

  // Common inputs and outputs.
  static const int batches = 1;
  static const int inputSize = 32;
  static const int inputChannels = 3;
  static const int kInputElements = inputSize * inputSize * inputChannels * batches;
  static const int kInputShape[] = {4, batches, inputSize, inputSize, inputChannels};
  static float kInputData[kInputElements];
  static const int filterSize = 4;
  static const int outputChannels = 4;
  static const int kFilterElements = filterSize * filterSize * inputChannels * outputChannels;
  static const int kFilterShape[] = {4, outputChannels, filterSize, filterSize, inputChannels};
  static float kFilterData[kFilterElements];
  static const int kBiasElements = outputChannels;
  static const int kBiasShape[] = {1, outputChannels};
  static const float kBiasData[] = {1, 2, 3, 4};
  static const int outputSize = (inputSize - filterSize)/2 + 1;
  static const int kOutputElements = batches * outputSize * outputSize * outputChannels;
  static const int kOutputShape[] = {4, batches, outputSize, outputSize, outputChannels};

  static TfLiteConvParams common_conv_params = {
      kTfLitePaddingValid,  // padding
      2,                    // stride_width
      2,                    // stride_height
      kTfLiteActNone,       // activation
      1,                    // dilation_width_factor
      1,                    // dilation_height_factor
  };
  // Randomly initialize inputs and filter
  tflite::testing::InitInputAndFilterRandomly(kInputData, kFilterData, kInputElements, kFilterElements);

  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  uint8_t output_data[kOutputElements];

  // 4 BIT HERE
  float weights_min = -4.f;
  float weights_max = 3.5f;

  int weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);

  float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 3 /* Packed dimension needs to be 3 */, {}};

  float input_scale = 0.5f;
  float output_scale = 1.0f;

  uint8_t input_quantized[kInputElements];
  uint8_t filter_quantized[kFilterElements];
  int32_t bias_quantized[kBiasElements];


  tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                             kFilterElements, filter_scale, weight_zero_point);

  auto packed_weights4 =
        tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
            filter_quantized, 1u * kFilterElements, inputChannels, &packing);


  tflite::testing::TestConvQuantizedPerLayer(
      kInputShape, kInputData,
      input_quantized, input_scale, kFilterShape,
      reinterpret_cast<const uint8_t*>(packed_weights4.data()),
      filter_scale, weight_zero_point, weights_min, weights_max, &packing,
      kBiasShape, kBiasData, bias_quantized,
      kOutputShape, output_data, output_scale,
      &common_conv_params, "4 Bit: ");

  // 5 BIT HERE
  weights_min = -8.f;
  weights_max = 7.5f;

  weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 5);

  filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 5);

  packing = {5, 16, 3 /* Packed dimension needs to be 3 */, {}};

  input_scale = 0.5f;
  output_scale = 1.0f;

  tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                             kFilterElements, filter_scale, weight_zero_point);

  auto packed_weights5 =
        tflite::testing::PackedSub8BitCustomQuantization<uint16_t>(
            filter_quantized, 1u * kFilterElements, inputChannels, &packing);


  tflite::testing::TestConvQuantizedPerLayer(
      kInputShape, kInputData,
      input_quantized, input_scale, kFilterShape,
      reinterpret_cast<const uint8_t*>(packed_weights5.data()),
      filter_scale, weight_zero_point, weights_min, weights_max, &packing,
      kBiasShape, kBiasData, bias_quantized,
      kOutputShape, output_data, output_scale,
      &common_conv_params, "5 Bit: ");

  // 6 BIT HERE
  weights_min = -16.f;
  weights_max = 15.5f;

  weight_zero_point =
        ZeroPointFromMinMaxPacked(weights_min, weights_max, 6);

  filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 6);

  packing = {6, 32, 3 /* Packed dimension needs to be 3 */, {}};

  input_scale = 0.5f;
  output_scale = 1.0f;

  tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                             kFilterElements, filter_scale, weight_zero_point);

  auto packed_weights6 =
        tflite::testing::PackedSub8BitCustomQuantization<uint32_t>(
            filter_quantized, 1u * kFilterElements, inputChannels, &packing);


  tflite::testing::TestConvQuantizedPerLayer(
      kInputShape, kInputData,
      input_quantized, input_scale, kFilterShape,
      reinterpret_cast<const uint8_t*>(packed_weights6.data()),
      filter_scale, weight_zero_point, weights_min, weights_max, &packing,
      kBiasShape, kBiasData, bias_quantized,
      kOutputShape, output_data, output_scale,
      &common_conv_params , "6 Bit: ");

  // 8 BIT NOT PACKED HERE
  input_scale = 0.5f;
  filter_scale = 0.5f;
  output_scale = 1.0f;
  tflite::testing::TestConvQuantizedPerLayerNotPacked(
      kInputShape, kInputData,
      input_quantized, input_scale, kFilterShape,
      kFilterData, filter_quantized, filter_scale,
      kBiasShape, kBiasData, bias_quantized,
      kOutputShape, output_data, output_scale,
      &common_conv_params, "8bit not packed: ");
  micro_test::reporter->Report("");
}

TF_LITE_MICRO_TEST(DepthwiseConvInvokeComparison) {
  /*
   * This test compares the invoke run times of packed vs non-packed implementations
   * of the depthwise conv kernel.
   * */

  static const int batches = 1;
  static const int inputSize = 32;
  static const int inputChannels = 3;
  static const int kInputElements = inputSize * inputSize * inputChannels * batches;
  static const int kInputShape[] = {4, batches, inputSize, inputSize, inputChannels};
  static float kInputData[kInputElements];
  static const int filterSize = 4;
  static const int depthMultiplier = 2;
  static const int outputChannels = inputChannels * depthMultiplier;
  static const int kFilterElements = filterSize * filterSize * 1 * outputChannels;
  static const int kFilterShape[] = {4, 1, filterSize, filterSize, outputChannels};
  static float kFilterData[kFilterElements];
  static const int kBiasElements = outputChannels;
  static const int kBiasShape[] = {1, outputChannels};
  static const float kBiasData[] = {1, 2, 3, 4, -1, -3};
  static const int outputSize = (inputSize - filterSize)/2 + 1;
  static const int kOutputElements = batches * outputSize * outputSize * outputChannels;
  static const int kOutputShape[] = {4, batches, outputSize, outputSize, outputChannels};

  TfLiteDepthwiseConvParams dconv_params = {
      kTfLitePaddingValid,  /* Padding */
      2,                    /* Stride Width */
      2,                    /* Stride Height */
      2,                    /* Depth Multiplier */
      kTfLiteActNone,       /* Activation */
      1,                    /* Dilation Width */
      1                     /* Dilation Height */
  };

  // Randomly initialize inputs and filter
  tflite::testing::InitInputAndFilterRandomly(kInputData, kFilterData, kInputElements, kFilterElements);


  // 4 BIT HERE
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ScaleFromMinMaxPacked;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  float input_scale = 0.5f;
  int input_zero_point = 128;
  float output_scale = 1.0f;
  int output_zero_point = 128;
  float weights_min = -3.5f;
  float weights_max = 4.0f;

  int filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 4);
  float filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 4);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1 /* Packed dimension needs to be 1 */, {}};

  uint8_t input_quantized[kInputElements];
  uint8_t filter_quantized[kFilterElements];
  int32_t bias_quantized[kBiasElements];
  uint8_t output_data[kOutputElements];

  tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                             kFilterElements, filter_scale, filter_zero_point);

  auto packed_weights4 =
          tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
              filter_quantized, 1u * kFilterElements, outputChannels, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      kInputShape, kInputData, input_quantized, input_scale, input_zero_point,
      kFilterShape, reinterpret_cast<const uint8_t*>(packed_weights4.data()), filter_scale,
      filter_zero_point, kBiasShape, kBiasData, bias_quantized,
      kOutputShape, output_data, output_scale,
      output_zero_point, dconv_params, &packing, weights_min, weights_max, "4 Bit: ");

  // 5 BIT HERE
  weights_min = -7.5f;
  weights_max = 8.0f;

  filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 5);
  filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 5);

  packing = {5, 16, 1 /* Packed dimension needs to be 1 */, {}};
  tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                             kFilterElements, filter_scale, filter_zero_point);

  auto packed_weights5 =
          tflite::testing::PackedSub8BitCustomQuantization<uint16_t>(
              filter_quantized, 1u * kFilterElements, outputChannels, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      kInputShape, kInputData, input_quantized, input_scale, input_zero_point,
      kFilterShape, reinterpret_cast<const uint8_t*>(packed_weights5.data()), filter_scale,
      filter_zero_point, kBiasShape, kBiasData, bias_quantized,
      kOutputShape, output_data, output_scale,
      output_zero_point, dconv_params, &packing, weights_min, weights_max, "5 Bit: ");

  // 6 BIT HERE
  weights_min = -15.5f;
  weights_max = 16.0f;

  filter_zero_point =
          ZeroPointFromMinMaxPacked(weights_min, weights_max, 6);
  filter_scale = ScaleFromMinMaxPacked(weights_min, weights_max, 6);

  packing = {6, 32, 1 /* Packed dimension needs to be 1 */, {}};
  tflite::AsymmetricQuantize(kFilterData, filter_quantized,
                             kFilterElements, filter_scale, filter_zero_point);

  auto packed_weights6 =
          tflite::testing::PackedSub8BitCustomQuantization<uint32_t>(
              filter_quantized, 1u * kFilterElements, outputChannels, &packing);

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      kInputShape, kInputData, input_quantized, input_scale, input_zero_point,
      kFilterShape, reinterpret_cast<const uint8_t*>(packed_weights6.data()), filter_scale,
      filter_zero_point, kBiasShape, kBiasData, bias_quantized,
      kOutputShape, output_data, output_scale,
      output_zero_point, dconv_params, &packing, weights_min, weights_max, "6 Bit: ");

  // 8 BIT NOT PACKED HERE
  input_scale = 0.5f;
  input_zero_point = 128;
  filter_scale = 0.5f;
  filter_zero_point = 128;
  output_scale = 1.0f;
  output_zero_point = 128;

  tflite::testing::TestDepthwiseConvQuantizedPerLayerNotPacked(
      kInputShape, kInputData, input_quantized, input_scale, input_zero_point,
      kFilterShape, kFilterData, filter_quantized, filter_scale,
      filter_zero_point, kBiasShape, kBiasData, bias_quantized, kOutputShape, output_data, output_scale,
      output_zero_point, &dconv_params, "8bit not packed: ");
  micro_test::reporter->Report("");
}

TF_LITE_MICRO_TEST(FullyConnectedInvokeComparison) {

}

TF_LITE_MICRO_TEST(MNISTPackedComparison) {
  // Test entire model on MNIST data
}


TF_LITE_MICRO_TESTS_END
