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
#include "tensorflow/lite/micro/all_ops_resolver.h"
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
        for (size_t i = 0; i < container_bits; i += 8) {
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
TfLiteStatus ValidateDepthwiseConvGoldens(const T* expected_output_data,
                                          int output_length,
                                          TfLiteDepthwiseConvParams params,
                                          float tolerance, int tensors_size,
                                          TfLiteTensor* tensors) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  int input_depth = tensors[0].dims->data[3];
  int output_depth = tensors[1].dims->data[3];

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
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&params);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_ENSURE_OK(context, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_ENSURE_OK(context, registration->invoke(&context, &node));

  if (registration->free) {
    registration->free(&context, user_data);
  }

  const T* output_data = tflite::GetTensorData<T>(&tensors[kOutputTensorIndex]);
  for (int i = 0; i < output_length; ++i) {
    auto vx = expected_output_data[i];
    auto vy = output_data[i];
    auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx));
    if (delta > tolerance) {
      std::cout << i << ",";
    }
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
      IntArrayFromInts(filter_zero_points)};
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};

  float bias_scales[] = {1, filter_scale * input_scale};
  int bias_zero_points[] = {1, 128};
  TfLiteAffineQuantization bias_quant = {FloatArrayFromFloats(bias_scales),
                                         IntArrayFromInts(bias_zero_points)};
  tensors[2].quantization = {kTfLiteAffineQuantization, &bias_quant};

  if (packing) {
          SetPackingParams(tensors[1], weights_min, weights_max, packing);
  }

  AsymmetricQuantize(golden, golden_quantized, output_dims_count, output_scale,
                     output_zero_point);
  ValidateDepthwiseConvGoldens(golden_quantized, output_dims_count, params,
                               1.0, tensors_size, tensors);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN



TF_LITE_MICRO_TEST(DepthwiseConvQuantizedPackedWeights4Bit) {

  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;
  using tflite::testing::F2QB;
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

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1 /* Packed dimension needs to be 1 */};

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

TF_LITE_MICRO_TESTS_END
