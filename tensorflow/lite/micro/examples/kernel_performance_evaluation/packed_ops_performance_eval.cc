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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "tensorflow/lite/micro/examples/kernel_performance_evaluation/models/mnist_packed_4.h"
#include "tensorflow/lite/micro/examples/kernel_performance_evaluation/models/mnist_packed_5.h"
#include "tensorflow/lite/micro/examples/kernel_performance_evaluation/models/mnist_packed_6.h"
#include "tensorflow/lite/micro/examples/kernel_performance_evaluation/models/mnist_8.h"

#include "tensorflow/lite/micro/examples/kernel_performance_evaluation/models/mnist_packed_4refdata.h"
#include "tensorflow/lite/micro/examples/kernel_performance_evaluation/models/mnist_packed_5refdata.h"
#include "tensorflow/lite/micro/examples/kernel_performance_evaluation/models/mnist_packed_6refdata.h"
#include "tensorflow/lite/micro/examples/kernel_performance_evaluation/models/mnist_8refdata.h"

float output_dquant( uint8_t x )
{
  return 0.00390625f * static_cast<float>(x);
}

uint8_t input_quant( float x)
{
  return static_cast<uint8_t>(x/0.003921568859368563f);
}

void run_model( const uint8_t *model_fb, float data[2][28][28][1], float label[2][2] ) {
  // Set up logging
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter *error_reporter = &micro_error_reporter;

// Map the model into a usable data structure. This doesn't involve any
// copying or parsing, it's a very lightweight operation.
  const tflite::Model *model = ::tflite::GetModel(model_fb);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }


// This pulls in all the operation implementations we need
  tflite::AllOpsResolver resolver;

// Create an area of memory to use for input, output, and intermediate arrays.
// `arena_used_bytes` can be used to retrieve the optimal size.
//const int tensor_arena_size = 2208 + 16 + 100 /* some reserved space */;
  const int tensor_arena_size = 65536 /* some reserved space */;
  uint8_t tensor_arena[tensor_arena_size];

// Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);

// Allocate memory from the tensor_arena for the model's tensors
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
// At the time of writing, the hello world model uses 2208 bytes, we leave
// 100 bytes head room here to make the test less fragile and in the same
// time, alert for substantial increase.
  TF_LITE_MICRO_EXPECT_LE(interpreter.arena_used_bytes(), 65536);

// Obtain a pointer to the model's input tensor
  TfLiteTensor *input = interpreter.input(0);
  //uint8_t input_buffer[28*28];
  //input->data.data = input_buffer;
// Make sure the input has the properties we expect
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
// The property "dims" tells us the tensor's shape. It has one element for
// each dimension. Our input is a 2D tensor containing 1 element, so "dims"
// should have size 2.
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
// The value of each element gives the length of the corresponding tensor.
// We should expect two single element tensors (one is contained within the
// other).
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(28, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(28, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[3]);
// The input is a 32 bit floating point value
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

  TfLiteTensor *output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  for (int h = 0; h < 2; ++h) {
// Provide an input value
    float ref_data[2] = {label[h][0], label[h][1]};

// Run the model on this input and check that it succeeds
    const int number_of_invocations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int invoke_index = 0; invoke_index < number_of_invocations; invoke_index++) {
      for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
          input->data.uint8[28*i+j] = input_quant(data[h][i][j][0]);
        }
      }
      TfLiteStatus invoke_status = interpreter.Invoke();
      TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    micro_test::reporter->Report("Image%d %d Invoke run time = %d us", h,
                                 number_of_invocations, duration);
// Obtain the output value from the tensor
    float value0 = output_dquant(output->data.uint8[0]);
    float value1 = output_dquant(output->data.uint8[1]);
// Check that the output value is within 0.001 of the expected  value
// (produced using from a face-quantized prediction using the original model)
    TF_LITE_MICRO_EXPECT_NEAR(ref_data[0], value0, 0.05f);
    TF_LITE_MICRO_EXPECT_NEAR(ref_data[1], value1, 0.05f);
  }

}

namespace tflite {
namespace testing {
namespace {

template<typename T>
void InitInputAndFilterRandomly(T* input, T* weights, int input_size, int weights_size) {
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
                                 T* output_data, int output_length,
                                 TfLiteConvParams* conv_params, std::string desc,
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

  // Start main benchmarking loop
  // Increase the variable benchmarking_iterations to make result representative
  const int number_of_invocations = 100;
  auto start = std::chrono::high_resolution_clock::now();

  for (int j = 0; j < number_of_invocations; j++) {
    TfLiteStatus return_val = runner.Invoke();
    if (return_val != kTfLiteOk) {
      return return_val;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  micro_test::reporter->Report("%s %d Invoke run time =  %d us", desc.c_str(), number_of_invocations, duration);

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
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D();
  micro::KernelRunner runner(
      registration, tensors, tensors_size, inputs_array, outputs_array,
      reinterpret_cast<void*>(&params), micro_test::reporter);

  const char* init_data = reinterpret_cast<const char*>(&params);

  // TODO(b/154240825): Use a test macro here which fails and returns.
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }

  const int number_of_invocations = 100;
  auto start = std::chrono::high_resolution_clock::now();

  for (int j = 0; j < number_of_invocations; j++) {
    TfLiteStatus return_val = runner.Invoke();
    if (return_val != kTfLiteOk) {
      return return_val;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  micro_test::reporter->Report("%s %d Invoke run time =  %d us", desc.c_str(), number_of_invocations, duration);

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

template <typename T>
TfLiteStatus TestFullyConnectedQuantized(
    const int* input_dims_data, const T* input_data, const float input_min,
    const float input_max, const int* weights_dims_data, const T* weights_data,
    const float weights_min, const float weights_max,
    TfLiteCustomSub8BitPackingDetails* packing, const int* bias_dims_data,
    const int32_t* bias_data, const float bias_scale, const int* output_dims_data,
    const float output_min, const float output_max,
    TfLiteFusedActivation activation, T* output_data, std::string desc) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

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

  const int number_of_invocations = 100;
  auto start = std::chrono::high_resolution_clock::now();

  for (int j = 0; j < number_of_invocations; j++) {
    status = runner.Invoke();
    if (status != kTfLiteOk) {
      return status;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  micro_test::reporter->Report("%s %d Invoke run time =  %d us", desc.c_str(), number_of_invocations, duration);

  return kTfLiteOk;
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

TF_LITE_MICRO_TEST(FullyConnectedInvokeRuntime4Bit) {
  //
  // Test the performance of fully connected packed kernels
  //
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;
  using tflite::testing::F2QB;
  using tflite::testing::ZeroPointFromMinMax;
  using tflite::testing::ZeroPointFromMinMaxPacked;

  const float input_sf = 2.0;
  const float weights_sf = 1.0;
  const float input_min = -128.0f / input_sf;
  const float input_max = 127.0f / input_sf;
  float weights_min = -8.0f / weights_sf;
  float weights_max = 7.0f / weights_sf;
  const float bias_scale = 1.0f / (input_sf * weights_sf);
  const float output_min = -128.0f / 32.0f;
  const float output_max = 127.0f / 32.0f;
  const int batches = 1;

  const size_t num_inputs = 1024;
  const int input_dims_data[] = {2, batches, num_inputs};

  uint8_t input_data[num_inputs];
  const int num_outputs = 1024;

  const int weights_dims_data[] = {2, num_outputs, num_inputs};
  uint8_t weights_data[num_inputs*num_outputs];
  tflite::testing::InitInputAndFilterRandomly(input_data, weights_data, num_inputs, num_inputs*num_outputs);

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1, {}};

  auto packed_weights4 =
      tflite::testing::PackedSub8BitCustomQuantization<uint8_t>(
          weights_data, 1u * num_inputs*num_outputs, num_inputs, &packing);
  const int bias_dims_data[] = {1, num_outputs};
  int32_t bias_data[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    bias_data[i] = i % 4;
  }
  const int output_dims_data[] = {2, batches, num_outputs};

  uint8_t output_data[num_outputs];

  TfLiteStatus return_status = tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          packed_weights4.data(), weights_min, weights_max, &packing,
          bias_dims_data, bias_data, bias_scale,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data, "4 Bit: ");
  TF_LITE_MICRO_EXPECT_EQ(return_status, kTfLiteOk);

  packing = {5, 16, 1, {}};

  weights_min = -16.0f / weights_sf;
  weights_max = 15.0f / weights_sf;
  auto packed_weights5 =
      tflite::testing::PackedSub8BitCustomQuantization<uint16_t>(
          weights_data, 1u * num_inputs*num_outputs, num_inputs, &packing);


  return_status = tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          reinterpret_cast<const uint8_t*>(packed_weights5.data()),
          weights_min, weights_max, &packing,
          bias_dims_data, bias_data, bias_scale,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data, "5 Bit: ");
  TF_LITE_MICRO_EXPECT_EQ(return_status, kTfLiteOk);

  weights_min = -32.0f / weights_sf;
  weights_max = 31.0f / weights_sf;
  packing = {6, 32, 1, {}};

  auto packed_weights6 =
      tflite::testing::PackedSub8BitCustomQuantization<uint32_t>(
          weights_data, 1u * num_inputs*num_outputs, num_inputs, &packing);

  return_status = tflite::testing::TestFullyConnectedQuantized<uint8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      reinterpret_cast<const uint8_t*>(packed_weights6.data()),
      weights_min, weights_max, &packing,
      bias_dims_data, bias_data, bias_scale,
      output_dims_data, output_min, output_max, kTfLiteActNone,
      output_data, "6 Bit: ");
  TF_LITE_MICRO_EXPECT_EQ(return_status, kTfLiteOk);

  weights_min = -128.0f / weights_sf;
  weights_max = 127.0f / weights_sf;
  return_status = tflite::testing::TestFullyConnectedQuantized<uint8_t>(
          input_dims_data, input_data, input_min, input_max, weights_dims_data,
          weights_data, weights_min, weights_max, nullptr,
          bias_dims_data, bias_data, bias_scale,
          output_dims_data, output_min, output_max, kTfLiteActNone,
          output_data, "8 Bit (not packed): ");
  TF_LITE_MICRO_EXPECT_EQ(return_status, kTfLiteOk);
}

TF_LITE_MICRO_TEST(ConvInvokeComparison) {
  //
  // This test compares the invoke run times of packed vs non-packed implementations
  // of the conv kernel.
  //

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

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 3, {}};

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

  packing = {5, 16, 3, {}};

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

  packing = {6, 32, 3, {}};

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
  //
  // This test compares the invoke run times of packed vs non-packed implementations
  // of the depthwise conv kernel.
  //

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
      kTfLitePaddingValid,  // Padding
      2,                    // Stride Width
      2,                    // Stride Height
      2,                    // Depth Multiplier
      kTfLiteActNone,       // Activation
      1,                    // Dilation Width
      1                     // Dilation Height
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

  TfLiteCustomSub8BitPackingDetails packing = {4, 8, 1, {}};

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

  packing = {5, 16, 1, {}};
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

  packing = {6, 32, 1, {}};
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

TF_LITE_MICRO_TEST(MNISTPackedComparison) {
  // Test entire model on MNIST data
  micro_test::reporter->Report("\n8 BIT (not packed):");
  run_model(mnist_8_data, mnist_8_refdata, mnist_8_refdata_label);

  micro_test::reporter->Report("\n4 BIT:");
  run_model(mnist_packed_4_data, mnist_packed_4_refdata, mnist_packed_4_refdata_label);

  micro_test::reporter->Report("\n5 BIT:");
  run_model(mnist_packed_5_data, mnist_packed_5_refdata, mnist_packed_5_refdata_label);

  micro_test::reporter->Report("\n6 BIT:");
  run_model(mnist_packed_6_data, mnist_packed_6_refdata, mnist_packed_6_refdata_label);
}

TF_LITE_MICRO_TESTS_END
