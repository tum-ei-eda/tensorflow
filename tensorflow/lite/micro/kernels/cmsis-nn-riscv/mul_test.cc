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
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void TestMulQuantized(std::initializer_list<int> input1_dims_data,
                      std::initializer_list<T> input1_data,
                      std::initializer_list<int> input2_dims_data,
                      std::initializer_list<T> input2_data,
                      const float input_min, const float input_max,
                      std::initializer_list<int> output_dims_data,
                      const float output_min, const float output_max,
                      std::initializer_list<T> expected_output_data,
                      T* output_data, TfLiteFusedActivation activation,
                      int error_tolerance) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  ::tflite::AllOpsResolver resolver;

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, "input1_tensor",
                            input_min, input_max),
      CreateQuantizedTensor(input2_data, input2_dims, "input2_tensor",
                            input_min, input_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_MUL);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteMulParams builtin_data = {
      .activation = activation,
  };

  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  for (int i = 0; i < output_dims_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              error_tolerance);
  }
}

}  // namespace

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(Int8NoActivation) {
  using tflite::testing::F2QS;
  const float input_min = -1;
  const float input_max = 1;
  const float output_min = -1;
  const float output_max = 1;

  int8_t output_data[4];
  tflite::testing::TestMulQuantized({4, 1, 2, 2, 1},  // input1 dims
                                    {
                                        F2QS(-0.8, input_min, input_max),
                                        F2QS(0.2, input_min, input_max),
                                        F2QS(0.9, input_min, input_max),
                                        F2QS(0.7, input_min, input_max),
                                    },                // input1 data
                                    {4, 1, 2, 2, 1},  // input2 dims
                                    {
                                        F2QS(0.6, input_min, input_max),
                                        F2QS(0.4, input_min, input_max),
                                        F2QS(0.9, input_min, input_max),
                                        F2QS(0.8, input_min, input_max),
                                    },  // input2 data
                                    input_min, input_max,
                                    {4, 1, 2, 2, 1},  // output dims
                                    output_min, output_max,
                                    {
                                        F2QS(-0.48, output_min, output_max),
                                        F2QS(0.08, output_min, output_max),
                                        F2QS(0.81, output_min, output_max),
                                        F2QS(0.56, output_min, output_max),
                                    },  // expected output data
                                    output_data, kTfLiteActNone, 1);
}

TF_LITE_MICRO_TEST(Int8NoActivationLargeMultiplier) {
  using tflite::testing::F2QS;
  const float input_min = -100;
  const float input_max = 100;
  const float output_min = -10;
  const float output_max = 10;

  int8_t output_data[4];
  tflite::testing::TestMulQuantized(
      {4, 1, 2, 2, 1},
      {
          F2QS(-4, input_min, input_max),
          F2QS(2, input_min, input_max),
          F2QS(3, input_min, input_max),
          F2QS(1, input_min, input_max),
      },
      {4, 1, 2, 2, 1},
      {
          /* F2QS(-1, input_min, input_max), F2QS(-3, input_min, input_max), */
          F2QS(-1, input_min, input_max),
          F2QS(-3, input_min, input_max),
          F2QS(4, input_min, input_max),
          F2QS(2, input_min, input_max),
      },
      input_min, input_max, {4, 1, 2, 2, 1}, output_min, output_max,
      {
          F2QS(4, output_min, output_max),
          F2QS(-6, output_min, output_max),
          F2QS(12, output_min, output_max),
          F2QS(2, output_min, output_max),
      },
      // In Tensorflow Lite, this test have a max allowed error of 1.4f.
      // A difference of 1.4 in floating points corresponds to 18 quantized
      // for the output min/max [-10, 10].
      output_data, kTfLiteActNone, 18);
}

TF_LITE_MICRO_TEST(Int8NoActivationBroadcast) {
  using tflite::testing::F2QS;
  const float input_min = -3.0;
  const float input_max = 3.0;
  const float output_min = -3.0;
  const float output_max = 3.0;

  int8_t output_data[6];
  tflite::testing::TestMulQuantized({4, 1, 3, 1, 2},  // input1 shape
                                    {
                                        F2QS(-2.0, input_min, input_max),
                                        F2QS(0.2, input_min, input_max),
                                        F2QS(0.7, input_min, input_max),
                                        F2QS(0.8, input_min, input_max),
                                        F2QS(1.1, input_min, input_max),
                                        F2QS(2.0, input_min, input_max),
                                    },       // input1 data
                                    {1, 1},  // input2 shape
                                    {
                                        F2QS(0.1, input_min, input_max),
                                    },  // input2 data
                                    input_min, input_max,
                                    {4, 1, 3, 1, 2},  // output shape
                                    output_min, output_max,
                                    {
                                        F2QS(-0.2, output_min, output_max),
                                        F2QS(0.02, output_min, output_max),
                                        F2QS(0.07, output_min, output_max),
                                        F2QS(0.08, output_min, output_max),
                                        F2QS(0.11, output_min, output_max),
                                        F2QS(0.2, output_min, output_max),
                                    },  // expected output data
                                    output_data, kTfLiteActNone, 1);
}
TF_LITE_MICRO_TESTS_END
