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
// #include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/packed_weights/hello_world_packed_4.h"
#include "tensorflow/lite/micro/examples/packed_weights/hello_world_packed_5.h"
#include "tensorflow/lite/micro/examples/packed_weights/hello_world_packed_6.h"
#include "tensorflow/lite/micro/examples/packed_weights/hello_world_2x4in8.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


#include "tensorflow/lite/micro/examples/packed_weights/hello_world_packed_4refdata.h"
#include "tensorflow/lite/micro/examples/packed_weights/hello_world_packed_5refdata.h"
#include "tensorflow/lite/micro/examples/packed_weights/hello_world_packed_6refdata.h"

uint8_t input_quant( float x )
{
  return static_cast<uint8_t>(x/0.024639942381096416f);
}

float output_dquant( uint8_t x)
{
  return (x-128)*7.812500e-03f;
}


void run_model( const uint8_t *model_fb, float refdata[][3] ) {
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
  const int tensor_arena_size = 16 * 1024 + 16 + 200 /* some reserved space */;
  uint8_t tensor_arena[tensor_arena_size];

// Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);

// Allocate memory from the tensor_arena for the model's tensors
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
// At the time of writing, the hello world model uses 2208 bytes, we leave
// 100 bytes head room here to make the test less fragile and in the same
// time, alert for substantial increase.
//TF_LITE_MICRO_EXPECT_LE(interpreter.arena_used_bytes(), 2208 + 100);

// Obtain a pointer to the model's input tensor
  TfLiteTensor *input = interpreter.input(0);

// Make sure the input has the properties we expect
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
// The property "dims" tells us the tensor's shape. It has one element for
// each dimension. Our input is a 2D tensor containing 1 element, so "dims"
// should have size 2.
  TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
// The value of each element gives the length of the corresponding tensor.
// We should expect two single element tensors (one is contained within the
// other).
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
// The input is a 32 bit floating point value
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

  TfLiteTensor *output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  for (int i = 0; i < 4; ++i) {
    float x = refdata[i][0];
    float ref_y = refdata[i][2];
// Provide an input value
    input->data.uint8[0] = input_quant(x);

// Run the model on this input and check that it succeeds
    TfLiteStatus invoke_status = interpreter.Invoke();
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

// Obtain the output value from the tensor
    float value = output_dquant(output->data.uint8[0]);
// Check that the output value is within 0.001 of the expected  value
// (produced using from a face-quantized prediction using the original model)
    std::cout << ref_y << " " << value << std::endl;
    TF_LITE_MICRO_EXPECT_NEAR(ref_y, value, 0.05f);
  }

}

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadAndRunPacked4Bit) {
    run_model(hello_world_packed_4_data, hello_world_packed_4_refdata);
}

TF_LITE_MICRO_TEST(LoadAndRunPacked5Bit) {
  run_model(hello_world_packed_5_data, hello_world_packed_5_refdata);
}

TF_LITE_MICRO_TEST(LoadAndRunPacked6Bit) {
  run_model(hello_world_packed_6_data, hello_world_packed_6_refdata);
}

TF_LITE_MICRO_TESTS_END
