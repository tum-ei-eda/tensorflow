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

#include "tensorflow/lite/c/common.h"
#include "flatbuffers/flatbuffers.h"
#include <gtest/gtest.h>

namespace tflite {

// NOTE: this tests only the TfLiteIntArray part of context.
// most of common.h is provided in the context of using it with
// interpreter.h and interpreter.cc, so interpreter_test.cc tests context
// structures more thoroughly.

TEST(IntArray, TestIntArrayCreate) {
  TfLiteIntArray* a = TfLiteIntArrayCreate(0);
  TfLiteIntArray* b = TfLiteIntArrayCreate(3);
  TfLiteIntArrayFree(a);
  TfLiteIntArrayFree(b);
}

TEST(IntArray, TestIntArrayCopy) {
  TfLiteIntArray* a = TfLiteIntArrayCreate(2);
  a->data[0] = 22;
  a->data[1] = 24;
  TfLiteIntArray* b = TfLiteIntArrayCopy(a);
  ASSERT_NE(a, b);
  ASSERT_EQ(a->size, b->size);
  ASSERT_EQ(a->data[0], b->data[0]);
  ASSERT_EQ(a->data[1], b->data[1]);
  TfLiteIntArrayFree(a);
  TfLiteIntArrayFree(b);
}

TEST(IntArray, TestIntArrayEqual) {
  TfLiteIntArray* a = TfLiteIntArrayCreate(1);
  a->data[0] = 1;
  TfLiteIntArray* b = TfLiteIntArrayCreate(2);
  b->data[0] = 5;
  b->data[1] = 6;
  TfLiteIntArray* c = TfLiteIntArrayCreate(2);
  c->data[0] = 5;
  c->data[1] = 6;
  TfLiteIntArray* d = TfLiteIntArrayCreate(2);
  d->data[0] = 6;
  d->data[1] = 6;
  ASSERT_FALSE(TfLiteIntArrayEqual(a, b));
  ASSERT_TRUE(TfLiteIntArrayEqual(b, c));
  ASSERT_TRUE(TfLiteIntArrayEqual(b, b));
  ASSERT_FALSE(TfLiteIntArrayEqual(c, d));
  TfLiteIntArrayFree(a);
  TfLiteIntArrayFree(b);
  TfLiteIntArrayFree(c);
  TfLiteIntArrayFree(d);
}

// @IFX_PATCH@
TEST(UInt8Array, TestUInt8ArrayCreate) {
  TfLiteUInt8Array* a = TfLiteUInt8ArrayCreate(0);
  TfLiteUInt8Array* b = TfLiteUInt8ArrayCreate(3);
  TfLiteUInt8ArrayFree(a);
  TfLiteUInt8ArrayFree(b);
}

TEST(UInt8Array, TestMatchesFlatBuffer) {
  struct testbuf {
    TfLiteUInt8Array tfl;
    uint8_t _a_data[4];
  } a;

  flatbuffers::Vector<uint8_t> *fb = reinterpret_cast<tflite::flatbuffers::Vector<uint8_t> *>(&tfl);

  // Check we have same-size size fields so reinterpret_cast of 
  a.tfl.size = static_cast<unit64_t>(0xdeadbeff)<<1;
  a.tfl.data[0] = 0xff;
  a.tfl.data[1] = 17;
  a.tfl.data[2] = 99;
  a.tfl.data[3] = 11111;
  ASSERT_EQ(tfl.a->size, fb->size());

  a.tfl.size = 4;
  ASSERT_EQ(a.tfl.data[3], (uint8_t)11111);
  for( int i = 0; i < 4; ++i ) {
    ASSERT_EQ(fb->Get(i), a.tfl.data[i]);
  }

}

TEST(UInt8Array, TestUInt8ArrayCopy) {
  TfLiteUInt8Array* a = TfLiteUInt8ArrayCreate(2);
  a->data[0] = 22;
  a->data[1] = 24;
  TfLiteUInt8Array* b = TfLiteUInt8ArrayCopy(a);
  ASSERT_NE(a, b);
  ASSERT_EQ(a->size, b->size);
  ASSERT_EQ(a->data[0], b->data[0]);
  ASSERT_EQ(a->data[1], b->data[1]);
  TfLiteUInt8ArrayFree(a);
  TfLiteUInt8ArrayFree(b);
}

TEST(UInt8Array, TestUInt8ArrayEqual) {
  TfLiteUInt8Array* a = TfLiteUInt8ArrayCreate(1);
  a->data[0] = 1;
  TfLiteUInt8Array* b = TfLiteUInt8ArrayCreate(2);
  b->data[0] = 5;
  b->data[1] = 6;
  TfLiteUInt8Array* c = TfLiteUInt8ArrayCreate(2);
  c->data[0] = 5;
  c->data[1] = 6;
  TfLiteUInt8Array* d = TfLiteUInt8ArrayCreate(2);
  d->data[0] = 6;
  d->data[1] = 6;
  ASSERT_FALSE(TfLiteUInt8ArrayEqual(a, b));
  ASSERT_TRUE(TfLiteUInt8ArrayEqual(b, c));
  ASSERT_TRUE(TfLiteUInt8ArrayEqual(b, b));
  ASSERT_FALSE(TfLiteUInt8ArrayEqual(c, d));
  TfLiteUInt8ArrayFree(a);
  TfLiteUInt8ArrayFree(b);
  TfLiteUInt8ArrayFree(c);
  TfLiteUInt8ArrayFree(d);
}

TEST(FloatArray, TestFloatArrayCreate) {
  TfLiteFloatArray* a = TfLiteFloatArrayCreate(0);
  TfLiteFloatArray* b = TfLiteFloatArrayCreate(3);
  TfLiteFloatArrayFree(a);
  TfLiteFloatArrayFree(b);
}

TEST(Types, TestTypeNames) {
  auto type_name = [](TfLiteType t) {
    return std::string(TfLiteTypeGetName(t));
  };
  EXPECT_EQ(type_name(kTfLiteNoType), "NOTYPE");
  EXPECT_EQ(type_name(kTfLiteFloat32), "FLOAT32");
  EXPECT_EQ(type_name(kTfLiteFloat16), "FLOAT16");
  EXPECT_EQ(type_name(kTfLiteInt16), "INT16");
  EXPECT_EQ(type_name(kTfLiteInt32), "INT32");
  EXPECT_EQ(type_name(kTfLiteUInt8), "UINT8");
  EXPECT_EQ(type_name(kTfLiteInt8), "INT8");
  EXPECT_EQ(type_name(kTfLiteInt64), "INT64");
  EXPECT_EQ(type_name(kTfLiteBool), "BOOL");
  EXPECT_EQ(type_name(kTfLiteComplex64), "COMPLEX64");
  EXPECT_EQ(type_name(kTfLiteString), "STRING");
}

TEST(Quantization, TestQuantizationFree) {
  TfLiteTensor t;
  // Set these values, otherwise TfLiteTensorFree has uninitialized values.
  t.allocation_type = kTfLiteArenaRw;
  t.dims = nullptr;
  t.dims_signature = nullptr;
  t.quantization.type = kTfLiteAffineQuantization;
  t.sparsity = nullptr;
  auto* params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  params->scale = TfLiteFloatArrayCreate(3);
  params->zero_point = TfLiteIntArrayCreate(3);
  t.quantization.params = reinterpret_cast<void*>(params);
  TfLiteTensorFree(&t);
}

TEST(Sparsity, TestSparsityFree) {
  TfLiteTensor t = {};
  // Set these values, otherwise TfLiteTensorFree has uninitialized values.
  t.allocation_type = kTfLiteArenaRw;
  t.dims = nullptr;
  t.dims_signature = nullptr;

  // A dummy CSR sparse matrix.
  t.sparsity = static_cast<TfLiteSparsity*>(malloc(sizeof(TfLiteSparsity)));
  t.sparsity->traversal_order = TfLiteIntArrayCreate(2);
  t.sparsity->block_map = nullptr;

  t.sparsity->dim_metadata = static_cast<TfLiteDimensionMetadata*>(
      malloc(sizeof(TfLiteDimensionMetadata) * 2));
  t.sparsity->dim_metadata_size = 2;

  t.sparsity->dim_metadata[0].format = kTfLiteDimDense;
  t.sparsity->dim_metadata[0].dense_size = 4;

  t.sparsity->dim_metadata[1].format = kTfLiteDimSparseCSR;
  t.sparsity->dim_metadata[1].array_segments = TfLiteIntArrayCreate(2);
  t.sparsity->dim_metadata[1].array_indices = TfLiteIntArrayCreate(3);

  TfLiteTensorFree(&t);
}

}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
