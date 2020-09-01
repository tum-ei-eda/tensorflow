/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_TESTING_TEST_UTILS_H_
#define TENSORFLOW_LITE_MICRO_TESTING_TEST_UTILS_H_

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

//Mock allocator for using buffers in Micro-kernel functions
class MockAllocator {
public:
	MockAllocator(uint8_t* tensor_arena, size_t arena_size, size_t alignment) {
		alignment_ = alignment;
		std::uintptr_t data_as_uintptr_t = reinterpret_cast<std::uintptr_t>(tensor_arena);
		uint8_t* aligned_buffer = reinterpret_cast<uint8_t*>(
		      ((data_as_uintptr_t + (alignment - 1)) / alignment) * alignment);
		tensor_arena_ = aligned_buffer;
		next_buffer_index_ = 0;
		arena_size_ = arena_size - (aligned_buffer - tensor_arena);
	};
	TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ctx, size_t bytes, void** ptr) {
		(*ptr) = &tensor_arena_[next_buffer_index_];
		size_t aligned_size = (((bytes + (alignment_ - 1)) / alignment_) * alignment_);
		if (&tensor_arena_[next_buffer_index_] + aligned_size > tensor_arena_ + arena_size_ ) {
			ctx->ReportError(ctx, "Error in memory allocation. Buffer is too small.");
			return kTfLiteError;
		}
		next_buffer_index_ += aligned_size;
		return kTfLiteOk;
	};
	TfLiteStatus RequestScratchBufferInArena(struct TfLiteContext* ctx, size_t bytes, int* buffer_idx)
	{
		*buffer_idx = next_buffer_index_;
		return kTfLiteOk;
	};
	void* GetScratchBuffer(struct TfLiteContext* ctx, int buffer_idx)
	{
		return &tensor_arena_[buffer_idx];
	};

private:
	int next_buffer_index_;
	uint8_t* tensor_arena_;
	size_t arena_size_;
	size_t alignment_;
};

// Note: These methods are deprecated, do not use.  See b/141332970.
// USE WITH CARE!! Returns pointer to data member of argument
// so this object's lifetime must outlive any access to its underlying
// data via this pointer.
// Pass-by lvalue ref of argument protects against simple programmer
// oops but is by no means foolproof.
inline TfLiteIntArray* IntArrayFromInitializer(
    std::initializer_list<int> &int_initializer) {
  return IntArrayFromInts(int_initializer.begin());
}

// Derives the quantization range max from scaling factor and zero point.
template <typename T>
inline float MaxFromZeroPointScale(const int zero_point, const float scale) {
  return (std::numeric_limits<T>::max() - zero_point) * scale;
}

// Derives the quantization range min from scaling factor and zero point.
template <typename T>
inline float MinFromZeroPointScale(const int zero_point, const float scale) {
  return (std::numeric_limits<T>::min() - zero_point) * scale;
}

// Derives the quantization scaling factor from a min and max range.
template <typename T>
inline float ScaleFromMinMax(const float min, const float max) {
  return (max - min) /
         static_cast<float>((std::numeric_limits<T>::max() * 1.0) -
                        std::numeric_limits<T>::min());
}

//@IFX_PATCH@
// Derives the quantization scaling factor from a min and max range
// and coding bitwidth.   N.b. no support for limited_range etc.

inline float ScaleFromMinMaxPacked(const float min, const float max, 
                                   const uint32_t num_bits) {
  float code_points  = static_cast<float>((1<<num_bits)-1);
  return (max - min) / code_points;
}

// Derives the quantization zero point from a min and max range.

template <typename T>
inline int ZeroPointFromMinMax(const float min, const float max) {
  return static_cast<int>(std::numeric_limits<T>::min()) +
         static_cast<int>(-min / ScaleFromMinMax<T>(min, max) + 0.5f);
}

//@IFX_PATCH@
// Derives the quantization zero point from a min and max range
// and coding bitwidth.   N.b. no support for limited_range, nudging etc.
inline int ZeroPointFromMinMaxPacked(const float min, const float max, 
                                     const uint32_t num_bits) {
  return static_cast<int>(-min / ScaleFromMinMaxPacked(min, max, num_bits) + 0.5f);
}


//@IFX_PATCH@
// Derives the quantization scale from a min and max range
// and coding bitwidth.   N.b. no support for limited_range, nudging etc.
template<uint32_t NUM_BITS>
inline uint8_t F2QB(const float value, const float min, const float max) {
  int32_t result = ZeroPointFromMinMaxPacked(min, max, NUM_BITS) +
                   (value / ScaleFromMinMaxPacked(min, max, NUM_BITS)) + 0.5f;
  const int32_t min_code = 0;
  const int32_t max_code = (1<<NUM_BITS)-1;
  if (result < min_code) {
    result = min_code;
  }
  if (result > max_code) {
    result = max_code;
  }
  return static_cast<uint8_t>(result);
}

// Converts a quantized value to coded float
float Q2F(int32_t code, float scale, float zero_point);

// Converts a quantized value to coded float for quantization
// params of specified tensor
float Q2F(int32_t code, const TfLiteTensor *tensor);

// Converts a float value into an unsigned eight-bit quantized value
// for quantizations params of specified tensor 
uint8_t F2Q(float value, const TfLiteTensor *tensor);

// Converts a float value into an unsigned eight-bit quantized value.
uint8_t F2Q(float value, float min, float max);

// Converts a float value into a signed eight-bit quantized value.
int8_t F2QS(const float value, const float min, const float max);

// Converts a float value into a signed thirty-two-bit quantized value.  Note
// that values close to max int and min int may see significant error due to
// a lack of floating point granularity for large values.
int32_t F2Q32(const float value, const float scale);

// TODO(b/141330728): Move this method elsewhere as part clean up.
void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                     ErrorReporter* error_reporter, TfLiteContext* context);

TfLiteTensor CreateFloatTensor(std::initializer_list<float> data,
                               TfLiteIntArray* dims, bool is_variable = false);

TfLiteTensor CreateBoolTensor(std::initializer_list<bool> data,
                              TfLiteIntArray* dims, bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(const uint8_t* data, TfLiteIntArray* dims,
                                   float min, float max,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(std::initializer_list<uint8_t> data,
                                   TfLiteIntArray* dims, float min, float max,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(const int8_t* data, TfLiteIntArray* dims,
                                   float min, float max,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(std::initializer_list<int8_t> data,
                                   TfLiteIntArray* dims, float min, float max,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(float* data, uint8_t* quantized_data,
                                   TfLiteIntArray* dims,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(float* data, int8_t* quantized_data,
                                   TfLiteIntArray* dims,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(float* data, int16_t* quantized_data,
                                   TfLiteIntArray* dims,
                                   bool is_variable = false);

TfLiteTensor CreateQuantized32Tensor(const int32_t* data, TfLiteIntArray* dims,
                                     float scale, bool is_variable = false);

TfLiteTensor CreateQuantized32Tensor(std::initializer_list<int32_t> data,
                                     TfLiteIntArray* dims, float scale,
                                     bool is_variable = false);

template <typename input_type = int32_t,
          TfLiteType tensor_input_type = kTfLiteInt32>
inline TfLiteTensor CreateTensor(const input_type* data, TfLiteIntArray* dims,
                                 bool is_variable = false) {
  TfLiteTensor result;
  result.type = tensor_input_type;
  result.data.raw = reinterpret_cast<char*>(const_cast<input_type*>(data));
  result.dims = dims;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(input_type);
  result.is_variable = is_variable;
  return result;
}

template <typename input_type = int32_t,
          TfLiteType tensor_input_type = kTfLiteInt32>
inline TfLiteTensor CreateTensor(std::initializer_list<input_type> data,
                                 TfLiteIntArray* dims,
                                 bool is_variable = false) {
  return CreateTensor<input_type, tensor_input_type>(data.begin(), dims,
                                                     is_variable);
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TESTING_TEST_UTILS_H_
