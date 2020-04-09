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

#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace fully_connected {
namespace {

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

TfLiteStatus CalculateOpData(TfLiteContext* context,
                             TfLiteFullyConnectedParams* params,
                             TfLiteType data_type, const TfLiteTensor* input,
                             const TfLiteTensor* weights,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, weights, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }
  return status;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

inline void CalculateOutputNode(const int8_t *weights, const int8_t *input, const int32_t *bias, int8_t *output,
		int32_t accum_depth, int32_t weights_offset, int32_t input_offset, int32_t output_offset,
		int32_t output_multiplier, int32_t output_shift, int32_t activation_min, int32_t activation_max) {

	// Add bias
	int32_t accum = bias ? *bias : 0;

	// Multiply and accumulate
	MultiplyAccumulate(weights, input, &accum, weights_offset, input_offset, accum_depth);

	// Quantize down
	accum = MultiplyByQuantizedMultiplier(accum, output_multiplier, output_shift);

	// Add offset
	accum += output_offset;

	// Clamp the result
	accum = ActivationFunctionWithMinMax(accum, activation_min, activation_max);

	*output = static_cast<int8_t>(accum);
}

inline void CalculateTwoOutputNodes(const int8_t *weights, const int8_t *input, const int32_t *bias, int8_t *output,
		int32_t accum_depth, int32_t weights_offset, int32_t input_offset, int32_t output_offset,
		int32_t output_multiplier, int32_t output_shift, int32_t activation_min, int32_t activation_max) {

	// Add bias
	int32_t accum[2] = {0, 0};
	if(bias)
	{
		accum[0] = bias[0];
		accum[1] = bias[1];
	}

	// Multiply and accumulate
	MultiplyAccumulateTwo(weights, input, accum, weights_offset, input_offset, accum_depth);

	// Quantize down
	accum[0] = MultiplyByQuantizedMultiplier(accum[0], output_multiplier, output_shift);
	accum[1] = MultiplyByQuantizedMultiplier(accum[1], output_multiplier, output_shift);

	// Add offset
	accum[0] += output_offset;
	accum[1] += output_offset;

	// Clamp the result
	accum[0] = ActivationFunctionWithMinMax(accum[0], activation_min, activation_max);
	accum[1] = ActivationFunctionWithMinMax(accum[1], activation_min, activation_max);

	output[0] = static_cast<int8_t>(accum[0]);
	output[1] = static_cast<int8_t>(accum[1]);
}

TfLiteStatus EvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                               TfLiteFullyConnectedParams* params, OpData* data,
                               const TfLiteTensor* input,
                               const TfLiteTensor* weights,
							   const TfLiteTensor* bias, TfLiteTensor* output) {
	//Get input info
	const int8_t* input_data = GetTensorData<int8_t>(input);
	const int32_t input_offset = -input->params.zero_point;

	//Get weights info
	const int8_t* weights_data = GetTensorData<int8_t>(weights);
	const int32_t weights_offset = -weights->params.zero_point;
	RuntimeShape weights_shape = GetTensorShape(weights);
	TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
	const int weights_dim_count = weights_shape.DimensionsCount();
	const int accum_depth = weights_shape.Dims(weights_dim_count - 1);

	//Get bias info
	const int32* bias_data = GetTensorData<int32_t>(bias);

	//Get output info
	int8_t* output_data = GetTensorData<int8_t>(output);
	const int32_t output_offset = output->params.zero_point;
	RuntimeShape output_shape = GetTensorShape(output);
	TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);
	const int32_t output_multiplier = data->output_multiplier;
	// TODO(b/138810107): Figure out whether output shift should be inverted
	const int output_shift = -data->output_shift;
	const int32_t output_activation_min = data->output_activation_min;
	const int32_t output_activation_max = data->output_activation_max;
	TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
	const int batches = output_shape.Dims(0);
	const int output_depth = output_shape.Dims(1);
	TFLITE_DCHECK_LE(output_depth, weights_shape.Dims(weights_dim_count - 2));

	const int8_t *input_ptr = input_data;
	int8_t* output_ptr = output_data;
	for (int b = 0; b < batches; ++b) {
		const int8_t *weights_ptr = weights_data;
		const int32_t *bias_ptr = bias_data;
		for (int32_t out_c = 0; out_c <= (output_depth - 2); out_c += 2) {

			CalculateTwoOutputNodes(weights_ptr, input_ptr, bias_ptr, output_ptr,
					accum_depth, weights_offset, input_offset, output_offset,
					output_multiplier, output_shift, output_activation_min, output_activation_max);
			weights_ptr += 2 * accum_depth;
			bias_ptr += 2;
			output_ptr += 2;
		}
		if (output_depth % 2){

			CalculateOutputNode(weights_ptr, input_ptr, bias_ptr, output_ptr,
					accum_depth, weights_offset, input_offset, output_offset,
					output_multiplier, output_shift, output_activation_min, output_activation_max);
			output_ptr++;
		}
		input_ptr += accum_depth;
	}
	return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFullyConnectedParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* weights, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
  const int32_t input_offset = -input->params.zero_point;
  const int32_t weights_offset = -weights->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = weights_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

#define TF_LITE_FULLY_CONNECTED(output_data_type)                      \
  reference_ops::FullyConnected(                                       \
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input), \
      GetTensorShape(weights), GetTensorData<uint8_t>(weights),          \
      GetTensorShape(bias), GetTensorData<int32_t>(bias),              \
      GetTensorShape(output), GetTensorData<output_data_type>(output))
  switch (output->type) {
    case kTfLiteUInt8:
      TF_LITE_FULLY_CONNECTED(uint8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_FULLY_CONNECTED(int16_t);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Quantized FullyConnected expects output data type uint8 or int16");
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFullyConnectedParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* weights,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  tflite::reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(weights), GetTensorData<float>(weights),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  //Portable optimized!
  TfLiteType data_type = input->type;
  OpData local_data_object;
  OpData* data = &local_data_object;
  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, data_type, input,
                                        weights, bias, output, data));
										
  switch (weights->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalFloat(context, node, params, data, input, weights, bias,
                       output);
    case kTfLiteInt8:
      return EvalQuantizedInt8(context, node, params, data, input, weights, bias,
                               output);

    case kTfLiteUInt8:
      return EvalQuantized(context, node, params, data, input, weights, bias,
                           output);

    default:
      TF_LITE_KERNEL_LOG(context, "Type %d not currently supported.",
                         weights->type);
      return kTfLiteError;
  }
  
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FULLY_CONNECTED() {
  static TfLiteRegistration r = {};
  r.init = fully_connected::Init;
  r.free = fully_connected::Free;
  r.prepare = fully_connected::Prepare;
  r.invoke = fully_connected::Eval;
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
