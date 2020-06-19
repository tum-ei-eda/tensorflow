/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_

#include <cmath>
#include <cstdint>
#include <limits>


#include "tensorflow/lite/experimental/custom_quantization_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {


namespace custom_quant {

const uint32_t PACKED_SUB8BIT_UNIFORM_DETAILS_MAGIC = 0xa4592d92;

//
// Unpack and check quantized filter format details information
//
TfLiteStatus
ParseSub8BitPackedQuantizationDetails(const void *details,
                                   TfLiteQuantization &quantization,
                                   tflite::ErrorReporter *error_reporter) {


    auto custom_quant = static_cast<const tflite::CustomQuantization *>(details);
    // Recogniseable size...
    const flatbuffers::Vector<uint8_t> *data_bytes = custom_quant->custom();
    if (!data_bytes || data_bytes->Length() != sizeof(TfLiteCustomSub8BitPackingDetails)+4) {
            TF_LITE_REPORT_ERROR(
            error_reporter,
            "Custom quantization details are null or unrecognized length");
            return kTfLiteError;
    }
    

    {
        // No alignment concerns here as CustomQuantization is one
        // QuantizationDetails option which is 16-byte aligned.
        uint32_t first_four_bytes = *reinterpret_cast<const uint32_t *>(data_bytes->Data());
        uint32_t magic = flatbuffers::EndianScalar(first_four_bytes);
        if (magic != tflite::custom_quant::PACKED_SUB8BIT_UNIFORM_DETAILS_MAGIC)
        {
            TF_LITE_REPORT_ERROR(
            error_reporter,
            "Custom quantization details have unrecognized value %08x in  number field ", magic);
            return kTfLiteError;
        }
    }
    quantization.details.type = kTfLiteSub8BitPackedUniformDetail;
    quantization.details.data.custom_sub8bit_packing = 
        reinterpret_cast<const TfLiteCustomSub8BitPackingDetails *>(data_bytes->Data()+4);
}


TfLiteStatus ParseCustomQuantizationDetails(const void *details,
                                            TfLiteQuantization &quantization,
                                            tflite::ErrorReporter *error_reporter)
{
    quantization.details.type = kTfLiteUnknownDetails;
    if (!details)
        return kTfLiteOk;

    // Very probably we have a tensor holding supported packed uniform quantized data.
    return ParseSub8BitPackedQuantizationDetails(details, quantization, error_reporter);

}


} // namespace custom_quant

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_
