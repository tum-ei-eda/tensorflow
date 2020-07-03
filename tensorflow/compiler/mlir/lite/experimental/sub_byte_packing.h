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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_SUB_BYTE_PACKING_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_SUB_BYTE_PACKING_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "mlir/IR/Operation.h"        // from @llvm-project
#include "mlir/IR/Value.h"            // from @llvm-project
#include "tensorflow/lite/schema/schema_generated.h"

// @IFX_PATCH@
// Layout and export of tensors contants with packed sub-byte sized elements
// (currently used for for sub-byte bitwidth quantized constants)
//
class SubBytePacking {
 public:

  SubBytePacking(mlir::Value value);

  //
  // Create buffer holding coding <= 8-bit uniform quantized
  // tensor value.  For sub-8-bit values packed formats are
  // used where supported (i.e. if Packable() )
  //
  flatbuffers::Offset<tflite::Buffer> CreateQuantizedValuesBuffer(
      flatbuffers::FlatBufferBuilder& builder, mlir::Operation* inst,
      const absl::string_view tensor_data) const;

  //
  // Type of details Structure produced.
  tflite::QuantizationDetails QuantizationDetailsType() const;

  //
  // Shape is not available when packing object needs to be
  // constructed so need member to setup when it is available.
  // No-op if not Packable
  void SetPackingRunLength(const std::vector<int32_t>& shape);

  // Sub-byte packing info is packed as CustomQuantization
  // entry in the quantizaton strcuture.   This inserts the
  // corresponding entry into flat-buffer and returns offset.
  //
  flatbuffers::Offset<void> CustomDetails(
    flatbuffers::FlatBufferBuilder& _fbb) const;

 // 
 inline bool Packable() const { return bits_per_item_ != 0; }


 protected:
  // Create buffer holding  data coding sub-8-bit uniform quantized
  // tensor values packed into `container_bits` container values.
  //
  template <typename CONTAINER_T, size_t container_bits>
  flatbuffers::Offset<tflite::Buffer> CreatePackedValueBuffer(
      flatbuffers::FlatBufferBuilder& builder, mlir::Operation* inst,
      const uint8_t* tensor_data, size_t tensor_data_size) const;

 private:
  // Bitwidth of addressable container word use to hold packed data items.
  // Implicitly: currently packed from lsb upwards
  unsigned int container_bits_;

  // Size of individual packed data items 
  // (0 if not packing possible/needed)
  unsigned int bits_per_item_;

  // Minor dimensions for which values are densely packed.
  // For other dimensions corresponding sub-tensor starts
  // at addressable container word.
  unsigned int packed_minor_dims_;

  // Product of sizes of packed minor dimensions.
  size_t packing_run_length_;
  mlir::Operation* src_const_op;
};

#endif // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_SUB_BYTE_PACKING_H_
