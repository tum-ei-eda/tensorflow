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

#include "tensorflow/compiler/mlir/lite/experimental/sub_byte_packing.h"
#define IFX_PATCH_LOGGING 1
#include <stddef.h>
#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/QuantTypes.h"    // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/IR/Function.h"                 // from @llvm-project
#include "mlir/IR/Location.h"                 // from @llvm-project
#include "mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/IR/Module.h"                   // from @llvm-project
#include "mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/IR/StandardTypes.h"            // from @llvm-project
#include "mlir/IR/Types.h"                    // from @llvm-project
#include "mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/Translation.h"                 // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/lite/utils/stateful_ops_utils.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/delegates/flex/allowlisted_flex_ops.h"
#include "tensorflow/lite/experimental/custom_quantization_util.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/versioning/op_version.h"
#include "tensorflow/lite/tools/versioning/runtime_version.h"
#include "tensorflow/lite/version.h"

#if IFX_PATCH_LOGGING
#include <iostream>
#endif
using llvm::dyn_cast;
using llvm::formatv;
using llvm::isa;
using llvm::Optional;
using llvm::StringRef;
using llvm::Twine;
using mlir::Dialect;
using mlir::ElementsAttr;
using mlir::FuncOp;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::NoneType;
using mlir::Operation;
using mlir::Region;
using mlir::StringAttr;
using mlir::TensorType;
using mlir::Type;
using mlir::UnknownLoc;
using mlir::Value;
using tensorflow::OpOrArgLocNameMapper;
using tensorflow::OpOrArgNameMapper;
using tensorflow::Status;
using tflite::flex::IsAllowlistedFlexOp;
using xla::StatusOr;

// Same prefix as in identifiers
namespace tfl = mlir::TFL;

bool SubBytePacking::value_buffer_packing_s = false;

bool SubBytePacking::value_buffer_packing_log_s = false;

SubBytePacking::SubBytePacking(Value value)
    : container_bits_(0u),
      bits_per_item_(0u),
      packed_minor_dims_(0u),
      src_const_op(nullptr) {
  auto defing_op = value.getDefiningOp();
  if (!defing_op) return;
  auto qcst_op = dyn_cast<tfl::QConstOp>(defing_op);

  // TODO We should be emitting warnings if sub-bytpe bitwidtchs
  // for values that are no simply constants or are inputs to unsupported
  // op types (which should be specifiable as a option to allow for customized
  // implementations of the interpreter)

  // Must be a quantized constant value only used by supported
  // TFL operation - no support for sub-byte packed variable values
  // as little practical value.
  if (!qcst_op) return;

  src_const_op = qcst_op.getOperation();
  // Op ocur has to be quantized and for now we handle only < 8 bit
  // uniform quantization.  At this point the conversion will have
  // selected 8-bit integral storage type with Max-Min <= 2^7-1 indicating
  // less than 8-bits actually used)
  auto etype = qcst_op.qtype().getElementType();
  if (!etype) return;
  auto uqtype = etype.dyn_cast<mlir::quant::UniformQuantizedType>();
  if (!uqtype) return;
  if (uqtype.getStorageTypeIntegralWidth() != 8) return;

  // Check consumers of constant are all supported ops.
  //

#if IFX_PATCH_LOGGING
  if(value_buffer_packing_log_s) {
    qcst_op.emitRemark("CANDIDATE FOR PACKING!");
  }
#endif

  bool consistent_usage = true;
  for (auto user_i : value.getUsers()) {
    Operation* op = user_i;
    // Can we really have a null "user"? If so  will it eventually
    // get normalized away?   Might be safer to simply play safe and avoid
    // a packed constant in this case...
    if (!op) continue;

    int req_packed_minor_dims = 0;

    // TODO currently support-edops and packed minor dimensions
    // hard-wired. But really ought to be specifiable as as conversion
    // arguments...
    if (isa<tfl::FullyConnectedOp>(op)) {
      req_packed_minor_dims = 1;
    } else if (isa<tfl::Conv2DOp>(op)) {
      req_packed_minor_dims = 1;
    } else if (isa<tfl::DepthwiseConv2DOp>(op)) {
      req_packed_minor_dims = 1;
    } else {
      op->emitWarning(
          "Packed < 8-bit uniform quantized values used but not supported.  "
          "Will be stored as 8 bits.");
      consistent_usage = false;
    }

    if (packed_minor_dims_ && req_packed_minor_dims &&
        packed_minor_dims_ != req_packed_minor_dims) {
      op->emitWarning("Requires " + Twine(req_packed_minor_dims) +
                      " packed minor dimensinons"
                      "but other other usages shared constant require " +
                      Twine(packed_minor_dims_));
      consistent_usage = false;
    }
    packed_minor_dims_ = req_packed_minor_dims;
  }

  if (!consistent_usage) {
    return;
  }

  // Reconstruct required bit-width... difference will be 2^N-1 or 2^N - 2
  // So look for first most-significant set bit and use that to derive
  // bitwidth
  auto range = uqtype.getStorageTypeMax() - uqtype.getStorageTypeMin();
  unsigned int bit = uqtype.getStorageTypeIntegralWidth();
  for (;;) {
    // Looks like "zero bit" width... should probably never reach here
    // but who knows. TF is a big place...
    if (bit == 0) return;
    --bit;
    if (((range >> bit) & static_cast<int64_t>(1)) != 0) {
      break;
    }
  }

  // TODO Perhaps support figurability here too?
  unsigned int num_bits = bit + 1u;
  switch (num_bits) {
    case 4:
      // 2 4 bit weights per byte
      bits_per_item_ = num_bits;
      container_bits_ = 8u;
      break;
    case 5:
      // 3 5 bit weights per half-word
      bits_per_item_ = num_bits;
      container_bits_ = 16u;
      break;
    case 6:
      // 3 5 bit weights per half-word
      bits_per_item_ = num_bits;
      container_bits_ = 32u;
      break;
    default:
      // 8 bits and above requires no special handling but for unsupported
      // bitwidths below 8 we should warn that the user is wasting
      // precision because theyŕe going to be stored in 8 bits.
      if (num_bits < 8) {
        auto msg =
            "Packed  " + Twine(num_bits) +
            " bit quantized values not supported.  Will be stored as 8 bits.";
        qcst_op.emitWarning(msg);
      }
      return;
  }
#if IFX_PATCH_LOGGING
  if(value_buffer_packing_log_s) {
    qcst_op.emitRemark("Weights need to be packed!");
  }
#endif
}

void SubBytePacking::SetPackingRunLength(const std::vector<int32_t>& shape) {
  // Non-packable...
  if (!Packable()) return;

  // Record size of minor dimensions so we can
  // align on container boundary for major dimension
  // (N.b. this has to be taken into accout in operator
  // implementations)

  if (shape.empty()) {
    container_bits_ = 0;
    src_const_op->emitWarning(
        "Value has undefined shape - packed quantized values not possible.  "
        "Will store unpacked 8-bit values.");
  } else {
    auto rev_shape_i = shape.rbegin();

    // We densely pack packed_minor_dims_ minor diemensions
    // in a dense tensor so run-legnth of Packing
    // is product of minor dimension sizes.
    unsigned int packed_run_length = 1;
    for (unsigned int d = 0; d < packed_minor_dims_; ++d) {
      packed_run_length *= *rev_shape_i;
      ++rev_shape_i;
    }
    packing_run_length_ = packed_run_length;
  }
}

tflite::QuantizationDetails SubBytePacking::QuantizationDetailsType() const {
  if (Packable() && value_buffer_packing_s)
    return tflite::QuantizationDetails_CustomQuantization;
  else
    return tflite::QuantizationDetails_NONE;
}

flatbuffers::Offset<void> SubBytePacking::CustomDetails(
    flatbuffers::FlatBufferBuilder& _fbb) const {
  if (!Packable() ) return 0;
  if(!value_buffer_packing_s){
    return 0;
  }
  tflite::CustomQuantizationT qdetails;
  TfLiteCustomSub8BitPackingDetails details_struct;
  details_struct.bits_per_item = bits_per_item_;
  details_struct.container_bits = container_bits_;
  details_struct.packed_minor_dims = packed_minor_dims_;

  // Push back a "magic number" as this is for custom extensions and
  // we like to be able to spot if we run into someone else's

  uint32_t magic_bytes =
      tflite::custom_quant::PACKED_SUB8BIT_UNIFORM_DETAILS_MAGIC;
  for (unsigned int i = 0; i < 4u; ++i) {
    // Flatbuffers are little-endian...
    qdetails.custom.push_back(static_cast<uint8_t>(magic_bytes & 0xffu));
    magic_bytes >>= 8u;
  }

  const uint8_t* details_bytes = reinterpret_cast<uint8_t*>(&details_struct);
  for (unsigned int i = 0; i < sizeof(TfLiteCustomSub8BitPackingDetails); ++i) {
    // Flatbuffers are little-endian...
    qdetails.custom.push_back(details_bytes[i]);
  }
  return tflite::CustomQuantization::Pack(_fbb, &qdetails).Union();
}

// Create buffer holding  data coding sub-8-bit uniform quantized
// tensor values packed into `container_bits` container values.
//
template <typename CONTAINER_T, size_t container_bits>
flatbuffers::Offset<tflite::Buffer> SubBytePacking::CreatePackedValueBuffer(
    flatbuffers::FlatBufferBuilder& builder, Operation* inst,
    const uint8_t* tensor_data, size_t tensor_data_size) const {
  assert(container_bits <= 32u);
  std::vector<uint8_t> packed_data;
  CONTAINER_T mask = (static_cast<CONTAINER_T>(1) << bits_per_item_) -
                     static_cast<CONTAINER_T>(1);
  uint32_t container_buf = 0;
  CONTAINER_T bits_in_container = 0;
  for (size_t i = 0; i < tensor_data_size; ++i) {
    // Little-ending packing...
    container_buf |= (tensor_data[i] & mask) << bits_in_container;
    bits_in_container += bits_per_item_;
    // Flush container when insufficient space for another item
    // Start of each minor dimension to ensure CONTAINER_T aligned...
    // ToDO IFX_PATCH: probably more efficient to align on selected dimension
    // (ideally: dependent on op) to handle depthwise conv / inner loop 2D conv
    if (bits_in_container + bits_per_item_ > container_bits ||
        (i % packing_run_length_ == (packing_run_length_ - 1))) {
      // Flatbuffers are stored little-endian
      for (size_t i = 0; i < container_bits; i += 8) {
        uint8_t byte = (container_buf & 0xff);
        packed_data.push_back(byte);
        container_buf >>= 8;
      }
      bits_in_container = 0;
      container_buf = 0;
    }
  }

  assert(bits_in_container == 0);
  // flatbuffers::Offset<flatbuffers::Vector<CONTAINER_T>>
  //
  auto buffer_data = builder.CreateVector(packed_data);

#if IFX_PATCH_LOGGING

  if(value_buffer_packing_log_s) {
    std::ostringstream msg;
    msg << "Packing ";
    for (size_t i = 0; i < tensor_data_size; ++i) {
      msg << " " << std::hex << (uint32_t)(tensor_data[i] & mask);
    }
    msg << " into";
    unsigned int i = 0;
    const size_t container_bytes = (container_bits / 8);
    for (size_t i = 0; i < packed_data.size(); ++i) {
      // TFlite flatbuffers are little-endian
      if (i % container_bytes == 0) {
        msg << "|";
      }
      size_t j = (i / container_bytes) * container_bytes + container_bytes - 1u -
                (i % container_bytes);
      uint32_t v = packed_data[j];
      msg << " " << std::hex << v;
    }

    inst->emitRemark(msg.str());
  }
#endif

  return tflite::CreateBuffer(builder, buffer_data);
}

//
// Create buffer holding coding <= 8-bit uniform quantized
// tensor value.  For sub-8-bit values packed formats are
// used where supported.
//
flatbuffers::Offset<tflite::Buffer> SubBytePacking::CreateQuantizedValuesBuffer(
    flatbuffers::FlatBufferBuilder& builder, Operation* inst,
    const absl::string_view tensor_data) const {
  auto raw_tensor_data = reinterpret_cast<const uint8_t*>(tensor_data.data());

  if (Packable()) {
    if (value_buffer_packing_s) {
      // Temporary scaffolding pending availability of post-processing
      // flat-buffer filter to pack to suit selected target platform
      // for TFlite(u).
      switch (container_bits_) {
        case 8: {
#ifdef IFX_PATCH_LOGGING
          if(value_buffer_packing_log_s) {
            inst->emitRemark("Actually Packing 8-bit container with " +
                            Twine(bits_per_item_) + " bit weights!!");
          }
#endif
          return CreatePackedValueBuffer<uint8_t, 8>(
              builder, inst, raw_tensor_data, tensor_data.size());
        }

        case 16: {
#ifdef IFX_PATCH_LOGGING
          if(value_buffer_packing_log_s) {
            inst->emitRemark("Actually Packing 16-bit container with " +
                            Twine(bits_per_item_) + " bit weights!!");
          }
#endif
          return CreatePackedValueBuffer<int16_t, 16>(
              builder, inst, raw_tensor_data, tensor_data.size());
        }

        case 32: {
#ifdef IFX_PATCH_LOGGING
          if(value_buffer_packing_log_s) {
            inst->emitRemark("Actually Packing 32-bit container with " +
                            Twine(bits_per_item_) + " bit weights!!");
          }
#endif
          return CreatePackedValueBuffer<uint32_t, 32>(
              builder, inst, raw_tensor_data, tensor_data.size());
        }

        default: {
          // No supported packed container format could be identified
          // Fall through to use generic 8-bit unpacked format.

          break;
        }
      }
    }

#ifdef IFX_PATCH_LOGGING
    if(value_buffer_packing_log_s) {
      inst->emitRemark("Leaving " + Twine(bits_per_item_) +
                        " unpacked but enforcing " +
                        Twine(bits_per_item_) +
                        " bit width");
    }
#endif
    // Fall-through to here if format not supported or packing disabled.
    // Enforce specified item bit-width so  bugs in bitwidth inference
    // upstream can be detected by tests wrttien using TF-lite intererpreter with standard
    // op kernels.
      uint8_t mask = static_cast<uint8_t>((1u << bits_per_item_) - 1u);
      std::vector<uint8_t> narrowed_data;
      for (size_t i = 0; i < tensor_data.size(); ++i) {
        narrowed_data.push_back(raw_tensor_data[i] & mask);
      }
#ifdef IFX_PATCH_LOGGING

      if(value_buffer_packing_log_s) {
        std::ostringstream msg;
        msg << "Enforced values ";
        for (size_t i = 0; i < tensor_data.size(); ++i) {
          msg << " " << (uint32_t)(narrowed_data[i]);
        }
        inst->emitRemark(msg.str());
      }

#endif
      auto buffer_data = builder.CreateVector(narrowed_data);
      return tflite::CreateBuffer(builder, buffer_data);
  } else {
#ifdef IFX_PATCH_LOGGING
      if(value_buffer_packing_log_s) {
        inst->emitRemark("Not packable!");
      }
#endif
    // Not Packable - leave as-is
    auto buffer_data = builder.CreateVector(raw_tensor_data, tensor_data.size());
    return tflite::CreateBuffer(builder, buffer_data);
  }
}
