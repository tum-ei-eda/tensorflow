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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_IO_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_IO_OPS_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {
namespace experimental {

// An operation that can save a dataset to one or more files.
class SaveDatasetOp : public HybridAsyncOpKernel {
 public:
  static constexpr const char* const kCompression = "compression";
  static constexpr const char* const kPath = "path";
  static constexpr const char* const kShardFunc = "shard_func";
  static constexpr const char* const kShardFuncOtherArgs =
      "shard_func_other_args";
  static constexpr const char* const kUseShardFunc = "use_shard_func";

  explicit SaveDatasetOp(OpKernelConstruction* ctx);

  Status DoCompute(OpKernelContext* ctx) override;

 private:
  static constexpr const int kFileFormatVersion = 2;

  Status ConsumeElement();

  Status GetShardIndex(IteratorContext* ctx,
                       InstantiatedCapturedFunction* function,
                       const std::vector<Tensor>& element, int64* shard_index);

  Status WriteData(OpKernelContext* ctx, DatasetBase* dataset,
                   std::unique_ptr<CapturedFunction> captured_func,
                   const std::string& run_dir, uint64* num_elements);

  Status WriteMetadataFile(Env* env, const std::string& path, uint64 run_id,
                           const DataTypeVector& output_dtypes,
                           uint64 num_elements, bool finalized);

  bool use_shard_func_;
  std::string compression_;
  std::shared_ptr<FunctionMetadata> func_metadata_;
};

// An operation that can load a dataset from one or more files.
class LoadDatasetOp : public DatasetOpKernel {
 public:
  static const char* const kCompression;
  static const char* const kDatasetType;
  static const char* const kOutputTypes;
  static const char* const kOutputShapes;
  static const char* const kPath; 
  static const char* const kReaderFunc;
  static const char* const kReaderFuncOtherArgs;
  static const char* const kReaderFuncTarguments;

  explicit LoadDatasetOp(OpKernelConstruction* ctx);

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;

  std::string compression_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::shared_ptr<FunctionMetadata> func_metadata_;
};

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_IO_OPS_H_
