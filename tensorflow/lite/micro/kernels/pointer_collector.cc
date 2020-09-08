/*
 * pointer_collector.cc
 *
 *  Created on: 10.08.2020
 *      Author: krusejakob
 */

#include "tensorflow/lite/micro/kernels/pointer_collector.h"


#if TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT
#include <vector>
#include <set>
#include <fstream>
#include <memory>

//
// Implementation of pointer collector (will be owned) by
// PointerCollectors singleton.
class PointerCollector::Implementation  {
 public:


  void addPointer(const std::string &pointer) {
    pointers_.push_back(pointer);
  }

  Implementation(const char *kernel_name, const char *local_argtype_decls, const char *signature);

 protected:
  friend class PointerCollectors;

  // Vector that stores all the pointers to the used kernels
  std::vector<std::string> pointers_;
  std::string kernel_name_;
  std::string signature_;
  std::string local_argtype_decls_;

};

//
// singleton owning all all pointer collector implementations
// Used to implement auto-dump on exit  without dependency
// on static object destruction ordering.
// 

class PointerCollectors {
public:
  PointerCollectors()
  {}


  ~PointerCollectors();

  void addPointerCollector(PointerCollector::Implementation *collector) {
    collectors_.push_back(std::unique_ptr<PointerCollector::Implementation>(collector));
  }


  void writeInvokeRecordedCppFunctions(std::ostream &os);
  
  static PointerCollectors &instance() {  
    static PointerCollectors singleton; 
    return singleton;
  }

protected:
  std::vector<std::unique_ptr<PointerCollector::Implementation>> collectors_;
  std::string output_path_;
};


PointerCollector::Implementation::Implementation(const char *kernel_name, const char *local_argtype_decls, const char *signature)
    : kernel_name_(kernel_name)
    , signature_(signature)
    , local_argtype_decls_(local_argtype_decls)
  {
    PointerCollectors::instance().addPointerCollector(this);
  }


PointerCollector::PointerCollector(const char *kernel_name, const char *local_argtype_decls, const char *signature) 
  : impl_( new PointerCollector::Implementation(kernel_name, local_argtype_decls, signature))
{

}


void PointerCollector::addPointer(const std::string &pointer) {
  impl_->addPointer(pointer);
}


void PointerCollectors::writeInvokeRecordedCppFunctions(std::ostream &os) {

  for (auto &collector : collectors_) {
    os << "namespace " << collector->kernel_name_ << " {\n\n";
    os << collector->local_argtype_decls_ << "\n";

    os << "size_t invoke_counter = 0;\n\n";
    std::set<std::string> unique_pointers;
        
    unique_pointers.insert(collector->pointers_.begin(), collector->pointers_.end());

    os << "typedef TfLiteStatus (*RecordedVariantFPtr)(" << collector->signature_ << ");\n";
    // Handle non-usage...
    if (unique_pointers.empty() ) {
      os << "RecordedVariantFPtr recordedVariant() { return nullptr; }\n";
    } else {

      // Functino declarations
      for (auto & fn_name : unique_pointers) {
        os << "TfLiteStatus "<< fn_name
        << "(\n" << collector->signature_ << ");\n";
      }
      os << "\n";

      // Function pointer table for sequence of invocations recorded for model.
      // Common special case: if same Variant is used throughout use table size 1
      size_t ptr_tbl_size = unique_pointers.size() == 1u ? 1u : collector->pointers_.size();
      os << "TfLiteStatus (*eval_functions[" << ptr_tbl_size << "])"
        << "(" << collector->signature_ << ") = {\n";
      for( size_t i = 0; i < ptr_tbl_size; ++i ) {
        os << "  " << collector->pointers_[i] << ",\n";
      }
      os << "};\n";

      os << "RecordedVariantFPtr recordedVariant() {\n"
        << "  auto fptr = eval_functions[invoke_counter];\n"
        << "  invoke_counter = (invoke_counter + 1) % (sizeof(eval_functions)/sizeof(eval_functions[0]));\n"
        << "  return fptr;\n"
        << "}\n\n";
    }
    os << "} // namespace " << collector->kernel_name_ << "\n\n";
  }

  os << "void resetRecordedVariants() { \n";
  for (auto &collector : collectors_) {
    os << "  " << collector->kernel_name_ << "::invoke_counter = 0;\n";
  }
  os << "}\n\n";
}


PointerCollectors::~PointerCollectors() {
#if TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES
  std::ofstream myfile;
  myfile.open("tensorflow/lite/micro/kernels/generated/static_eval_tables.cc", std::fstream::out);
  myfile << 
  "#include \"tensorflow/lite/c/common.h\"\n"
  "#include \"tensorflow/lite/c/builtin_op_data.h\"\n"
  "\n"
  "namespace tflite {\n"
  "namespace ops {\n"
  "namespace micro {\n\n";
  writeInvokeRecordedCppFunctions(myfile);
  myfile <<
  "} // namespace micro\n"
  "} // namespace ops\n"
  "} // namespace tflite\n";
  myfile.close();
#endif
}

#endif


#if 0


FullyConnectedPointerCollector::~FullyConnectedPointerCollector() {
  //Store everything in the output file here
  std::ofstream myfile;
  myfile.open(path, std::fstream::out);
  myfile << "#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GENERATED_FULLY_CONNECTED_POINTER_TABLE_H_\n";
  myfile << "#define TENSORFLOW_LITE_MICRO_KERNELS_GENERATED_FULLY_CONNECTED_POINTER_TABLE_H_\n\n";

  myfile << "TfLiteStatus (*eval_functions[" << counter << "])(TfLiteContext* context, TfLiteFullyConnectedParams* params,\n" <<
    "OpData* opData, const TfLiteEvalTensor* input, const TfLiteEvalTensor* weights,\n" <<
    "const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) = {\n";

  for (size_t i = 0; i < pointers.size(); i++) {
    myfile << pointers[i] << ",\n";
  }
  myfile << "};\n\n#endif /* TENSORFLOW_LITE_MICRO_KERNELS_GENERATED_FULLY_CONNECTED_POINTER_TABLE_H_ */\n";
  myfile.close();
}
#endif
