/*
 * static_init_support.cc
 *
 *  Created on: 10.08.2020
 *      Author: stevensa
 */

#include "tensorflow/lite/micro/kernels/static_init_support.h"

#include <cstddef>
#include <fstream>
#include <set>

#if TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT

namespace tflite {
namespace ops {
namespace micro {

// Vector: needs a named sub-initializer that has to be output first
CppItems &CppItems::operator<<(const char *literal) {
  elements_.push_back(
      std::unique_ptr<CppInitializerBase>(new CppLiteral(literal)));
  return *this;
}

CppItems &CppItems::operator<<(float value) {
  elements_.push_back(std::unique_ptr<CppInitializerBase>(
      new CppPrimitiveInitializer<float>(value)));
  return *this;
}

CppItems &CppItems::operator<<(const CppNamedStruct &structref) {
  named_sub_inits_.push_front(
      std::unique_ptr<CppDefinitionBase>(new CppNamedStruct(structref)));
  elements_.push_back(std::unique_ptr<CppInitializerBase>(
      new CppInitializerPointer(structref.getId())));
  return *this;
}

CppItems &CppItems::operator<<(const CppPODStructInitializer &substruct) {
  elements_.push_back(std::unique_ptr<CppInitializerBase>(
      new CppPODStructInitializer(substruct)));
  return *this;
}

//
// Implementation of pointer collector (will be owned) by
// PointerCollectors singleton.
class PointerCollector::Implementation {
 public:
  void addPointer(const std::string &pointer) { pointers_.push_back(pointer); }

  Implementation(const char *kernel_name, const char *local_argtype_decls,
                 const char *signature);

 protected:
  friend class PointerCollectors;
  friend class CppInitializerCollector;

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
  PointerCollectors() {}

  void addPointerCollector(PointerCollector::Implementation *collector) {
    collectors_.push_back(
        std::unique_ptr<PointerCollector::Implementation>(collector));
  }

  void writeInvokeRecordedCppFunctions(std::ostream &os);

  void recordLiteralForPointer(void *ptr, const std::string &identifier) {
    pointer_literals_[ptr] = identifier;
  }

  std::string getLiteralForPointer(void *ptr) {
    std::string res;
    auto lit_i = pointer_literals_.find(ptr);
    if (lit_i != pointer_literals_.end()) {
      res = lit_i->second;
    }
    return res;
  }

 protected:
  std::vector<std::unique_ptr<PointerCollector::Implementation>> collectors_;
  // LUT to find name for pointer (mainly intendded for function pointers)
  std::map<void *, std::string> pointer_literals_;
  std::string output_path_;
};

void PointerCollectors::writeInvokeRecordedCppFunctions(std::ostream &os) {
  os << "namespace tflite {\n"
        "namespace ops {\n"
        "namespace micro {\n\n";

  for (auto &collector : collectors_) {
    os << "namespace " << collector->kernel_name_ << " {\n\n";
    os << collector->local_argtype_decls_ << "\n";

    os << "size_t invoke_counter = 0;\n\n";
    std::set<std::string> unique_pointers;

    unique_pointers.insert(collector->pointers_.begin(),
                           collector->pointers_.end());

    os << "typedef TfLiteStatus (*RecordedVariantFPtr)("
       << collector->signature_ << ");\n";
    // Handle non-usage...
    if (unique_pointers.empty()) {
      os << "RecordedVariantFPtr recordedVariant() { return nullptr; }\n";
    } else {
      // Functino declarations
      for (auto &fn_name : unique_pointers) {
        os << "TfLiteStatus " << fn_name << "(\n"
           << collector->signature_ << ");\n";
      }
      os << "\n";

      // Function pointer table for sequence of invocations recorded for model.
      // Common special case: if same Variant is used throughout use table size
      // 1
      size_t ptr_tbl_size =
          unique_pointers.size() == 1u ? 1u : collector->pointers_.size();
      os << "RecordedVariantFPtr eval_functions[" << ptr_tbl_size << "] = {\n";
      for (size_t i = 0; i < ptr_tbl_size; ++i) {
        os << "  " << collector->pointers_[i] << ",\n";
      }
      os << "};\n";

      os << "RecordedVariantFPtr recordedVariant() {\n"
         << "  auto fptr = eval_functions[invoke_counter];\n"
         << "  invoke_counter = (invoke_counter + 1) % "
            "(sizeof(eval_functions)/sizeof(eval_functions[0]));\n"
         << "  return fptr;\n"
         << "}\n\n";
    }
    os << "} // namespace " << collector->kernel_name_ << "\n\n";
  }

  os << "} // namespace micro\n"
        "} // namespace ops\n"
        "} // namespace tflite\n";
}

class CppInitializerCollector : public PointerCollectors {
 public:
  static CppInitializerCollector &instance();

  void recordOpDataHeaders(const char *op_name, const char *headers,
                           const char *type);

  void recordStaticOpdata(const char *op_name, CppItems *op_data);

  void writeStaticOpDataHeaders(std::ostream &os);

  void writeStaticOpDataDefinitions(std::ostream &os);

  ~CppInitializerCollector() {
#if TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES
    std::ofstream myfile;
    myfile.open(
        "tensorflow/lite/micro/kernels/recorded_model/static_eval_tables.cc",
        std::fstream::out);
    myfile << "#include \"tensorflow/lite/c/common.h\"\n"
              "#include \"tensorflow/lite/c/builtin_op_data.h\"\n"
              "\n";
    writeStaticOpDataHeaders(myfile);
    myfile << "\n";
    writeInvokeRecordedCppFunctions(myfile);

    myfile << "\n";
    writeStaticOpDataDefinitions(myfile);
    myfile.close();
#endif
  }

  // Map associating operator supporting static initializatino data
  // with required headers  (identified via node pointer)
  // with recorded C++ static initialization data
  std::map<std::string, std::unique_ptr<CppNamedStructVecInitializer>>
      per_inst_user_data_;
  std::map<std::string, std::string> op_headers_;
};

CppInitializerCollector &CppInitializerCollector::instance() {
  static CppInitializerCollector inst;
  return inst;
}

void CppInitializerCollector::recordOpDataHeaders(const char *op_name,
                                                  const char *headers,
                                                  const char *op_data_type) {
  std::string key(op_name);
  auto &headers_for_op = op_headers_[key];
  TF_LITE_ASSERT(headers_for_op.empty());
  headers_for_op = std::string(headers);
  auto &op_user_data = per_inst_user_data_[key];
  op_user_data.reset(
      new CppNamedStructVecInitializer("op_user_data", op_data_type));
}

void CppInitializerCollector::recordStaticOpdata(const char *op_name,
                                                 CppItems *op_data) {
  std::string key(op_name);
  auto &inst_user_data = per_inst_user_data_[key];
  auto pod_init = new CppPODStructInitializer(op_data);
  inst_user_data->pushBackElt(pod_init);
}

void CppInitializerCollector::writeStaticOpDataHeaders(std::ostream &os) {
  for (auto &hdr_i : op_headers_) {
    os << hdr_i.second;
    os << "\n";
  }
}

void CppInitializerCollector::writeStaticOpDataDefinitions(std::ostream &os) {
  os << "namespace tflite {\n"
        "namespace ops {\n"
        "namespace micro {\n\n";

  for (auto &id_i : per_inst_user_data_) {
    os << "namespace " << id_i.first << " {\n\n";
    if (id_i.second->getSize() == 0) {
      // Special-case: never used.  Just dummy function to resolve linking.
      os << id_i.second->getType()
         << " *recordedStaticOpData() {\n"
            "  return nullptr;\n"
            "}\n\n";
    } else {
      id_i.second->serialize(os, "");

      os << "  size_t inst_counter = 0;\n\n"
         << id_i.second->getType()
         << " *recordedStaticOpData() {\n"
            "  return &op_user_data[inst_counter++];\n"
            "}\n\n";
    }
    os << "} // namespace " << id_i.first << "\n\n";
  }

  os << "void resetStaticDataCounters() { \n";
  for (auto &collector : collectors_) {
    os << "  " << collector->kernel_name_ << "::invoke_counter = 0;\n";
  }
  for (auto &id_i : per_inst_user_data_) {
    if (id_i.second->getSize() > 0u) {
      os << "  " << id_i.first << "::inst_counter = 0;\n";
    }
  }
  os << "}\n\n";

  os << "} // namespace micro\n"
        "} // namespace ops\n"
        "} // namespace tflite\n";
}

PointerCollector::Implementation::Implementation(
    const char *kernel_name, const char *local_argtype_decls,
    const char *signature)
    : kernel_name_(kernel_name),
      signature_(signature),
      local_argtype_decls_(local_argtype_decls) {}

PointerCollector::PointerCollector(const char *kernel_name,
                                   const char *local_argtype_decls,
                                   const char *signature)
    : impl_(new PointerCollector::Implementation(
          kernel_name, local_argtype_decls, signature)) {
  CppInitializerCollector::instance().addPointerCollector(impl_);
}

void PointerCollector::addPointer(const std::string &literal, void *ptr) {
  impl_->addPointer(literal);
  CppInitializerCollector::instance().recordLiteralForPointer(ptr, literal);
}

void CppPointerLiteral::serialize_initializer(std::ostream &os,
                                              const std::string &id_prefix) {
  auto literal = CppInitializerCollector::instance().getLiteralForPointer(ptr_);
  TF_LITE_ASSERT(!literal.empty());
  os << literal;
}

//
// Primary entry point for tflite(u) post-compiler...
//

void writeStaticOpDataHeaders(std::ostream &os) {
  CppInitializerCollector::instance().writeStaticOpDataHeaders(os);
}

void writeStaticOpDataDefinitions(std::ostream &os) {
  CppInitializerCollector::instance().writeStaticOpDataDefinitions(os);
}

void recordStaticOpdata(const char *op_name, CppItems *op_data) {
  CppInitializerCollector::instance().recordStaticOpdata(op_name, op_data);
}

void recordLiteralForPointer(const std::string &literal, void *ptr) {
  CppInitializerCollector::instance().recordLiteralForPointer(ptr, literal);
}

DefineStaticOpDataHeaders::DefineStaticOpDataHeaders(
    const char *op_name, const char *headers, const char *user_data_type) {
  CppInitializerCollector::instance().recordOpDataHeaders(op_name, headers,
                                                          user_data_type);
}

void writeCppFunctionsToInvokeRecorded(std::ostream &os) {
  CppInitializerCollector::instance().writeInvokeRecordedCppFunctions(os);
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif
