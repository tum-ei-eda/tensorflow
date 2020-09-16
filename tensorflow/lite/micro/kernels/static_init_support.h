/*
 * static_init_support.h
 *
 *  Created on: 10.08.2020
 *      Author: stevensa
 */

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_STATIC_INIT_SUPPORT_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_STATIC_INIT_SUPPORT_H_

#if TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/op_macros.h"



#include <string>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#define TLITE_MICRO_SELECTED_KERNEL_VARIANT(funptr_name) \
  (pointer_collector.addPointer(#funptr_name, reinterpret_cast<void*>(&funptr_name)), &funptr_name)

#define TT_LITE_MICRO_EVAL_VARIANT_FPTR(funptr_name) \
  (recordLiteralForPointer(#funptr_name, reinterpret_cast<void*>(&funptr_name)), &funptr_name)


#define KERNEL_VARIANT_COLLECT_INFO(kernel_name, type_decls, headers, op_data_type, funptr_signature) \
tflite::ops::micro::PointerCollector pointer_collector( \
    kernel_name, \
    type_decls, \
    funptr_signature \
); \
tflite::ops::micro::DefineStaticOpDataHeaders op_data_info( \
  kernel_name, \
  headers, \
  op_data_type \
);

#define TF_LITE_MICRO_RECORD_OP_USER_DATA(kernel_name, op_data) \
  recordStaticOpdata(kernel_name, op_data)

namespace tflite {
namespace ops {
namespace micro {

class PointerCollectors;

class PointerCollector  {
 public:

  /*
   * Constructor 
   */
  PointerCollector(const char *kernel_name, const char *local_argtype_decls, const char *signature);



  /*
   * Method that is used to add a pointer to the pointer list
   */
  void addPointer(const std::string &pointer, void *ptr);

  static void setOutputPath(const std::string &output_path);

 protected:
  friend class PointerCollectors;
  class Implementation;

  Implementation *impl_;

private:
  PointerCollector(const PointerCollector &);
};

class CppNamedStruct;
class CppPODStructInitializer;

struct CppInitializerBase {
  virtual void serialize_initializer(std::ostream &os,
                                     const std::string &id_prefix) = 0;
  virtual void serialize_dependencies(std::ostream &os,
                                      const std::string &id_prefix) = 0;

  void serialize(std::ostream &os, const std::string &id_prefix) {
    serialize_dependencies(os, id_prefix);
    serialize_initializer(os, id_prefix);
  }
};

template <typename T>
class CppPrimitiveInitializer : public CppInitializerBase {
 public:
  CppPrimitiveInitializer(const T val) : val_(val) {}

  void serialize_dependencies(std::ostream &os, const std::string &id_prefix) {}

  void serialize_initializer(std::ostream &os, const std::string &id_prefix) {
    os << std::to_string(val_);
  }

 protected:
  T val_;
};

class CppNamedItemBase : virtual public CppInitializerBase {
 protected:
  CppNamedItemBase() {}

 public:
  CppNamedItemBase(const char *id) : id_(id) {}

  const char *getId() const { return id_; }

 protected:
  const char *id_;
};

class CppInitializerReference : public CppNamedItemBase {
 public:
  CppInitializerReference(const char *id) : CppNamedItemBase(id) {}

  void serialize_dependencies(std::ostream &os, const std::string &id_prefix) {}

  void serialize_initializer(std::ostream &os, const std::string &id_prefix) {
    os << id_prefix << id_;
  }
};


class CppInitializerPointer : public CppNamedItemBase {
 public:
  CppInitializerPointer(const char *id) : CppNamedItemBase(id) {}

  void serialize_dependencies(std::ostream &os, const std::string &id_prefix) {}

  void serialize_initializer(std::ostream &os, const std::string &id_prefix) {
    os << "&" << id_prefix << id_;
  }
};


class CppLiteral : public CppInitializerBase {
 public:
  CppLiteral(const char *literal) : literal_(literal) {}

  CppLiteral(const std::string &literal) : literal_(literal) {}

  CppLiteral(std::string &&literal)
      : literal_(std::forward<std::string>(literal)) {}

  void serialize_dependencies(std::ostream &os, const std::string &id_prefix) {}

  void serialize_initializer(std::ostream &os, const std::string &id_prefix) {
    os << literal_;
  }

 protected:
  std::string literal_;
};

class CppPointerLiteral : public CppInitializerBase {
 public:
  CppPointerLiteral(void *ptr) : ptr_(ptr) {}


  void serialize_dependencies(std::ostream &os, const std::string &id_prefix) {}

  void serialize_initializer(std::ostream &os, const std::string &id_prefix);

 protected:
  void *ptr_;
};


class CppDefinitionBase : public CppNamedItemBase {
 public:
  CppDefinitionBase(const char *id, const char *type)
      : CppNamedItemBase(id), type_(type) {}


  const char *getType() const { return type_; }

 protected:
  const char *type_;
};

template <typename T>
class CppNamedVec : public CppDefinitionBase {
 public:
  CppNamedVec(const char *id, const char *type, const T *data, size_t len)
      : CppDefinitionBase(id, type)
      , null_(data == nullptr) {
    if (!null_) {
      for (size_t i = 0; i < len; ++i) {
        data_.push_back(data[i]);
      }
    }
  }

  void serialize_dependencies(std::ostream &os, const std::string &id_prefix) {}

  void serialize_initializer(std::ostream &os, const std::string &id_prefix) {
    if (null_) {
      os << "constexpr " << type_ << " *" << id_prefix << id_ << " = nullptr;\n";
    } else {
      os << type_ << " " << id_prefix << id_ << "[] = {\n";
      for (size_t i = 0; i < data_.size(); ++i) {
        os << data_[i] << ", ";
      }
      os << "\n};\n";
    }
  }

 protected:
  // We have copy data as (de)allocation before serialization is possible
  std::vector<T> data_;
  bool null_;
};


class CppItems  {
 public:
  CppItems() {}

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, CppItems &>::type
  operator<<(T value) {
    elements_.push_back(std::unique_ptr<CppInitializerBase>(
        new CppPrimitiveInitializer<T>(value)));
    return *this;
  }

  // Vector: needs a named sub-initializer that has to be output first
  template <typename T>
  CppItems &operator<<(const CppNamedVec<T> &subvec);
  
  // Vector: needs a named sub-initializer that has to be output first
  CppItems &operator<<(const char *literal);


  template <typename T>
  typename std::enable_if<std::is_pointer<T>::value,
                          CppItems &>::type
  operator<<(T value);

  CppItems &operator<<(const CppNamedStruct &structref);

  CppItems &operator<<(const CppPODStructInitializer &substruct);

  typedef std::deque<std::unique_ptr<CppDefinitionBase>> named_subinits_t;
  typedef std::vector<std::unique_ptr<CppInitializerBase>> elements_t;

  const named_subinits_t &defnsForElements() const { return named_sub_inits_; }

  const elements_t &elements() const { return elements_; }

  named_subinits_t named_sub_inits_;
  elements_t elements_;

};  // namespace micro


class CppPODStructInitializer : public CppInitializerBase {
 public:
  CppPODStructInitializer(CppItems *cppitems) 
    : cppitems_(cppitems)
  {
  }


  void serialize_dependencies(std::ostream &os, const std::string &id_prefix) {
    for (auto &si : cppitems_->defnsForElements()) {
      si->serialize(os, id_prefix);
    }
  }

  void serialize_initializer(std::ostream &os, const std::string &id_prefix) {
    os << "{";
    auto &elts = cppitems_->elements();
    for (size_t i = 0; i < elts.size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      elts[i]->serialize_initializer(os, id_prefix);
    }
    os << "}";
  }

  std::shared_ptr<CppItems> cppitems_;

};  // namespace micro


class CppNamedStruct : public CppDefinitionBase {
 public:
  CppNamedStruct(const char *id, const char *type, CppItems *cppitems)
      : CppDefinitionBase(id, type)
      , cppitems_(cppitems)
    {}

  void serialize_dependencies(std::ostream &os, const std::string &id_prefix) {
    std::string sub_prefix = id_prefix + id_ + "_";
    cppitems_.serialize_dependencies(os, sub_prefix);
  }

  void serialize_initializer(std::ostream &os, const std::string &id_prefix) {
    os << type_ << " " << id_prefix << id_ << " = \n";
    std::string sub_prefix = id_prefix + id_ + "_";
    cppitems_.serialize_initializer(os, sub_prefix);
    os << ";\n";
  }

protected:
  CppPODStructInitializer cppitems_;
};

class CppNamedStructVecInitializer : public CppDefinitionBase {
 public:
  CppNamedStructVecInitializer(const char *id, const char *type)
      : CppDefinitionBase(id, type) {}

  void serialize_dependencies(std::ostream &os, const std::string &id_prefix) {
    for (size_t i = 0; i < elts_.size(); ++i) {
      std::string sub_prefix = id_prefix + id_ + std::to_string(i) + "_";
      elts_[i]->serialize_dependencies(os, sub_prefix);
    }
  }

  void serialize_initializer(std::ostream &os, const std::string &id_prefix) {
    os << getType() << " " << id_prefix << id_ << "[] = {\n";
    for (size_t i = 0; i < elts_.size(); ++i) {
      os << "  ";
      std::string sub_prefix = id_prefix + id_ + std::to_string(i) + "_";
      elts_[i]->serialize_initializer(os, sub_prefix);
      if (i < elts_.size()-1) {
        os << ", ";
      }
      os << "\n";
    }
    os << "};\n";
  }

  void pushBackElt(CppPODStructInitializer *elt) {
    elts_.push_back(std::unique_ptr<CppPODStructInitializer>(elt));
  }


  size_t getSize() const { return elts_.size(); }

 protected:
  std::vector<std::unique_ptr<CppPODStructInitializer>> elts_;

};  

//
// Implementation of CppItems stream ops
// 

template <typename T>
CppItems &CppItems::operator<<(const CppNamedVec<T> &subvec) {
  named_sub_inits_.push_front(
      std::unique_ptr<CppDefinitionBase>(new CppNamedVec<T>(subvec)));
  elements_.push_back(std::unique_ptr<CppInitializerBase>(
      new CppInitializerReference(subvec.getId())));
  return *this;
}


template <typename T>
typename std::enable_if<std::is_pointer<T>::value,
                        CppItems &>::type
CppItems::operator<<(T value) {
  elements_.push_back(std::unique_ptr<CppPointerLiteral>(
      new CppPointerLiteral(reinterpret_cast<void *>(value))));
  return *this;
}


//
// Primary entry-points for tflite(u) post-compiler...
//

void writeStaticOpDataHeaders(std::ostream &os);

void writeStaticOpDataDefinitions(std::ostream &os);

void recordStaticOpdata(const char *op_name, CppItems *op_data);

void writeCppFunctionsToInvokeRecorded(std::ostream &os);

void recordLiteralForPointer(const std::string &literal, void *ptr);

class DefineStaticOpDataHeaders {
 public:
  DefineStaticOpDataHeaders(const char *op_name, const char *headers,
                            const char *user_data_type);
};

}  // namespace micro
}  // namespace ops
}  // namespace tflite


#elif TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS

#define TLITE_MICRO_SELECTED_KERNEL_VARIANT(funptr_name) \
  &funptr_name

#define TT_LITE_MICRO_EVAL_VARIANT_FPTR(funptr_name) \
  &funptr_name

#define KERNEL_VARIANT_COLLECT_INFO(kernel_name, type_decls, headers, op_data_type, funptr_signature)

#define TF_LITE_MICRO_RECORD_OP_USER_DATA(kernel_name, op_data)

#else
#define TLITE_MICRO_SELECTED_KERNEL_VARIANT(funptr_name) \
  &funptr_name

#define TT_LITE_MICRO_EVAL_VARIANT_FPTR(funptr_name) \
  &funptr_name
  
#define KERNEL_VARIANT_COLLECT_INFO(kernel_name, type_decls, headers, op_data_type, funptr_signature)

#define TF_LITE_MICRO_RECORD_OP_USER_DATA(kernel_name, op_data)

#endif

#endif /* TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_STATIC_INIT_SUPPORT_H_ \
        */
