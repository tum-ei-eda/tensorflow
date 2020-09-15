/*
 * static_init_support.h
 *
 *  Created on: 10.08.2020
 *      Author: krusejakob
 */

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_STATIC_INIT_SUPPORT_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_STATIC_INIT_SUPPORT_H_

#include <tensorflow/lite/kernels/op_macros.h>
#include <string>
#include <sstream>
#include <vector>
#include <deque>

namespace tflite {
namespace ops {
namespace micro {

class CppVectorInfo
{
public:
    CppVectorDim(size_t le) 
    : len_(len)
    {
    }

protected:
    friend class CppStructInitWriter;
    size_t len_;
};

struct CppInitializerBase
{
    virtual void serialize(std::ostream &os, const std::string &id_prefix) = 0;
};

template<typename T>
class CppPrimitiveInitializer
    : public CppInitializerBase
{
public:
    CppPrimitiveInitialize(const T val)
        : val_(val)
    {}

    void serialize(&os, const std::string &id_prefix) {
        os << std::to_string(val_);
    }
protected:
    T val_;
};

class CppInitializerReference
    : public CppInitializerBase
{
public:
    CppInitializerReference(const char *id)
        : id_(id)
    {
    }

    
    void serialize(&os, const std::string &id_prefix) {
        os << id_prefix << id_;
    }
protected:
    const char *id_;
};

class CppNamedItemBase
    : public CppInitializerBase
{

};

class CppNamedVec :
    : public CppNamedVec
{
protected:
    CppNamedVecInitialzer(const char *type, const char *id, const T*data, size_t len)
        : type_(type)
        , id_(id)
        , data_(data)
        , len_(len)
    {
    }

    void serialize(std::ostream &os, const std::string &id_prefix) {
        os << type << " "<< id_prefix << id
           << "[] = {\n";
        for( size_t i = 0; i < len_; ++i) {
            os << data_[i] << ", ";
        }
        os << "\n};\n";
    }    
  
  const char *id() const { return id_; }
protected:
    const char *type_;
    const char *id_;
    const T *data_;
    size_t len_;
}


// Helper functions for top-level code generation.
class CppPODStructInitializer {
public:
  CppPODStructInitializer() {}


  // Simple  scalar field - no need for a named sub-initializer.
  template <typename T>
  auto operator<<(T &&value) -> decltype(((void)std::to_string(T(0)), *this)) 
  {      
      // Scalar object .. dimensions must not be sepcified...
      TF_LITE_ASSERT( pending_dims_.empty() );
      if (elts_ > 0) {
          out_ << ", ";
        	}
      ++elts_;
      out_ << std::forward<T>(value);
      return *this;

  }

  template<class T>
  CppStructInitWriter& operator<<(CppNamedVec<T> &&subvec) {
      named_sub_inits.push_back(std::make_unique<CppNamedVec>(subvec));
      named_sub_inits.push_front(elements_.back().get());
      return *this;
  }

 private:
    typedef std::deque<> 
    std::deque<std::unique_pointer<CppNamedVec> *> named_sub_inits_;
    std::vector<std::unique_pointer<CppInitializerBase>> elements_;
}; 


}  // namespace micro
}  // namespace ops
}  // namespace tflite



#endif /* TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_STATIC_INIT_SUPPORT_H_ */
