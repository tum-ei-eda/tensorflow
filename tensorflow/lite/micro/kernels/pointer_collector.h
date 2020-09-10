/*
 * pointer_collector.h
 *
 *  Created on: 10.08.2020
 *      Author: krusejakob
 */

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_POINTER_COLLECTOR_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_POINTER_COLLECTOR_H_

#include "tensorflow/lite/c/common.h"

#if TF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT

#include <string>
#include <iostream>


#define TLITE_MICRO_SELECTED_KERNEL_VARIANT(funptr_name) \
  (pointer_collector.addPointer(#funptr_name), &funptr_name)

#define KERNEL_VARIANT_COLLECT_INFO(kernel_name, type_decls, funptr_signature) \
  static tflite::ops::micro::PointerCollector pointer_collector( \
    kernel_name, \
    type_decls, \
    funptr_signature \
);

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
  void addPointer(const std::string &pointer);

  static void setOutputPath(const std::string &output_path);

 protected:
  friend class PointerCollectors;
  class Implementation;

  Implementation *impl_;

private:
  PointerCollector(const PointerCollector &);
};

//
// Primary entry point for tflite(u) post-compiler...
//
void writeCppFunctionsToInvokeRecorded(std::ostream &os);

#if 0 // TODO REMOVE
class ConvPointerCollector: public PointerCollector {
 public:
  ConvPointerCollector(std::string _path): PointerCollector(_path) {}

  ~ConvPointerCollector();
};

class DepthwiseConvPointerCollector: public PointerCollector {
 public:
  DepthwiseConvPointerCollector(std::string _path): PointerCollector(_path) {}

  ~DepthwiseConvPointerCollector();
};

class FullyConnectedPointerCollector: public PointerCollector {
 public:
  FullyConnectedPointerCollector(std::string _path): PointerCollector(_path) {}

  ~FullyConnectedPointerCollector();
};


#endif

}  // namespace micro
}  // namespace ops
}  // namespace tflite


#elif TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS

#define KERNEL_VARIANT_COLLECT_INFO(kernel_name, type_decls, funptr_signature)

#else
#define TLITE_MICRO_SELECTED_KERNEL_VARIANT(funptr_name) \
  &funptr_name

  
#define KERNEL_VARIANT_COLLECT_INFO(kernel_name, type_decls, funptr_signature)

#endif

#endif /* TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_POINTER_COLLECTOR_H_ */
