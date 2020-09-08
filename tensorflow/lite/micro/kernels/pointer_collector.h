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

#define TFLITE_MICRO_POINTER_COLLECTOR(kernel_name, func_ptr_signature) \
ConvPointerCollector pointer_collector(kernel_name,func_ptr_signature);

#define TLITE_MICRO_SELECTED_KERNEL_VARIANT(funptr_name) \
  (pointer_collector.addPointer(#funptr_name), &funptr_name)



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

  static std::string writeInvokeRecordedCppFunctions(std::ostream &os);

 protected:
  friend class PointerCollectors;
  class Implementation;

  Implementation *impl_;

private:
  PointerCollector(const PointerCollector &);
};


#if 0
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

#elif TF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS


#else
#define TLITE_MICRO_SELECTED_KERNEL_VARIANT(kernel_name,funptr_name) \
  &funptr_name
#endif

#endif /* TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_POINTER_COLLECTOR_H_ */
