/*
 * pointer_collector.h
 *
 *  Created on: 10.08.2020
 *      Author: krusejakob
 */

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_POINTER_COLLECTOR_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_POINTER_COLLECTOR_H_

#include <vector>
#include <string>
#include <fstream>

class PointerCollector  {
 public:

  /*
   * Method that is used to add a pointer to the pointer list
   */
  void add_pointer(std::string pointer) {
    pointers.push_back(pointer);
    counter++;
  }

  /*
   * Constructor that initializes the path
   */
  PointerCollector(std::string _path) {
    this->path = _path;
  }

  /*
   * Destructor that stores the pointer vector in a file for later use
   */
  virtual ~PointerCollector() {}

 protected:

  // Counter that counts the number of times the kernel is used
  unsigned int counter = 0;
  // Vector that stores all the pointers to the used kernels
  std::vector<std::string> pointers;
  // Path to the output file
  std::string path;
};

class ConvPointerCollector: public PointerCollector {
 public:
  ConvPointerCollector(std::string _path): PointerCollector(_path) {}

  ~ConvPointerCollector() {
    //Store everything in the output file here
    std::ofstream myfile;
    myfile.open(path, std::fstream::out);
    myfile << "#ifndef TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_CONV_POINTER_TABLE_H_\n";
    myfile << "#define TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_CONV_POINTER_TABLE_H_\n\n";

    myfile << "TfLiteStatus (*eval_functions[" << counter << "])(TfLiteConvParams* params, OpData* data,\n" <<
        "    const TfLiteTensor* input, const TfLiteTensor* filter, \n" <<
        "    const TfLiteTensor* bias, TfLiteTensor* output, TfLiteContext* context) = {\n";

    for (size_t i = 0; i < pointers.size(); i++) {
      myfile << pointers[i] << ",\n";
    }
    myfile << "};\n\n#endif /* TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_CONV_POINTER_TABLE_H_ */\n";
    myfile.close();
  }
};

class DepthwiseConvPointerCollector: public PointerCollector {
 public:
  DepthwiseConvPointerCollector(std::string _path): PointerCollector(_path) {}

  ~DepthwiseConvPointerCollector() {
    //Store everything in the output file here
    std::ofstream myfile;
    myfile.open(path, std::fstream::out);
    myfile << "#ifndef TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_DEPTHWISE_CONV_POINTER_TABLE_H_\n";
    myfile << "#define TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_DEPTHWISE_CONV_POINTER_TABLE_H_\n\n";

    myfile << "TfLiteStatus (*eval_functions[" << counter << "])(TfLiteContext* context, const TfLiteDepthwiseConvParams& params,\n" <<
        "    const OpData* data, const TfLiteTensor* input, const TfLiteTensor* filter, \n" <<
        "    const TfLiteTensor* bias, TfLiteTensor* output) = {\n";

    for (size_t i = 0; i < pointers.size(); i++) {
      myfile << pointers[i] << ",\n";
    }
    myfile << "};\n\n#endif /* TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_DEPTHWISE_CONV_POINTER_TABLE_H_ */\n";
    myfile.close();
  }
};



#endif /* TENSORFLOW_LITE_MICRO_KERNELS_PORTABLE_OPTIMIZED_POINTER_TABLES_POINTER_COLLECTOR_H_ */
