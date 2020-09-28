#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/core/lib/wav/wav_io.h"
//##include "tensorflow/lite/micro/kernels/micro_utils.h"
//#include "tensorflow/lite/micro/micro_string.h"

#include <cstring>

namespace tflite{
namespace ops {
namespace micro {
namespace decodewav {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

/**
 * Found in svdf.cc :
 * For TFlite Micro implementation:
 * - No more tensor allocation 
 * - Tensors must be known ahead of time for interpreter 
 * - Micro runtime does not support tensor resizing! 
 */


struct OpData{
    // Some Data used during Compute 
};

void* Init (TfLiteContext* context, const char* buffer, size_t length) {
    //Use of : context->AllocatePersistentBuffer !=nullptr to allocate Buffer for data 
    /*  if (context->AllocatePersistentBuffer(context, sizeof(OpData), &data) ==
      kTfLiteError) {
    return nullptr;
  }
  */
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node){
    // Where tensors are reinterpreted and functions are called, depending on the input type or value.
    
    // const float* matrix_ptr = matrix; ---> Instead of vector ????

    //     const int32_t output_max = std::numeric_limits<int16_t>::max();
    //     const int32_t output_min = std::numeric_limits<int16_t>::min();
    // Usage is okay. But why when it's obviously a const along all the program. Can be predefined.
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* ouput = GetOutput(context, node, kOutputTensor);
    

    TF_LITE_


}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node){
    //auto 
    // Check node->user_data
    // Check node-> builtin_data 
     
    // Mostly here the params are staic_casted from the used internal function 
    TFLITE_DCHECK(context != nullptr);
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

    const TfLiteTensor* contents = context;// Find a way to extract wav content, from input mabye input 0 ??:.

    TF_LITE_ENSURE(context, input != nullptr); // can be deleted 
    TF_LITE_ENSURE(context, output != nullptr);    

    //TF_LITE_ENSURE_EQ(context,)
    
    //Check the output and input dims and tensors. 


    TF_LITE_ENSURE_TYPES_EQ(context,contents,kTfLiteScalar) //Wrong
    TF_LITE_ENSURE_MSG(context,(contents.size())
    

}

}
} //namespace micro
} //namespace ops 
} // namespace tflite 