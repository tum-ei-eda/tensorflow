#include "tensorflow/lite/c/common.h"
#include <cstdint>
#include <memory>



// Encapsulates strategies for access/addressing, allocation/deallocation and
// construction/destruction of objects. Using C++11 Implementation for
// simplicity and backwords comp.
//      -> Using allocator_traits
// Goal : minimal allocator to use TFLM Scratch Buffer.

namespace tflite {
namespace ops {
namespace micro {

template <typename T> class ArenaBufferAllocator {
private:
  T *memory_ptr;
  std::size_t memory_size;
  TfLiteContext *context;
  bool allocated; 

public:
  typedef std::size_t size_type;
  typedef T *pointer;
  typedef T value_type;
  bool allocated_init; // Init/Prepare = 0 / Eval = 1
  void *raw;

  // Other members for the allocator (optional)
  typedef const T *const_pointer;
  typedef ptrdiff_t difference_type;
  typedef T &reference;
  typedef const T &const_reference;
  typedef void *void_pointer;
  typedef const void *const_void_pointer;

/*   typedef propagate_on_container_copy_assignment = false_type;
  typedef propagate_on_container_move_assignment = true_type;
  typedef propagate_on_container_swap = true_type; */


  // Construction
  ArenaBufferAllocator() noexcept {}; // Needed for initilisation. More testing

  ArenaBufferAllocator(void *ptr, std::size_t memory_size,
                       TfLiteContext *ctx) throw()
      : memory_size(memory_size), context(ctx), allocated_init(true) {
    memory_ptr = static_cast<T *>(ptr);
    context->ReportError(context, "Inside Allocator Init");
  };
  ArenaBufferAllocator(int ptr_int, std::size_t memory_size,
                       TfLiteContext *ctx) throw()
      : memory_size(memory_size), context(ctx), allocated_init(false) {
    raw = context->GetScratchBuffer(context, ptr_int);
    context->ReportError(context, "Inside Allocator Eval");
  };

  ArenaBufferAllocator(const ArenaBufferAllocator &other) noexcept {}
  //: memory_ptr(other.memory_ptr), memory_size(other.memory_size){};

  template <class U>
  ArenaBufferAllocator(const ArenaBufferAllocator<U> &other) noexcept {}
  //: memory_ptr(other.memory_ptr), memory_size(other.memory_size){};

  ~ArenaBufferAllocator() {}

  pointer address(reference value) const { return &value; }
  const_pointer address(const_reference value) const { return &value; }

  // For vector allocation: happepns every 2^n elems. We don't want that.
  // Reallocation not allowed if allocated first time.
  // Use limit allowed as max_size
  pointer allocate(size_type n, const void *hint = 0) {
/*     context->ReportError(context,"What?");
    context->ReportError(context, "In allocate,size(%d) mem_size %d", n,
                         memory_size);

    context->ReportError(context, "Init/Prepare?: %d", allocated_init); */
    if (n > memory_size) {
      //context->ReportError(context, "Bad Memory size in allocation, got %d", n);
      throw std::bad_array_new_length();
    }
    if (allocated_init == true) {
      return memory_ptr;
    } else if (allocated_init == false) {
      return reinterpret_cast<T *>(raw);
    }
    return memory_ptr;
  }
  void deallocate(T *ptr, size_type n) {}

  size_type max_size() const { return memory_size; }

    template <typename U> struct rebind {
      typedef ArenaBufferAllocator<U> other;
    };

    template <class U>
    ArenaBufferAllocator &operator=(const ArenaBufferAllocator<U> &) {
      return *this;
    }
/*     ArenaBufferAllocator<T> &operator=(const ArenaBufferAllocator &other) {
      return *this;
    } */
};

template <typename T, typename U>
bool operator==(const ArenaBufferAllocator<T> &lhs,
                const ArenaBufferAllocator<U> &rhs) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(const ArenaBufferAllocator<T> &lhs,
                const ArenaBufferAllocator<U> &rhs) noexcept {
  return false;
}


} // namespace micro
} // namespace ops
} // namespace tflite */

