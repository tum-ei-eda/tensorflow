node {
  name: "quant_dense_input"
  op: "Placeholder"
  attr {
    key: "_user_specified_name"
    value {
      s: "quant_dense_input"
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense/MatMul/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/BatchMin"
  op: "Min"
  input: "sequential/quant_dense/MatMul/ReadVariableOp"
  input: "sequential/quant_dense/MatMul/kquant/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/Minimum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/Minimum"
  op: "Minimum"
  input: "sequential/quant_dense/MatMul/kquant/BatchMin"
  input: "sequential/quant_dense/MatMul/kquant/Minimum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/BatchMax"
  op: "Max"
  input: "sequential/quant_dense/MatMul/ReadVariableOp"
  input: "sequential/quant_dense/MatMul/kquant/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/Maximum"
  op: "Maximum"
  input: "sequential/quant_dense/MatMul/kquant/BatchMax"
  input: "sequential/quant_dense/MatMul/kquant/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "ReadVariableOp"
  input: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars"
  op: "FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense/MatMul/ReadVariableOp"
  input: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  input: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  attr {
    key: "narrow_range"
    value {
      b: false
    }
  }
  attr {
    key: "num_bits"
    value {
      i: 8
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/Identity"
  op: "Identity"
  input: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/IdentityN"
  op: "IdentityN"
  input: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense/MatMul/ReadVariableOp"
  attr {
    key: "T"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "_gradient_op_type"
    value {
      s: "CustomGradient-3819"
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul"
  op: "MatMul"
  input: "quant_dense_input"
  input: "sequential/quant_dense/MatMul/kquant/IdentityN"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense/BiasAdd/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense/BiasAdd/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/BiasAdd"
  op: "BiasAdd"
  input: "sequential/quant_dense/MatMul"
  input: "sequential/quant_dense/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sequential/quant_dense/Relu"
  op: "Relu"
  input: "sequential/quant_dense/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/BatchMin"
  op: "Min"
  input: "sequential/quant_dense/Relu"
  input: "sequential/quant_dense/oquant/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/Minimum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/Minimum"
  op: "Minimum"
  input: "sequential/quant_dense/oquant/BatchMin"
  input: "sequential/quant_dense/oquant/Minimum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/BatchMax"
  op: "Max"
  input: "sequential/quant_dense/Relu"
  input: "sequential/quant_dense/oquant/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/Maximum"
  op: "Maximum"
  input: "sequential/quant_dense/oquant/BatchMax"
  input: "sequential/quant_dense/oquant/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "ReadVariableOp"
  input: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars"
  op: "FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense/Relu"
  input: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  input: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  attr {
    key: "narrow_range"
    value {
      b: false
    }
  }
  attr {
    key: "num_bits"
    value {
      i: 8
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/Identity"
  op: "Identity"
  input: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/IdentityN"
  op: "IdentityN"
  input: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense/Relu"
  attr {
    key: "T"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "_gradient_op_type"
    value {
      s: "CustomGradient-3841"
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_1/MatMul/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/BatchMin"
  op: "Min"
  input: "sequential/quant_dense_1/MatMul/ReadVariableOp"
  input: "sequential/quant_dense_1/MatMul/kquant/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/Minimum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/Minimum"
  op: "Minimum"
  input: "sequential/quant_dense_1/MatMul/kquant/BatchMin"
  input: "sequential/quant_dense_1/MatMul/kquant/Minimum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/BatchMax"
  op: "Max"
  input: "sequential/quant_dense_1/MatMul/ReadVariableOp"
  input: "sequential/quant_dense_1/MatMul/kquant/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/Maximum"
  op: "Maximum"
  input: "sequential/quant_dense_1/MatMul/kquant/BatchMax"
  input: "sequential/quant_dense_1/MatMul/kquant/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars"
  op: "FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense_1/MatMul/ReadVariableOp"
  input: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  input: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  attr {
    key: "narrow_range"
    value {
      b: false
    }
  }
  attr {
    key: "num_bits"
    value {
      i: 8
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/Identity"
  op: "Identity"
  input: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/IdentityN"
  op: "IdentityN"
  input: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense_1/MatMul/ReadVariableOp"
  attr {
    key: "T"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "_gradient_op_type"
    value {
      s: "CustomGradient-3860"
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul"
  op: "MatMul"
  input: "sequential/quant_dense/oquant/IdentityN"
  input: "sequential/quant_dense_1/MatMul/kquant/IdentityN"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense_1/BiasAdd/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_1/BiasAdd/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/BiasAdd"
  op: "BiasAdd"
  input: "sequential/quant_dense_1/MatMul"
  input: "sequential/quant_dense_1/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sequential/quant_dense_1/Relu"
  op: "Relu"
  input: "sequential/quant_dense_1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/BatchMin"
  op: "Min"
  input: "sequential/quant_dense_1/Relu"
  input: "sequential/quant_dense_1/oquant/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/Minimum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/Minimum"
  op: "Minimum"
  input: "sequential/quant_dense_1/oquant/BatchMin"
  input: "sequential/quant_dense_1/oquant/Minimum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/BatchMax"
  op: "Max"
  input: "sequential/quant_dense_1/Relu"
  input: "sequential/quant_dense_1/oquant/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/Maximum"
  op: "Maximum"
  input: "sequential/quant_dense_1/oquant/BatchMax"
  input: "sequential/quant_dense_1/oquant/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars"
  op: "FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense_1/Relu"
  input: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  input: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  attr {
    key: "narrow_range"
    value {
      b: false
    }
  }
  attr {
    key: "num_bits"
    value {
      i: 8
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/Identity"
  op: "Identity"
  input: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/IdentityN"
  op: "IdentityN"
  input: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense_1/Relu"
  attr {
    key: "T"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "_gradient_op_type"
    value {
      s: "CustomGradient-3882"
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_2/MatMul/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/BatchMin"
  op: "Min"
  input: "sequential/quant_dense_2/MatMul/ReadVariableOp"
  input: "sequential/quant_dense_2/MatMul/kquant/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/Minimum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/Minimum"
  op: "Minimum"
  input: "sequential/quant_dense_2/MatMul/kquant/BatchMin"
  input: "sequential/quant_dense_2/MatMul/kquant/Minimum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/BatchMax"
  op: "Max"
  input: "sequential/quant_dense_2/MatMul/ReadVariableOp"
  input: "sequential/quant_dense_2/MatMul/kquant/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/Maximum"
  op: "Maximum"
  input: "sequential/quant_dense_2/MatMul/kquant/BatchMax"
  input: "sequential/quant_dense_2/MatMul/kquant/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars"
  op: "FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense_2/MatMul/ReadVariableOp"
  input: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  input: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  attr {
    key: "narrow_range"
    value {
      b: false
    }
  }
  attr {
    key: "num_bits"
    value {
      i: 5
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/Identity"
  op: "Identity"
  input: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/IdentityN"
  op: "IdentityN"
  input: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars"
  input: "sequential/quant_dense_2/MatMul/ReadVariableOp"
  attr {
    key: "T"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "_gradient_op_type"
    value {
      s: "CustomGradient-3901"
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul"
  op: "MatMul"
  input: "sequential/quant_dense_1/oquant/IdentityN"
  input: "sequential/quant_dense_2/MatMul/kquant/IdentityN"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "sequential/quant_dense_2/BiasAdd/ReadVariableOp/resource"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_RESOURCE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "sequential/quant_dense_2/BiasAdd/ReadVariableOp/resource"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_2/BiasAdd"
  op: "BiasAdd"
  input: "sequential/quant_dense_2/MatMul"
  input: "sequential/quant_dense_2/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sequential/output/FakeQuantWithMinMaxArgs"
  op: "FakeQuantWithMinMaxArgs"
  input: "sequential/quant_dense_2/BiasAdd"
  attr {
    key: "max"
    value {
      f: 0.9921875
    }
  }
  attr {
    key: "min"
    value {
      f: -1.0
    }
  }
  attr {
    key: "narrow_range"
    value {
      b: false
    }
  }
  attr {
    key: "num_bits"
    value {
      i: 8
    }
  }
}
node {
  name: "Identity"
  op: "Identity"
  input: "sequential/output/FakeQuantWithMinMaxArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
versions {
  producer: 175
}
