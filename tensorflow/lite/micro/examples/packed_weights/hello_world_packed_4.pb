node {
  name: "quant_dense_input"
  op: "Placeholder"
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
          dim {
            size: 1
          }
          dim {
            size: 16
          }
        }
        tensor_content: "n\267\313\2761\006\352>ai4>PF\237\275|%}\276]\315\370=\2623\032?\230\355\235\275\337\311^>W\305\337=\363]\275=3\3513>D{\001?\230VE>\376\"\310\273N/\020\277"
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense/MatMul/ReadVariableOp/resource"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
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
        float_val: -0.5634322166442871
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
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
        float_val: 0.5996513366699219
      }
    }
  }
}
node {
  name: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "Identity"
  input: "sequential/quant_dense/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "T"
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
      s: "CustomGradient-1641"
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
          dim {
            size: 16
          }
        }
        tensor_content: "\000\000\000\000*!\177=g\221\232\276\000\000\000\000\000\000\000\000\362\233|>#,w\276\000\000\000\000\230\036\217\276\363\301\212>\004\370\252>&\"\207\276\021\370v\276-*\214\276~\347G>\000\000\000\000"
      }
    }
  }
}
node {
  name: "sequential/quant_dense/BiasAdd/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense/BiasAdd/ReadVariableOp/resource"
  attr {
    key: "T"
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
  name: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
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
        float_val: -0.0019266719464212656
      }
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
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
        float_val: 3.2302706241607666
      }
    }
  }
}
node {
  name: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "Identity"
  input: "sequential/quant_dense/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "T"
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
      s: "CustomGradient-1663"
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/ReadVariableOp/resource"
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
          dim {
            size: 16
          }
          dim {
            size: 16
          }
        }
        tensor_content: "\204G\014<\211\345H\275v7\242\275\025r\332>\"O\'>\002M\010\276(\311b\275\326P\263=\345\320\031\275V\030`>:Z\212\276\371\356,=\306\206,=\244\301.=\205\033C\276QW\255\276C\336\266=\3078\225>L\234\264\274rJ\233=\301W9\276?N\317>\200\013\255\276\350\334\304>\346\205\302=e\305A\276\377\372n>\320\314\177\276\271I\302>\005\206\220>cd\301>\240\031>>\177\217\367\276:\220s>\314 \007>~\265\r?_P\326>\242I\200\276\010\002\203\275\204\205\013\276\225P|\2762~\206\276\030\032\210=\266&+\276q\277\304\276\200\366\017\275\331\306\300\276\311\031\323>-\005\306\276\227\225\007\276\377\271\010=\367\373r>h.\010=\006cJ=\231\255\315\276\225\376>>\226\226\304\276\030\255\255\276\374\000\014>\351\030i\276\342\333\204>#\263\245=3\244R\276\344V\005>G\030\325<\344\273\274>.?\323>k\250\304\275\240\215\240\275\304\361U>\200\236=<\006\204Y\276\363^\263\276\240\004\334\276\221\376\251>\023\244\273>\304b\201>\372\202\263\276:L\305>\0107\331>K\004\276>\032\244\343\276`B\231>D\247\300\275{\211{\276{\347\302\276\337\344\255\276F\337m\275\"o,>puO=\360\274\232\276 T\262>c\035\235>\020\201\316=\030!\303\274|\3565>\372\260q\276\206Y3\276Yw\036\277\200\311\261=\222\312\323\276\217\370\313=\263\201\221>\236\005\030\276-\033 \277\227E\326\276B\010b>L\331\376\276\205Z|>H\200\351=I\373\267\276\317\235\316>h j>\267UJ>a\220]\275\361\235\004\276\340\000B\274T\252a>$[\266=\242\333\255\275Bp >\003s\235>\351\256\225\276\001nX=\007\000@\276\351\004\271\276QC7=pB\007>\340\017\n\277:\277>\276\'\337\207\275y\214(>\177\375\215=\200\2733>t\217\027\276\201\243\010>\357\372\326\276\227\"\322>\207W\346=\241\333\334=0\' >\307<\354\276\230\3135\275k6h\275T\243\300><\341k>\276]\000?\211\334E=NY\\<\302\252%>P\031\217<\366I\330>\341 7\276\212\240\305\276\307\030\014>\nS\036?X\206\252\276\026\334%\2765\014|>\353N)\276(c\006>\247\213\267=\177\324\200>\241\344u>\344\376\302\2755\360\300\2768\304+\276]\250/\276\325\275\253>\300*\034=\013\214\241\273\023\201\235>\261X\274\276\361\241\000=\216\260\226>R\025\240\276\235\264\277\276$\272\321\276#v\n\276n\210\346>\214i\305\276\352Y\343>\244\352\241\276\210\272#\276\214F\037\275r\314\036>\333\207\366\274\205\370\344\276\222C\307\275\273Z\020\277\035C\240>i\260\256\275@\327\027>\321\352\'>ZQ\014\277\232\023\344>\330\301o=\212\217a>\324\003\314\275\355\353\215\276cNo\276\313v\275\276\016q\265>Ls\n\277\032d\241<!y$\276\014\'}\276\322\003G\275B\340}=cs\325\276\257f\023\276\361u\374>\272\317\242>\r\364j>\312\1770\276\274C\r\277\233\324\'\276\374V\271=\3738f\276\024)\320\276V\203\244\275\032\313\270\275\2245\246\276\031\272\344\275\265\202\264>\332S\304>W\002\331>\036\005\003\277\000K\356\272L8\213\276\331Z\321\276\304n\366>\225F\027?\000>^\272\366\343\223\276/JR=S\260\362>7&\217\275\204\375\013?\025\326\342\276@\256\210>7\222\211\276wc\223\273\026/\316\276\325D\206\276\370<\260> \260-\275\'\323)\276\n\034d>\266\316\003>\311\327\205\276;\205\322\2766\310\205>k3\343=\256[k\276\374\361j>"
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense_1/MatMul/ReadVariableOp/resource"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
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
        float_val: -0.6020475625991821
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
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
        float_val: 0.6164904832839966
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "Identity"
  input: "sequential/quant_dense_1/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "T"
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
      s: "CustomGradient-1682"
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
          dim {
            size: 16
          }
        }
        tensor_content: "\314\276->F!\001>(\307e>\216\343x\276\2510O\273\003Om\276\000\000\000\000j;4>\370\323\202>\000\000\000\000\254\201g\276L\204\233>\212Y\377=\361\222\327=\201\376H>\370\215\\\276"
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/BiasAdd/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense_1/BiasAdd/ReadVariableOp/resource"
  attr {
    key: "T"
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
  name: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
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
        float_val: -0.0019266719464212656
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
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
        float_val: 2.9002716541290283
      }
    }
  }
}
node {
  name: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "Identity"
  input: "sequential/quant_dense_1/oquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "T"
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
      s: "CustomGradient-1704"
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/ReadVariableOp/resource"
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
          dim {
            size: 16
          }
          dim {
            size: 1
          }
        }
        tensor_content: "\221\0236?\213\367\177>\220\240\254\276\242\001x\276\270\357\276=\234I*\277\253!\304\276#\r\021?\336`\276\276\370[;\276T\303\245\275\252\2432\277\273\002\305>\211a`?v\264\300>\3761\235\276"
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense_2/MatMul/ReadVariableOp/resource"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
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
        float_val: -0.6844921112060547
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp/resource"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
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
        float_val: 0.8531386852264404
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1"
  op: "Identity"
  input: "sequential/quant_dense_2/MatMul/kquant/FakeQuantWithMinMaxVars/ReadVariableOp_1/resource"
  attr {
    key: "T"
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
      i: 4
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
      s: "CustomGradient-1723"
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
          dim {
            size: 1
          }
        }
        float_val: 0.12537506222724915
      }
    }
  }
}
node {
  name: "sequential/quant_dense_2/BiasAdd/ReadVariableOp"
  op: "Identity"
  input: "sequential/quant_dense_2/BiasAdd/ReadVariableOp/resource"
  attr {
    key: "T"
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
