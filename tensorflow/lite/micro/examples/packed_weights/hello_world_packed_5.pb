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
        tensor_content: "\030T\240\276\227\352\236\276P\214\300=x\016\244>\235M\">\357\277\010?\0060\212\276\037\377\364=8\371\264\276k\353\352>\300\325\023\274\000\021o\273\342\354\016\277J\350j>\366Y\n>B\274\261\276"
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
        float_val: -0.5585147142410278
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
        float_val: 0.5213661789894104
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
        tensor_content: "\000\000\000\000\000\000\000\000D\246V\2766\267\257\276n_\231\276\241\341&\276\000\000\000\000\270\013(>\000\000\000\000@}E=\000\000\000\000\000\000\000\000\000\000\000\000E\373\247\2766b\222\276\000\000\000\000"
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
        float_val: 2.7960045337677
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
      s: "CustomGradient-3841"
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
        tensor_content: "\3326P>G\266\236\276\372\230\301\276p\317\211\276;0\204>V\3744>\006\342\240\275\200\213J<\312\217\275\276\320,\t\2761\362]>\322\325\231\276c\226Q\276\265b\335=n+\t>\355\232\206\276~Nb>zRR\275\242j\230\276\254\262\250=1M\277>\205\006~\276\362\303\321=\306);\276:5\233>*\356\206\276K\251\224>\3178\003>\025\332\306\276\3573q>\034\262\261\276Z\247\256\276\241\274\341\275\317\t\000\277R?\307\276\344\224\365\275\300\030\020\277\262&\032\276\343\026\r?\316\2779\276\227\311\303>\246C\'\276\224\213\032\276\025\030.\277}@\216\276P_3\274\035O\264>Bq\324>\\\037\000\277\343)\232\276\034\001\303\276\261\233\315\275GG,\276\251[\262\275\371\321\355>\247I\245>o\001\013\276\031\347\257>\352\314\213\276\223\267\257>\370=H>[\331\365\275\371\233F\276A\307\364>\3154i>\326\256\256\275\250\226\315\276\212U\220\276\361\225\342<\3146\210\276_|\304>\t\341\261\276\321\374\346>\204|\206\275\\\214%\276P\221\223\276\204\301;>\005w\247\276P\300\345<w<\020?\307F\017\277N&2=\325!\315\275Q(\036\276\374\354\221\274_\255\276>\205\352\211>\370\030\222\276\032\273\357>C\354\322\276M\312\034>P\335\221\276\2352`\276\232\255\264=\373\310\301>\274\302\245\274\216ya>\366\025\256\275\034<\216\276o\264\327>\207c\340=\331\270\275\276~M\016\275,\364\363=\346\303]\276\313\247\234>\005\333M>.\367\313\276\273\360\212\276\265\211\245>h\3139=\334\013J\276.J\321>\201%\272>\372\303 \276\252E\246\276\252\304\021>\212\371\203>~\\O< {0\275\334\271\306\276\362{\257\276\3373Y>\317g\262>\341\276\272>\321\211\233>0\212\274<\227\234e>\027\025\202>\327\202\236\276M\314\313>\"(-\276\303\214\223>\342ji>\302\253,>C\226\203>K\314\323\275[T\305>\213\nN=\227\310\310=\274\326\240>\22769>\310\272\266\276\234Z\201\275[\224\231\275\t\251\210<\032y6>\316\0211>J\222R>\016\332\277\274\214W5<otp\276\377\017\263=\230S\356\275\274\335\213>\347\'\306>\226\324{\276\354r\304>\014\316\263\276\265\312\250>\227\255\274>\341Zf\276l\017\\=\200O\234\276\234p\034>&k\"\2762\231\213\276\2007B\275\203\373C\276\237M\315>6do\276K\317G>\3205\321=%\320C\276D\373f\276\020\225\321=\264\333\225\276({\210>\003\232\244=0\017\303\276i\361k\276\276\362\004\276\035\376\267><\013\303\276M\324\214\276\276\031C>\025|e\276p\027H\276\370\263{>EUQ\276\267V\202>\332y\222>\331\323\254>\034H\320>6\036!\276\340\006\320\2752}`\274+9\223\275f\337\207\275\220\320\261\275\375\212\331=\">D\276\327\213\214\276\273\037:>\014b\304>\337\322a>6\277\331\276\032\2568>\312\201\026\277\376?S=Io\323\276\254\024\314\276\322\365\250\276\272\361\315\276W\354\327>\365\014\227>\273\003\355<*\343\000>\300\034\324\276B#T\276\\XT>\376\035\273\276\202UR\276B\335\374>N\235z\276\301\305\316\276\000\300\302\274\372\260v>%V\351\276\343L\026\277`\340\317>\220g\333<\303\215\305\275\342\036\"\276\307~O\275\261\333Q>E4L\276\376\366\016\277]\346\236\276_/\033?\320\276\001\276D\273\026\276h\023\252\276t\337\327=d-\'\276\356 H\275\272\014\265=\302o>>)\337\216\276\275\360\203>\315\353\262\276r\325\212\276\230]\274\275\252O\256\276\270\305<\275Z\371\377\275"
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
        float_val: -0.6674710512161255
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
        float_val: 0.5937161445617676
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
        tensor_content: "\177\207\244>+_\332=\372\327\363\274-q\371:\031_\304=\037\251\324=V~\242\276\000\000\000\000`\372\201\276\000\000\000\000k5\323=\242\244\320=\371\210S\275\t\272\323=\000\000\000\000\224\025\254\276"
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
        float_val: 2.7135255336761475
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
      s: "CustomGradient-3882"
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
        tensor_content: "L\023,\277\323\215\313>\335\251i>\271\211\220\276\317~\273>o\224\300>\006\250<\2779\246\307\276t\315D\277\210\356\261>C\206\005?\3516\263>\020\273\021>\340%\006?\257\352y\2768\203T\276"
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
        float_val: -0.7582488059997559
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
        float_val: 0.513282835483551
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
      i: 5
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
        float_val: 0.10332982242107391
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
