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
        tensor_content: "\214G\007?\243\375\n?\325,\245=[\350\267>\003B\371\2763\n\306>^q\252>\337\202\376>Q\016\327\276qV\321>\274F\310\275\002Zt>\312\274]\276\332\264\203>\340\352\017\277#\350;>"
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
        float_val: -0.56238853931427
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
        float_val: 0.5419201850891113
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
      s: "CustomGradient-5997"
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
        tensor_content: "\021\310\221\276VE\204\276+\252;\276\347\335\304=\000\000\000\000\025\3328\276\353\264\224\276\344ef>\000\000\000\000d\346\235\276\000\000\000\000\247v\241\276\000\000\000\000] \204\276\000\000\000\000\357\017k>"
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
        float_val: 3.119103193283081
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
      s: "CustomGradient-6019"
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
        tensor_content: "\237\037\'\276\005\205\303>\024v9\274w8\r\276V\241O\276*h\211\276\000\240Z\276\220\350\377\275\233F\305>C;\217\276\227\004\336>7,\206>\265\327\340<\372\237\357\275\010\231\356<gy\307>\253\250\316=\332&\000=\030\215&\276*\002\224>~\332\025\276QH\335\275\000c\307\275\263\344\000\276T\364\251=\376\2218>\022\025\010>N\230\362=@\367o=\355\374~\276\021l\331\276fj!\276!\342\301<{\256x\276g\251\262\276e`\372=\023Y\316\276R\177\335\276`\374|<\2462\307<~f\022?\014_\024\277\255\330\257\276k\213\001?\266\257\320>\n!\324\276\313g\371\275^\270Y>\346Bw\276\333\257,\276\3649\031>\206\253\025\275\340\377q\275\241w\017;\357\257\300\276\314\253\325>;F\025>\237\371\r>yZ\026>\315?\'<K#W>\243\003\037\275tL\t\275\346\217C>\272\263i\275+\331\036\276\260\321[>\255H\222=7\253\314>\346g\004\276\315F\273>\346\244\211>\260W\276\276t/\316\275m{\256\276Bd\222\276\031\353\322\275\237\005\254>\273\221<\276\016\020Y>\264\300\026\276\377\014\241>\306\224\230>\307)Z>\222\301}>\\\3504\277k\373\312>tB^\276X\020r<pF4>\224C\247\272\267\363\013>\351\345\300>)\363\371=\375y\311\276\231;\225>\340\370Y\276M\031x>l|h\2765\311\276>\212\347\224\276Q\375\t\277\272\332F>>3\322\276n*\202\275>l\271>\005\031\270\276\035\037\202>\321\010\322=\370\334\302=\026\205\203\276(\223\270\276\323\327g>\376`\340\276\216\246\245>g\270\031\276\377\232`\276@\036T>\312Z\240\275Z\266\266>\271\001\264\275\037X\222>Bd\243>\323\227\014\276YG\306\275k3\201\276\240\320\274\275g\034\334\276`\305\302>)\276\245=<\233$>&\207\321>\244*\032\276\270\202\260>v\313~\276z\220\217>\201\327`>\364Y\233<r\371\222>\375\242\261\275u2\356\275\363d!\276\325\307\374=\307j\211>\352G\273\276\002\002\226>9_\214\276\247U\007\276`\023\221\276\2555\244\276\201\241\237\276\241\002\035\2761\303\204\276\"\024\300\276I/\354=wX\323>\2464\302>\343\306\377\276\023\367h\275\000\3121\274D\357~\276]\270\036>\241Q\230\276\202K\202>\0055\261>o(o\275hEP=d\347r>\331\337R>\007\201\217\276;[m\276ux\302>A\020\300\276\177\213\211>\333\177\263\276\370l+=\326\004?>\266\315D=:\217\223\276_\245\264>yx\270>\243^\037>v)\334\276!\022\250\276\014\233\347=.\366j\275K\223\306\276h\210\250=.\202\236=\375jV>=6\250=`\330\327\275l\037\274=?\354\335\275\014\377\375=m\367\243=N\357\331\276\334\007\331>KC\270>\\\316\025\276\026f)=\237x,\275\177\261\245=\202\010\233\276\353\341\230>\376z\276\276*a\204\276\003\036=\276Yz&\276\234\024*>q,\250\276\2152\324\275\244*\301=\247\302\014\277\036\356c>\371\366|=C\313\267>p\245\001>\031vs\276ZO\335>2>x\276\254\343@\276\"MU\276\340Pg\275%\244\001>\316\301\325\276(\232\303\275T\321\215\276\3120\304\276\250\324~\276\376\300P>\324@a>_vZ\274:\304@\276yx\034>cV\252>\350\336\272\276q\026\203\2749\224\323\276\316\377a>(\224@\275>\003\227>\362 \355=\203\356t>F\310\202\276\340\377\375=\310u\177>C\345;>\t\360\340\275\341\224\311>\330\033e\276\3358\352\276\342Q\004\274/d\347>\320c\262>(\260/\275"
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
        float_val: -0.6687172055244446
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
        float_val: 0.5596206784248352
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
      s: "CustomGradient-6038"
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
        tensor_content: "\3758\003\275\033\361\205\276\224\375\010>\000\362\203\276\000\000\000\000\241)\263>\363;\317\273\236\026\020>u\235\205\276\302L\034>\221\350$>|\377\226\276\202\351\237\276t\353\'>6\217\177\275\000\000\000\000"
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
        float_val: 2.122218608856201
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
      s: "CustomGradient-6060"
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
        tensor_content: "\026\226;<J\235\003\277\355d\t?\033\274-\277\005\322\366=\272\3572\277~K\267>\035*\007?J#\227\276.\251\343>\243v;>iH\026\277\276\275\230\276\\S\"\277||\242=C\033Y>"
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
        float_val: -0.6761047840118408
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
        float_val: 0.5257765054702759
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
      i: 6
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
      s: "CustomGradient-6079"
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
        float_val: 0.15085484087467194
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
