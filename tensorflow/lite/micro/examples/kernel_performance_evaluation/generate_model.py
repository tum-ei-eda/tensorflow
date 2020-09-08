# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

#%%
import os
import sys
import shutil
import tensorflow as tf
# Numpy is a math library
import numpy as np
# Matplotlib is a graphing library
import matplotlib.pyplot as plt
# math is Python's math library
import math
# We'll use Keras to create a simple model architecture
from tensorflow.keras import layers
import utils
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten
from kquant.layers import QuantDense, QuantConv2D, QuantDepthwiseConv2D
from kquant.quantizer import TFMinMaxQuantizer, TFEmaMinMaxQuantizer
from kquant.tflite_utils.conversion import translate_from_keras_model, write_C_array_src_and_hdr
from kquant.tflite_utils.interpreter import FlatBufModel
from kquant.tflite_utils.quantization import uniform_quant_params, uniform_encode
from tensorflow.keras.datasets import mnist

# %% [markdown]
# ## Generate data
# Deep learning networks learn to model patterns in underlying data. In this notebook, we're going to train a network to model data generated by a [sine](https://en.wikipedia.org/wiki/Sine) function. This will result in a model that can take a value, `x`, and predict its sine, `y`.
# 
# In a real world application, if you needed the sine of `x`, you could just calculate it directly. However, by training a model to do this, we can demonstrate the basic principles of machine learning.
# 
# In the [hello_world](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/hello_world) sample for [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers/overview), we'll use this model to control LEDs that light up in a sequence.
# 
# The code in the following cell will generate a set of random `x` values, calculate their sine values, and display them on a graph:

# %%
# We'll use this many sample datapoints
SAMPLES = 1000
reload = True
translator = "C:/Inicio/tools/64/tflite_u-2.2.0.2/bin/tf_tfl_translate"


# Load the Mnist dataset and keep only zeros and ones
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
train_filter = np.where((Y_train == 0 ) | (Y_train == 1))
test_filter = np.where((Y_test == 0) | (Y_test == 1))
X_train, Y_train = X_train[train_filter], Y_train[train_filter]
X_train = np.expand_dims(X_train, axis=3)/255.0
print(X_train.shape)
Y_train = tf.keras.utils.to_categorical(Y_train, 2)
X_test, Y_test = X_test[test_filter], Y_test[test_filter]
X_test = np.expand_dims(X_test, axis=3)/255.0
Y_test = tf.keras.utils.to_categorical(Y_test, 2)
# Set random seed for reproducibility
tf.random.set_seed(1)
# %% [markdown]
# ## Add some noise
# Since it was generated directly by the sine function, our data fits a nice, smooth curve.
# 
# However, machine learning models are good at extracting underlying meaning from messy, real world data. To demonstrate this, we can add some noise to our data to approximate something more life-like.
# 
# In the following cell, we'll add some random noise to each value, then draw a new graph:


# %% [markdown]
# ## Split our data
# We now have a noisy dataset that approximates real world data. We'll be using this to train our model.
# 
# To evaluate the accuracy of the model we train, we'll need to compare its predictions to real data and check how well they match up. This evaluation happens during training (where it is referred to as validation) and after training (referred to as testing) It's important in both cases that we use fresh data that was not already used to train the model.
# 
# To ensure we have data to use for evaluation, we'll set some aside before we begin training. We'll reserve 20% of our data for validation, and another 20% for testing. The remaining 60% will be used to train the model. This is a typical split used when training models.
# 
# The following code will split our data and then plot each set as a different color:
# 

# %%

def weight_q(name, **kwargs):
    return TFEmaMinMaxQuantizer(
        name=name, init_min=-1.0, init_max=1.0, quant_delay=100,  num_bits=8, decay=0.99, **kwargs
    )

def weight_q_sub8( num_bits ):
    def qtizer(name, **kwargs):
        return TFEmaMinMaxQuantizer(
            name=name, init_min=-1.0, init_max=1.0, quant_delay=100,  num_bits=num_bits, decay=0.99, **kwargs
        )
    return qtizer


def act_q(name, **kwargs):
    return TFEmaMinMaxQuantizer(
        name=name, init_min=-4.0, init_max=4.0, quant_delay=100, num_bits=8, decay=0.99, **kwargs
    )



def quant_mnist_model(qbits):
    model = tf.keras.Sequential()

    model.add(QuantConv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=(28, 28, 1), kernel_quantizer=weight_q_sub8(qbits)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(QuantDepthwiseConv2D(kernel_size=(3,3), depth_multiplier=2, activation='relu', padding='same', depthwise_quantizer=weight_q_sub8(qbits)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(QuantDepthwiseConv2D(kernel_size=(3,3), activation='relu', padding='same', depthwise_quantizer=weight_q_sub8(qbits)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #model.add(layers.Lambda( lambda x: tf.identity(x), name="tap_input", input_shape=(1,)))
    # First layer takes a scalar input and feeds it through 16 "neurons". The
    # neurons decide whether to activate based on the 'relu' activation function.
    model.add(Flatten())
    model.add(QuantDense(64, activation='relu', 
                        kernel_quantizer=weight_q_sub8(qbits),
                        output_quantizer=act_q))

    # The new second layer may help the network learn more complex representations
    # Deliberate error with have an input_quantizer despite the previous layer
    # habing and output quantizer...
    model.add(QuantDense(2, activation='softmax',
                        kernel_quantizer=weight_q_sub8(qbits), output_quantizer=None))


    # We need an output fq layer to get final quantization for converter
    model.add(TFMinMaxQuantizer(name="output",
                                qmin=-1.0, qmax=127 / 128.0, quant_delay=100, num_bits=8,
                                narrow_range=False))
    model.summary()
    return model

q_params = {}
f_weights = {}
weight_codes = {}
#%%
QBITS = [8] # 4,5,6,
for qbits in QBITS:
    # 
    tf.keras.backend.clear_session()
    sess = tf.compat.v1.keras.backend.get_session()


    model_q = quant_mnist_model(qbits)
    #model_ref = hello_world_model()


    BATCH_SIZE = 16
    #model_ref.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model_q.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # 
    # And train....
    logsq = os.path.join(f"logs_{qbits}")
    shutil.rmtree(logsq, ignore_errors=True)
    os.makedirs(logsq, exist_ok=True)
    tensorboard_q = tf.keras.callbacks.TensorBoard(log_dir=logsq)

    weights_save_path = os.path.join("saves", f"mlir_translator_test_{qbits}")
    if reload and os.path.exists(weights_save_path+".index"):
        model_q.load_weights(weights_save_path)
    else:
        history_q = model_q.fit(X_train, Y_train, epochs=10, batch_size=BATCH_SIZE,
                                validation_data=(X_test, Y_test),
                                callbacks=[tensorboard_q])
        model_q.save_weights(weights_save_path)

    #for l in model_q.layers:
    #    if hasattr(l,'output_quantizer'):
    #        print( l.kernel_quantizer.name, l.kernel_quantizer.max_ema.get_weights())
    #    if hasattr(l, "max_ema"):
    #        print(l.max_ema.name, l.max_ema.get_weights())


    #qscale, qzero = uniform_quant_params(-1, 127/128.0, bits=8, narrow_range=False)
    #max_out = model_q.layers[-1].max_ema.non_trainable_weights[0]
    #min_out = model_q.layers[-1].min_ema.non_trainable_weights[0]

    # %% [markdown]
    # ## Evaluate our new model
    # Each training epoch, the model prints out its loss and mean absolute error for training and validation. You can read this in the output above (note that your exact numbers may differ):
    #
    # ```
    # Epoch 600/600
    # 600/600 [==============================] - 0s 109us/sample - loss: 0.0124 - mae: 0.0892 - val_loss: 0.0116 - val_mae: 0.0845
    # ```
    #
    # The following cell will print the same graphs showing our new training history:
    # You can see that we've already got useavble results - validation loss his 0.015, and validation MAE is  0.1.
    #

    # 


    opt_raw_graph = utils.conversion_raw_graph_def(model_q)

    #pb_pname = "hello_world_raw.pb"
    #pb_dir, pb_fname = os.path.split(pb_pname)
    #tf.io.write_graph( opt_raw_graph, pb_dir, pb_fname, as_text=True)


    # 
    #tf.keras.backend.clear_session()
    #sess = tf.compat.v1.keras.backend.get_session()
    #model_ref = tf.keras.models.load_model('hello_world.h5')

    # Calculate and print the loss on our test dataset
    loss_q = model_q.evaluate(X_test, Y_test)
    pred_y = model_q.predict(X_test)



    # [markdown]
    # The evaluation metrics we printed show that the model has a low loss and MAE on the test data, and the predictions line up visually with our data fairly well even with quantized weights.
    #
    # The model isn't perfect; its predictions don't form a smooth sine curve. For instance, the line is almost straight when `x` is between 4.2 and 5.2. If we wanted to go further, we could try further increasing the capacity of the model, perhaps using some techniques to defend from overfitting.
    #
    #
    # ## Convert to TensorFlow Lite
    # We now have an acceptably accurate model in-memory. However, to use this with TensorFlow Lite for Microcontrollers, we'll need to convert it into the correct format and download it as a file. To do this, we'll use the [TensorFlow Lite Converter](https://www.tensorflow.org/lite/convert). The converter outputs a file in a special, space-efficient format for use on memory-constrained devices.
    #
    # * Save the trained model using TF mechansims.
    # * Build a second copy of the quantized model (without training machinery)
    # * Load variables from the saved model
    # * ... and then finally freeze the result.  Ouch!
    # #



    tocolog = os.path.join(logsq, "toco_log")
    shutil.rmtree(tocolog, ignore_errors=True)
    os.makedirs(tocolog, exist_ok=True)

    if qbits < 8:
        type_suff = "_packed"
    else:
        type_suff = ""
    mlir_fname_root = f"mnist{type_suff}_{qbits}"
    tflite_u_fname = mlir_fname_root+".tflite"
    translate_from_keras_model(model_q, mlir_fname_root, 0.0, 1.0, use_toco=False,
                               constant_folding=False, translator=translator,
                               tool_cmdline_args={'experimental-pack-packable-quant-constants':'true'})
    if qbits < 8:
        # Create flatbuffer without actually packing narrow weights that is runnable on standard tflite
        # as a reference
        mlir_tflite_fname_root = f"mnist_unpacked_{qbits}"
        tflite_fname = mlir_tflite_fname_root+".tflite"
        translate_from_keras_model(model_q, mlir_tflite_fname_root, 0.0, 1.0, use_toco=False,
                                constant_folding=False, translator=translator,
                                tool_cmdline_args={'experimental-pack-packable-quant-constants':'false'}
                                )
    else:
        tflite_fname = tflite_u_fname

    write_C_array_src_and_hdr(f"{qbits}-Bit weight 'mnist' model", ".", mlir_fname_root,
                              aligned=True,
                              src_datafile=tflite_u_fname)
    print(model_q.layers)
    sys.exit(0)
    layer_vars = model_q.layers[-2].get_weights()
    f_weights[qbits] = [v for v in layer_vars[0].flatten()]
    fq_min = layer_vars[3]
    fq_max = layer_vars[4]

    q_params[qbits] = uniform_quant_params(fq_min, fq_max, bits=qbits)

    weight_codes[qbits] = uniform_encode(f_weights[qbits], scale_zero=q_params[qbits], bits=qbits)
    with open(mlir_fname_root+ "refdata.h", "w") as res:

        res.write(f"float {mlir_fname_root}_refdata[2][28][28][1] = {{\n")
        test_data = [np.expand_dims(X_test[0], axis=0), np.expand_dims(X_test[1], axis=0)]
        results = [[0,1], [1,0]]
        for i, x in enumerate(test_data):
            res.write(f"{{")
            for dim2 in range(28):
                res.write(f"{{")
                for dim3 in range(28):
                    res.write(f"{{{x[0][dim2][dim3][0]} }},")
                res.write(f"}},\n")
            res.write(f"}},\n")
        res.write(f"}};\n\n")
            
        res.write(f"float {mlir_fname_root}_refdata_label[2][2] = {{\n")
        for i, x in enumerate(test_data):
            y = model_q.predict([x])
            print(f"{y} should be close to {results[i]}")
            res.write( f"    {{{results[i][0]},{results[i][1]}}},\n")
        res.write(f"}};")

# %%
for qbits in QBITS:

    print("QUANT BITWIDTH:", qbits)
    print("WEIGHT PARAMS :", q_params[qbits])
    print("WEIGHT VALUES :", f_weights[qbits])


sys.exit(0)
