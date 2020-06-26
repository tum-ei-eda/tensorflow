# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Simple graph freezing and export

"""

import os.path
import sys
import tensorflow as tf
from tensorflow.lite.python.util import get_grappler_config as _get_grappler_config
from tensorflow.lite.python.util import get_tensor_name as _get_tensor_name
from tensorflow.lite.python.util import get_tensors_from_tensor_names as _get_tensors_from_tensor_names
from tensorflow.lite.python.util import is_frozen_graph as _is_frozen_graph
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.eager import def_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.util import nest
from tensorflow.python.keras.engine import base_layer_utils

def trace_model_call(model, input_signature=None):
  """Trace the model call to create a tf.function for exporting a Keras model.

  Args:
    model: A Keras model.
    input_signature: optional, a list of tf.TensorSpec objects specifying the
      inputs to the model.

  Returns:
    A tf.function wrapping the model's call function with input signatures set.

  Raises:
    ValueError: if input signature cannot be inferred from the model.
  """
  if input_signature is None:
    if isinstance(model.call, def_function.Function):
      input_signature = model.call.input_signature

  if input_signature is None:
    input_signature = model_input_signature(model)

  if input_signature is None:
    raise_model_input_error(model)

  # TODO(mdan): Should the model's call be autographed by default?
  @def_function.function(input_signature=input_signature, autograph=False)
  def _wrapped_model(*args):
    """A concrete tf.function that wraps the model's call function."""
    # When given a single input, Keras models will call the model on the tensor
    # rather than a list consisting of the single tensor.
    inputs = args[0] if len(input_signature) == 1 else list(args)

    with base_layer_utils.call_context().enter(
        model, inputs=inputs, build_graph=False, training=True, saving=True):
      outputs_list = nest.flatten(model(inputs=inputs, training=True))

    try:
      output_names = model.output_names
    except AttributeError:
      from tensorflow.python.keras.engine import training_utils  # pylint: disable=g-import-not-at-top
      output_names = training_utils.generic_output_names(outputs_list)
    return {name: output for name, output in zip(output_names, outputs_list)}

  return _wrapped_model

def conversion_raw_graph_def(model, constant_folding=False):


    """Prepares a TensorFlow GraphDef for toco conversion from keras model

    Args:
        constant_folding    Apply Grappler to fold constants in graphdef
                            !!DO NOT USE FOR TF2.1 TOCO!!
    Returns:
      The graphs, input tensors, output tensors data in serialized format.


    """


    if not isinstance(model.call, def_function.Function):
        # Pass `keep_original_batch_size=True` will ensure that we get an input
        # signature including the batch dimension specified by the user.
        input_signature = saving_utils.model_input_signature(
            model, keep_original_batch_size=True)

    func = trace_model_call(model, input_signature)
    # This seems to implicitly get the training=False variant
    concrete_func = func.get_concrete_function()

    return concrete_func.graph.as_graph_def()


def conversion_graph_def(model, constant_folding=False):


    """Prepares a TensorFlow GraphDef for toco conversion from keras model

    Args:
        constant_folding    Apply Grappler to fold constants in graphdef
                            !!DO NOT USE FOR TF2.1 TOCO!!
    Returns:
      The graphs, input tensors, output tensors data in serialized format.


    """


    if not isinstance(model.call, def_function.Function):
        # Pass `keep_original_batch_size=True` will ensure that we get an input
        # signature including the batch dimension specified by the user.
        input_signature = saving_utils.model_input_signature(
            model, keep_original_batch_size=True)

    func = saving_utils.trace_model_call(model, input_signature)
    # This seems to implicitly get the training=False variant
    concrete_func = func.get_concrete_function()

    raw_graph_def = concrete_func.graph.as_graph_def()
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(
        concrete_func, lower_control_flow=False)
    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != dtypes.resource
    ]
    output_tensors = frozen_func.outputs

    # Run a Grappler pass.  !!!NOT!!!!.   C
    # Constant folding removes FakeQuant Ops 
    # toco relies on to get stable min/max info on weights
    # Since toco has useable constant folding included we don want this

    if constant_folding:
        optimizers = ["constfold"]
        grappler_config = _get_grappler_config(optimizers)
        graph_def = frozen_func.graph.as_graph_def()
        opt_graph_def = _run_graph_optimizations(
            graph_def,
            input_tensors,
            output_tensors,
            config=grappler_config,
            graph=frozen_func.graph)
    else:
        opt_graph_def = frozen_func.graph.as_graph_def()
    
    # Checks dimensions in input tensor.
    for tensor in input_tensors:
      # Note that shape_list might be empty for scalar shapes.
      shape_list = tensor.shape.as_list()
      if None in shape_list[1:]:
        raise ValueError(
            "None is only supported in the 1st dimension. Tensor '{0}' has "
            "invalid shape '{1}'.".format(_get_tensor_name(tensor), shape_list))
      elif shape_list and shape_list[0] is None:
        # Set the batch size to 1 if undefined.
        shape = tensor.shape.as_list()
        shape[0] = 1
        tensor.set_shape(shape)


    return opt_graph_def, raw_graph_def, input_tensors, output_tensors
