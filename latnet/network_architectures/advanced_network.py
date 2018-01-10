
import tensorflow as tf
import numpy as np
from nn import *

# there are like 6 peices needed for the full network
# encoder network for state
# encoder network for boundary
# compression mapping
# decoder state
# decoder force and other

# define network configs
CONFIGS = {}

# define network configs
PADDING = {}

# number of residual blocks in compression mapping
CONFIGS['nr_residual_compression'] = 2

# numper of downsamples
CONFIGS['nr_downsamples'] = 4

# what nonlinearity to use, leakey_relu, relu, elu, concat_elu
CONFIGS['nonlinearity']="relu"
nonlinearity = set_nonlinearity(CONFIGS['nonlinearity'])

# gated res blocks
CONFIGS['gated']=False

# filter size for first res block. the rest of the filters are 2x every downsample
CONFIGS['filter_size']=12

# final filter size
CONFIGS['filter_size_compression']=128

# encoder state
def encoder_state(tensor, shape_converter=None, name='state_'):

  # TODO this code can be cleaned
  x_i = tensor.tf_tensor
  shape_converter = tensor.shape_converter

  # encoding peice
  filter_size = CONFIGS['filter_size']
  for i in xrange(CONFIGS['nr_downsamples']):
    filter_size = filter_size*2
    x_i = res_block(x_i, 
                    filter_size=filter_size,
                    nonlinearity=nonlinearity, 
                    stride=2, 
                    gated=CONFIGS['gated'], 
                    name="res_" + str(i))
    shape_converter.add_res_block(stride=2)

  x_i = conv_layer(x_i, 1, 1, CONFIGS['filter_size_compression'], "final_down_conv")
  shape_converter.add_conv(kernel_size=1, stride=1)
  tensor.tf_tensor = x_i 
  tensor.shape_converter = shape_converter
  return tensor

# encoder boundary
def encoder_boundary(x_i, shape_converter, name='boundary_'):
  # encoding peice
  filter_size = CONFIGS['filter_size']
  for i in xrange(CONFIGS['nr_downsamples']):
    filter_size = filter_size*2
    x_i = res_block(x_i, 
                    filter_size=filter_size,
                    nonlinearity=nonlinearity, 
                    stride=2, 
                    gated=CONFIGS['gated'], 
                    name="res_" + str(i))
  if shape_converter is not None:
    shape_converter.add_res_block(stride=2)

  x_i = conv_layer(x_i, 1, 1, 2*CONFIGS['filter_size_compression'], "final_down_conv")
  if shape_converter is not None:
  shape_converter.add_conv(kernel_size=1, stride=1)
  tensor.tf_tensor = x_i 
  tensor.shape_converter = shape_converter
  return x_i, shape_converter

# compression mapping boundary
def compression_mapping_boundary(y_i, compressed_boundary, shape_converter):
  off_set = int(compressed_boundary.get_shape()[1] - y_i.get_shape()[1])/2
  if off_set != 0:
    compressed_boundary = compressed_boundary[:,off_set:-off_set,off_set:-off_set]
  [compressed_boundary_mul, compressed_boundary_add] = tf.split(compressed_boundary, 
                                                      2, len(compressed_boundary.get_shape())-1)
  y_i = (y_i * compressed_boundary_mul) + compressed_boundary_add
  tensor.tf_tensor = x_i 
  tensor.shape_converter = shape_converter
  return y_i, shape_converter

# compression mapping
def compression_mapping(y_i, shape_converter, name=''):
  for i in xrange(CONFIGS['nr_residual_compression']):
    y_i = res_block(y_i, 
                    filter_size=CONFIGS['filter_size_compression'], 
                    nonlinearity=nonlinearity, 
                    stride=1, 
                    gated=CONFIGS['gated'], 
                    name="res_" + str(i))
    shape_converter.add_res_block(stride=1)

  tensor.tf_tensor = x_i 
  tensor.shape_converter = shape_converter
  return y_i, shape_converter

# decoder state
def decoder_state(y_i, shape_converter, lattice_size=9, extract_type=None, extract_pos=64):

  for i in xrange(CONFIGS['nr_downsamples']):
    filter_size = int(CONFIGS['filter_size']*pow(2,CONFIGS['nr_downsamples']-i-1))
    y_i = transpose_conv_layer(y_i, 2, 2, filter_size, "up_conv_" + str(i), nonlinearity=nonlinearity)
    shape_converter.add_trans_conv(kernel_size=2, stride=2)
    y_i = res_block(y_i, 
                    filter_size=filter_size,
                    nonlinearity=nonlinearity,
                    stride=1,
                    gated=CONFIGS['gated'],
                    name="res_" + str(i))
    shape_converter.add_res_block(stride=1)

  y_i = conv_layer(y_i, 3, 1, lattice_size, "last_conv")
  shape_converter.add_conv(kernel_size=3, stride=1)
  tensor.tf_tensor = tf.nn.tanh(y_i)
  tensor.shape_converter = shape_converter
  return tensor




