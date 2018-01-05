
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
CONFIGS['nr_downsamples'] = 3

# what nonlinearity to use, leakey_relu, relu, elu, concat_elu
CONFIGS['nonlinearity']="relu"
nonlinearity = set_nonlinearity(CONFIGS['nonlinearity'])

# gated res blocks
CONFIGS['gated']=True

# filter size for first res block. the rest of the filters are 2x every downsample
CONFIGS['filter_size']=32

# final filter size
CONFIGS['filter_size_compression']=128

# decoder state padding
nr_residual_tail = 2
PADDING['encoder_state_padding'] = int(pow(2,CONFIGS['nr_downsamples']) * 2 * nr_residual_tail)

# encoder state
def encoder_state(x_i, name='state_'):
  # encoding peice
  filter_size = CONFIGS['filter_size']
  for i in xrange(CONFIGS['nr_downsamples']):
    filter_size = filter_size*2
    x_i = conv_layer(x_i, 2, 2, 2*filter_size, "conv2x2_" + str(i), nonlinearity=nonlinearity)
    x_i = conv_layer(x_i, 1, 1, filter_size, "conv1x1_" + str(i), nonlinearity=nonlinearity)

  for i in xrange(nr_residual_tail):
    x_i = res_block(x_i, 
                    filter_size=CONFIGS['filter_size_compression'], 
                    nonlinearity=nonlinearity, 
                    stride=1, 
                    gated=CONFIGS['gated'], 
                    name="input_res_" + str(i))
  return x_i

# encoder boundary padding
PADDING['encoder_boundary_padding'] =  int(pow(2,CONFIGS['nr_downsamples']) * 2 * nr_residual_tail)

# encoder boundary
def encoder_boundary(x_i, name='boundary_'):
  # encoding peice
  filter_size = CONFIGS['filter_size']
  for i in xrange(CONFIGS['nr_downsamples']):
    filter_size = filter_size*2
    x_i = conv_layer(x_i, 2, 2, filter_size, "conv2x2_" + str(i), nonlinearity=nonlinearity)
    x_i = conv_layer(x_i, 1, 1, filter_size, "conv1x1_" + str(i), nonlinearity=nonlinearity)


  for i in xrange(nr_residual_tail):
    x_i = res_block(x_i, 
                    filter_size=2*CONFIGS['filter_size_compression'], 
                    nonlinearity=nonlinearity, 
                    stride=1, 
                    gated=CONFIGS['gated'], 
                    name="input_res_" + str(i))
  return x_i

# compression mapping boundary padding
PADDING['compression_mapping_boundary_padding'] = 0

# compression mapping boundary
def compression_mapping_boundary(y_i, compressed_boundary):
  off_set = int(compressed_boundary.get_shape()[1] - y_i.get_shape()[1])/2
  if off_set != 0:
    compressed_boundary = compressed_boundary[:,off_set:-off_set,off_set:-off_set]
  [compressed_boundary_mul, compressed_boundary_add] = tf.split(compressed_boundary, 
                                                      2, len(compressed_boundary.get_shape())-1)
  y_i = (y_i * compressed_boundary_mul) + compressed_boundary_add
  return y_i

# compression mapping padding
PADDING['compression_mapping_padding'] = int(pow(2,CONFIGS['nr_downsamples']) * 2 * CONFIGS['nr_residual_compression'])

# compression mapping
def compression_mapping(y_i, name=''):
  for i in xrange(CONFIGS['nr_residual_compression']):
    y_i = res_block(y_i, 
                    filter_size=CONFIGS['filter_size_compression'], 
                    nonlinearity=nonlinearity, 
                    stride=1, 
                    gated=CONFIGS['gated'], 
                    name="res_" + str(i))

  return y_i

# decoder state padding
PADDING['decoder_state_padding'] =  int(pow(2,CONFIGS['nr_downsamples']) * 2 * nr_residual_tail)

# decoder state
def decoder_state(y_i, lattice_size=9):
  for i in xrange(nr_residual_tail):
    y_i = res_block(y_i, 
                    filter_size=CONFIGS['filter_size_compression'], 
                    nonlinearity=nonlinearity, 
                    stride=1, 
                    gated=CONFIGS['gated'], 
                    name="input_res_" + str(i))

  for i in xrange(CONFIGS['nr_downsamples']):
    filter_size = int(CONFIGS['filter_size']*pow(2,CONFIGS['nr_downsamples']-i-1))
    y_i = conv_layer(y_i, 1, 1, filter_size*2, "conv1x1_" + str(i), nonlinearity=nonlinearity)
    y_i = transpose_conv_layer(y_i, 2, 2, filter_size, "transconv2x2_" + str(i), nonlinearity=nonlinearity)

  y_i = conv_layer(y_i, 1, 1, lattice_size, "last_conv")
  return tf.nn.tanh(y_i)




