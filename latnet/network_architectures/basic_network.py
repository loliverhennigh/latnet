
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
CONFIGS['nr_residual_compression'] = 1

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

# decoder state padding
decoder_state_padding = 2
for i in xrange(CONFIGS['nr_downsamples']):
  decoder_state_padding += 2*pow(2, i)
PADDING['encoder_state_padding'] = decoder_state_padding

# encoder state
def encoder_state(x_i, name='state_'):
  # encoding peice
  filter_size = CONFIGS['filter_size']
  for i in xrange(CONFIGS['nr_downsamples']):
    filter_size = filter_size*2
    print(x_i.get_shape())
    x_i = res_block(x_i, 
                    filter_size=filter_size,
                    nonlinearity=nonlinearity, 
                    stride=2, 
                    gated=CONFIGS['gated'], 
                    name="res_" + str(i))
    if x_i.get_shape()[1] % 2 == 1:
      x_i = x_i[:,:-1,:-1]

  print(x_i.get_shape())
  x_i = conv_layer(x_i, 1, 1, CONFIGS['filter_size_compression'], "final_down_conv")
  print(x_i.get_shape())
  return x_i

# encoder boundary padding
PADDING['encoder_boundary_padding'] = PADDING['encoder_state_padding']

# encoder boundary
def encoder_boundary(x_i, name='boundary_'):
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
    if x_i.get_shape()[1] % 2 == 1:
      x_i = x_i[:,:-1,:-1]

    #x_i = conv_layer(x_i, 3, 1, filter_size, "conv_" + str(i), nonlinearity=nonlinearity)
    #x_i = conv_layer(x_i, 2, 2, filter_size, "down_conv_" + str(i), nonlinearity=nonlinearity)

  x_i = conv_layer(x_i, 1, 1, 2*CONFIGS['filter_size_compression'], "final_down_conv")
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
decoder_state_padding = 1
for i in xrange(CONFIGS['nr_downsamples']):
  decoder_state_padding += 3*pow(2, i)
PADDING['decoder_state_padding'] = decoder_state_padding

# decoder state
def decoder_state(y_i, lattice_size=9, extract_type=None, extract_pos=64):

  for i in xrange(CONFIGS['nr_downsamples']):
    print(y_i.get_shape())
    filter_size = int(CONFIGS['filter_size']*pow(2,CONFIGS['nr_downsamples']-i-1))
    y_i = transpose_conv_layer(y_i, 4, 2, filter_size, "up_conv_" + str(i), nonlinearity=nonlinearity)
    y_i = res_block(y_i, 
                    filter_size=filter_size,
                    nonlinearity=nonlinearity,
                    stride=1,
                    gated=CONFIGS['gated'],
                    name="res_" + str(i))

  print(y_i.get_shape())
  y_i = conv_layer(y_i, 3, 1, lattice_size, "last_conv")
  print(y_i.get_shape())
  return tf.nn.tanh(y_i)




