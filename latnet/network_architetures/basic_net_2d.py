
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

# number of residual blocks before down sizing
CONFIGS['nr_residual'] = 2

# numper of downsamples
CONFIGS['nr_downsamples'] = 4

# what nonlinearity to use, leakey_relu, relu, elu, concat_elu
CONFIGS['nonlinearity']="relu",
nonlinearity = set_nonlinearity(CONFIGS['nonlinearity'])

# gated res blocks
CONFIGS['gated']=True,

# filter size for first res block. the rest of the filters are 2x every downsample
CONFIGS['filter_size']=16

# final filter size
CONFIGS['filter_size_compression']=128

def encoding_state(x_i, padding, name=''):
  # encoding peice
  filter_size = CONFIGS['filter_size']
  for i in xrange(CONFIGS['nr_downsamples']):
    for j in xrange(FLAGS.nr_residual - 1):
      filter_size = filter_size*2
      stride = 1
      if i == 0:
        stride = 2
      x_i = res_block(x_i, 
                      filter_size=filter_size, 
                      nonlinearity=nonlinearity, 
                      stride=stride, 
                      gated=CONFIGS['gated'], 
                      padding=padding, 
                      name=name + "res_downsample_" + str(i) + "_nr_residual_" + str(j), 
                      begin_nonlinearity=False) 

  x_i = res_block(x_i, 
                 filter_size=CONFIGS['filter_size_compression'], 
                 nonlinearity=nonlinearity, 
                 stride=1, 
                 gated=FLAGS.gated, 
                 padding=padding, 
                 name=name + "resnet_last_before_compression")
  return x_i

def encoding_boundary(x_i, padding, name=''):
  # encoding peice
  filter_size = CONFIGS['filter_size']/2
  for i in xrange(CONFIGS['nr_downsamples']):
    filter_size = filter_size*2
    x_i = res_block(x_i, 
                    filter_size=filter_size, 
                    nonlinearity=nonlinearity, 
                    stride=2, 
                    gated=CONFIGS['gated'], 
                    padding=padding, 
                    name=name + "resnet_down_sampled_" + str(i) + "_nr_residual_0", 
                    begin_nonlinearity=False) 

  x_i = res_block(x_i, 
                 filter_size=CONFIGS['filter_size_compression'], 
                 nonlinearity=nonlinearity, 
                 stride=1, 
                 gated=FLAGS.gated, 
                 padding=padding, 
                 name=name + "resnet_last_before_compression")
  return x_i

def compression_mapping_boundary(y_i, compressed_boundary):
    [compressed_boundary_mul, compressed_boundary_add] = tf.split(compressed_boundary, 
                                                        2, len(compressed_boundary.get_shape())-1)
  

def compression_mapping(y_i, padding, name=''):
  for i in xrange(CONFIGS['nr_residual_compression']:
    y_i = res_block(y_i, 
                    filter_size=CONFIGS['filter_size_compression'], 
                    nonlinearity=nonlinearity, 
                    stride=1, 
                    gated=FLAGS.gated, 
                    padding=padding, 
                    name="res_" + str(i))

  return y_i

def decoding_state(y_i, padding, lattice_size=9, extract_type=None, extract_pos=64):

  if (extract_type is not None) and (extract_type != 'False'):
    width = (CONFIGS['nr_downsamples']-1)*CONFIGS['nr_residual']*2
    ### hard setting extract_pos for now ###
    extract_pos = width + 1
    ########################################
    x_i = trim_tensor(x_i, extract_pos, width, extract_type)

  for i in xrange(FLAGS.nr_downsamples-1):
    filter_size = FLAGS.filter_size*pow(2,FLAGS.nr_downsamples-i-2)
    x_i = transpose_conv_layer(x_i, 4, 2, filter_size, padding, "up_conv_" + str(i))
    for j in xrange(FLAGS.nr_residual):
      x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated, padding=padding, name="resnet_up_sampled_" + str(i) + "_nr_residual_" + str(j+1))
      if (extract_type is not None) and (extract_type != 'False'):
        width = width-2
        x_i = trim_tensor(x_i, width+2, width, extract_type)

  x_i = transpose_conv_layer(x_i, 4, 2, lattice_size, padding, "up_conv_" + str(FLAGS.nr_downsamples))

  return tf.nn.tanh(x_i)




