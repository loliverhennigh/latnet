
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
def encoder_state(pipe):

  # encoding peice
  filter_size = CONFIGS['filter_size']
  for i in xrange(CONFIGS['nr_downsamples']):
    if i == 0:
      begin_nonlinearity = False
    else:
      begin_nonlinearity = True
    filter_size = filter_size*2
    pipe.res_block(in_name="state", out_name="state",
                   filter_size=filter_size,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=CONFIGS['gated'], 
                   begin_nonlinearity=begin_nonlinearity, 
                   weight_name="res_" + str(i))

  pipe.conv_layer(in_name="state", out_name="cstate",
                  kernel_size=1, stride=1, 
                  filter_size=CONFIGS['filter_size_compression'], 
                  weight_name="final_down_conv")
  return pipe

# encoder boundary
def encoder_boundary(pipe):

  # encoding peice
  filter_size = CONFIGS['filter_size']
  for i in xrange(CONFIGS['nr_downsamples']):
    if i == 0:
      begin_nonlinearity = False
    else:
      begin_nonlinearity = True
    filter_size = filter_size*2
    pipe.res_block(in_name="boundary", out_name="boundary",
                   filter_size=filter_size,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=CONFIGS['gated'], 
                   begin_nonlinearity=begin_nonlinearity, 
                   weight_name="res_" + str(i))

  pipe.conv_layer(in_name="boundary", out_name="cboundary",
                  kernel_size=1, stride=1, 
                  filter_size=CONFIGS['filter_size_compression'], 
                  weight_name="final_down_conv")
  return pipe

# compression mapping boundary
def compression_mapping_boundary(pipe):
  # split tensor
  pipe.split_tensor("cboundary", "cboundary_add", "cboundary_mask", 2, -1)

  # normalize cboundary_mask between 0 and 1
  pipe.nonlinearity("cboundary_mask", "sigmoid")

  # apply image mask
  pipe.image_combinde("cstate", "cboundary_add", "cboundary_mask", "cstate")

# compression mapping
def compression_mapping(pipe):
  for i in xrange(CONFIGS['nr_residual_compression']):
    if i == 0:
      begin_nonlinearity = False
    else:
      begin_nonlinearity = True
    pipe.res_block(in_name="cstate", out_name="cstate", 
                   filter_size=CONFIGS['filter_size_compression'], 
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=CONFIGS['gated'], 
                   begin_nonlinearity=begin_nonlinearity, 
                   weight_name="res_" + str(i))

# decoder state
def decoder_state(pipe, lattice_size=9):

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




