
import tensorflow as tf
import numpy as np

import sys
sys.path.append('../')

from nn import *

# TODO make this file into class with inheritance from pipe

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
def encoder_state(pipe, in_name, out_name):

  # encoding peice
  filter_size = CONFIGS['filter_size']
  pipe.res_block(in_name=in_name, out_name=out_name,
                 filter_size=filter_size,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=CONFIGS['gated'], 
                 begin_nonlinearity=False, 
                 weight_name="res_" + str(0))
  for i in xrange(CONFIGS['nr_downsamples']-1):
    pipe.res_block(in_name=out_name, out_name=out_name,
                   filter_size=filter_size,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=CONFIGS['gated'], 
                   begin_nonlinearity=True,
                   weight_name="res_" + str(i+1))
    filter_size = filter_size*2

  pipe.conv(in_name=out_name, out_name=out_name,
                  kernel_size=1, stride=1, 
                  filter_size=CONFIGS['filter_size_compression'], 
                  weight_name="final_down_conv")

# encoder boundary
def encoder_boundary(pipe, in_name, out_name):

  # encoding peice
  filter_size = CONFIGS['filter_size']
  pipe.res_block(in_name=in_name, out_name=out_name,
                 filter_size=filter_size,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=CONFIGS['gated'], 
                 begin_nonlinearity=False, 
                 weight_name="res_" + str(0))
  for i in xrange(CONFIGS['nr_downsamples']-1):
    pipe.res_block(in_name=out_name, out_name=out_name,
                   filter_size=filter_size,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=CONFIGS['gated'], 
                   begin_nonlinearity=True,
                   weight_name="res_" + str(i+1))
    filter_size = filter_size*2

  pipe.conv(in_name=out_name, out_name=out_name,
                  kernel_size=1, stride=1, 
                  filter_size=2*CONFIGS['filter_size_compression'],
                  weight_name="final_down_conv")

# compression mapping boundary
def compression_mapping_boundary(pipe, in_cstate_name, in_cboundary_name, out_name):
  # split tensor
  pipe.split_tensor(in_name=in_cboundary_name, 
                    a_out_name=in_cboundary_name + "_add", 
                    b_out_name=in_cboundary_name + "_mask", 
                    num_split=2, axis=3)

  # normalize cboundary_mask between 0 and 1
  pipe.nonlinearity(name=in_cboundary_name + "_mask", 
                    nonlinearity_name="sigmoid")

  # apply image mask
  pipe.image_combine(a_name=in_cstate_name, 
                      b_name=in_cboundary_name + "_add", 
                      mask_name=in_cboundary_name + "_mask", 
                      out_name=out_name)

# compression mapping
def compression_mapping(pipe, in_name, out_name):
  pipe.res_block(in_name=in_name, out_name=out_name, 
                 filter_size=CONFIGS['filter_size_compression'], 
                 nonlinearity=nonlinearity, 
                 stride=1, 
                 gated=CONFIGS['gated'], 
                 begin_nonlinearity=False,
                 weight_name="res_" + str(0))
  for i in xrange(CONFIGS['nr_residual_compression'] - 1):
    pipe.res_block(in_name=out_name, out_name=out_name, 
                   filter_size=CONFIGS['filter_size_compression'], 
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=CONFIGS['gated'], 
                   begin_nonlinearity=True, 
                   weight_name="res_" + str(i+1))

# decoder state
def decoder_state(pipe, in_name, out_name, lattice_size=9):

  filter_size = int(CONFIGS['filter_size']*pow(2,CONFIGS['nr_downsamples']-1))
  pipe.trans_conv(in_name=in_name, out_name=out_name,
                  kernel_size=2, stride=2, 
                  filter_size=filter_size, 
                  weight_name="up_conv_" + str(0), 
                  nonlinearity=nonlinearity)

  for i in xrange(CONFIGS['nr_downsamples'] - 1):
    filter_size = int(CONFIGS['filter_size']*pow(2,CONFIGS['nr_downsamples']-i-1))
    pipe.res_block(in_name=out_name, out_name=out_name, 
                   filter_size=filter_size,
                   nonlinearity=nonlinearity,
                   stride=1,
                   gated=CONFIGS['gated'],
                   begin_nonlinearity=False, 
                   weight_name="res_" + str(i+1))
    pipe.trans_conv(in_name=out_name, out_name=out_name,
                    kernel_size=2, stride=2, 
                    filter_size=filter_size, 
                    weight_name="up_conv_" + str(i+1), 
                    nonlinearity=nonlinearity)

  pipe.conv(in_name=out_name, out_name=out_name,
                  kernel_size=1, stride=1, 
                  filter_size=lattice_size, 
                  weight_name="final_down_conv")

  pipe.nonlinearity(name=out_name, nonlinearity_name='tanh')


