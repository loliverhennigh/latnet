
import tensorflow as tf
import numpy as np
from nn import *

# there are like 6 peices needed for the full network
# encoder network for state
# encoder network for boundary
#


# define network configs
CONFIGS = {}

# number of residual blocks before down sizing
CONFIGS['nr_residual'] = 2

# numper of downsamples
CONFIGS['nr_downsamples']=4

# what nonlinearity to use, leakey_relu, relu, elu, concat_elu
CONFIGS['nonlinearity']="relu",

# gated res blocks
CONFIGS['gated']=True,

# filter size for first res block. the rest of the filters are 2x every downsample
CONFIGS['filter_size']=16

# final filter size
CONFIGS['filter_size_compression']=128

def encoding_state(x_i, padding, name=''):
  # set nonlinearity
  nonlinearity = set_nonlinearity(CONFIGS['nonlinearity'])

  # encoding peice
  filter_size = CONFIGS['filter_size']
  for i in xrange(CONFIGS['nr_downsamples']):
    for j in xrange(FLAGS.nr_residual - 1):
      filter_size = filter_size*2
      if i == 0:
        stride = 2
      else:
        stride = 1
      x_i = res_block(x_i, 
                      filter_size=filter_size, 
                      nonlinearity=nonlinearity, 
                      stride=stride, 
                      gated=CONFIGS['gated'], 
                      padding=padding, 
                      name=name + "resnet_down_sampled_" + str(i) + "_nr_residual_0", 
                      begin_nonlinearity=False) 

  x_i = res_block(x_i, 
                 filter_size=FLAGS.filter_size_compression, 
                 nonlinearity=nonlinearity, 
                 keep_p=FLAGS.keep_p, 
                 stride=1, 
                 gated=FLAGS.gated, 
                 padding=padding, 
                 name=name + "resnet_last_before_compression")
  return x_i

def encoding_boundary(x_i, padding, name=''):
  # set nonlinearity
  nonlinearity = set_nonlinearity(CONFIGS['nonlinearity'])

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
                 filter_size=FLAGS.filter_size_compression, 
                 nonlinearity=nonlinearity, 
                 keep_p=FLAGS.keep_p, 
                 stride=1, 
                 gated=FLAGS.gated, 
                 padding=padding, 
                 name=name + "resnet_last_before_compression")
  return x_i






