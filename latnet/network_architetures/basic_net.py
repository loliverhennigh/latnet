
import tensorflow as tf
import numpy as np
from nn import *

# define network configs
CONFIGS = {}

# number of residual blocks before down sizing
CONFIGS['nr_residual'] = 2

# numper of downsamples
CONFIGS['nr_downsamples']=4

# what nonlinearity to use, leakey_relu, relu, elu, concat_elu
CONFIGS['nonlinearity']="relu",

# keep probability for res blocks
CONFIGS['keep_p']=1.0,

# gated res blocks
CONFIGS['gated']=False,

# filter size for first res block. the rest of the filters are 2x every downsample
CONFIGS['filter_size']=16

def encoding(inputs, name=''):
  x_i = inputs
  nonlinearity = set_nonlinearity(FLAGS.nonlinearity)
  if FLAGS.system == "fluid_flow":
    padding = (len(x_i.get_shape())-3)*["mobius"] + ["zeros"]
  elif FLAGS.system == "em":
    padding = ["mobius", "mobius"]

  for i in xrange(FLAGS.nr_downsamples):

    filter_size = FLAGS.filter_size*(pow(2,i))
    x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=2, gated=FLAGS.gated, padding=padding, name=name + "resnet_down_sampled_" + str(i) + "_nr_residual_0", begin_nonlinearity=False) 


    for j in xrange(FLAGS.nr_residual - 1):
      x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated, padding=padding, name=name + "resnet_down_sampled_" + str(i) + "_nr_residual_" + str(j+1))

  x_i = res_block(x_i, filter_size=FLAGS.filter_size_compression, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated, padding=padding, name=name + "resnet_last_before_compression")
  return x_i





