
import tensorflow as tf
import numpy as np

import sys
sys.path.append('../')

from latnetwork import LatNet
from nn import *

class LESNet(LatNet):
  # network name for saving
  network_name = "les"
  
  # network self.config
  @classmethod
  def add_options(cls, group):
    group.add_argument('--nr_residual_compression', help='network config', type=int,
                           default=1)
    group.add_argument('--nr_downsamples', help='network config', type=int,
                           default=2)
    group.add_argument('--nonlinearity', help='network config', type=str,
                           default='relu')
    group.add_argument('--filter_size', help='network config', type=int,
                           default=32)
    group.add_argument('--filter_size_compression', help='network config', type=int,
                           default=4)
  
  # encoder state
  def encoder_state(self, in_name, out_name):
  
    # encoding peice
    for i in xrange(self.config.nr_downsamples):
      self.downsample(in_name=in_name, out_name=out_name, sampling="avg")
      in_name = out_name

  # encoder boundary
  def encoder_boundary(self, in_name, out_name):
   
    # encoding peice
    for i in xrange(self.config.nr_downsamples):
      self.downsample(in_name=in_name, out_name=out_name, sampling="max")
      in_name = out_name
  
  # compression mapping
  def compression_mapping(self, in_cstate_name, in_cboundary_name, out_name, start_apply_boundary=False):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
  
    self.rename_tensor(old_name=in_cstate_name,
                       new_name=out_name)

    # 1x1 res block
    self.res_block(in_name=out_name, out_name=out_name, 
                   kernel_size=1,
                   filter_size=self.config.filter_size, 
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=False, 
                   weight_name="first_res")

    # apply residual blocks
    for i in xrange(self.config.nr_residual_compression):
      self.res_block(in_name=out_name, out_name=out_name, 
                     filter_size=self.config.filter_size, 
                     nonlinearity=nonlinearity, 
                     stride=1, 
                     gated=False, 
                     weight_name="res_" + str(i+1))
 
    # 1x1 res block
    self.res_block(in_name=out_name, out_name=out_name, 
                   kernel_size=1,
                   filter_size=self.config.filter_size_compression, 
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=False, 
                   weight_name="final_res")

  # decoder state
  def decoder_state(self, in_cstate_name, in_cboundary_name, out_name, lattice_size=9):
 
    self.rename_tensor(old_name=in_cstate_name,
                       new_name=out_name)
    for i in xrange(self.config.nr_downsamples): 
      self.trans_conv(in_name=out_name, out_name=out_name,
                      kernel_size=4, stride=2, 
                      filter_size=lattice_size,
                      weight_name="up_conv_" + str(i))

  # discriminator
  def discriminator_conditional(self, in_boundary_name, in_state_name, in_seq_state_names, out_name):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity('leaky_relu')
  
    # concat tensors
    self.concat_tensors(in_names=[in_boundary_name]
                               + [in_state_name] 
                               + in_seq_state_names, 
                        out_name=out_name, axis=-1) # concat on feature
  
    self.conv(in_name=out_name, out_name=out_name,
              kernel_size=4, stride=2,
              filter_size=32,
              nonlinearity=nonlinearity,
              weight_name="conv_0")
  
    self.conv(in_name=out_name, out_name=out_name,
              kernel_size=4, stride=2,
              filter_size=64,
              nonlinearity=nonlinearity,
              weight_name="conv_1")
  
    self.conv(in_name=out_name, out_name=out_name,
              kernel_size=4, stride=2,
  	    filter_size=128,
  	    nonlinearity=nonlinearity,
  	    weight_name="conv_2")
  
    self.conv(in_name=out_name, out_name=out_name,
              kernel_size=4, stride=1,
              filter_size=256,
              nonlinearity=nonlinearity,
              weight_name="conv_3")
  
    #self.out_tensors[out_name] = tf.reduce_mean(self.out_tensors[out_name], axis=[1,2], keep_dims=True)
  
    self.conv(in_name=out_name, out_name=out_name,
            kernel_size=1, stride=1,
            filter_size=1,
            weight_name="fc_0")
  
    self.nonlinearity(name=out_name, nonlinearity_name='sigmoid')
  
  def discriminator_unconditional(self, in_seq_state_names, out_layer, out_class):
  
    # set nonlinearity
    #nonlinearity = set_nonlinearity('leaky_relu')
    nonlinearity = set_nonlinearity('relu')
  
    self.concat_tensors(in_names=in_seq_state_names, out_name=out_class, axis=0) # concat on batch
  
    self.conv(in_name=out_class, out_name=out_class,
              kernel_size=4, stride=2,
              filter_size=32,
              nonlinearity=nonlinearity,
              weight_name="conv_0")
  
    self.rename_tensor(out_class, out_layer)
  
    self.conv(in_name=out_class, out_name=out_class,
              kernel_size=4, stride=2,
              filter_size=64,
              nonlinearity=nonlinearity,
              weight_name="conv_1")
  
    self.conv(in_name=out_class, out_name=out_class,
              kernel_size=4, stride=2,
  	    filter_size=128,
  	    nonlinearity=nonlinearity,
  	    weight_name="conv_2")
  
    self.conv(in_name=out_class, out_name=out_class,
              kernel_size=4, stride=1,
              filter_size=256,
              nonlinearity=nonlinearity,
              weight_name="conv_3")
  
    self.conv(in_name=out_class, out_name=out_class,
            kernel_size=1, stride=1,
            filter_size=1,
            weight_name="fc_0")
  
    self.nonlinearity(name=out_class, nonlinearity_name='sigmoid')
  
