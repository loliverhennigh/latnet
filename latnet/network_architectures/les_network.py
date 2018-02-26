
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
    group.add_argument('--nonlinearity', help='network config', type=str,
                           default='relu')
    group.add_argument('--nr_downsamples', help='network config', type=int,
                           default=2)
    group.add_argument('--nr_residual_compression', help='network config', type=int,
                           default=2)
    group.add_argument('--filter_size_compression', help='network config', type=int,
                           default=9)
  
  def collide(self, in_cstate_name, in_cboundary_f_name, out_name, weight_name):
  
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
  
    # apply boundary mask
    self.concat_tensors(in_names=[in_cstate_name, in_cboundary_f_name],
                        out_name=in_cstate_name + '_feq', axis=-1) # concat on feature
    self.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
              kernel_size=1, filter_size=256, stride=1, 
              nonlinearity=nonlinearity, weight_name="conv_0" + weight_name)
    self.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
              kernel_size=1, filter_size=256, stride=1, 
              nonlinearity=nonlinearity, weight_name="conv_1" + weight_name)
    self.conv(in_name=in_cstate_name + '_feq', out_name=out_name, 
              kernel_size=1, filter_size=self.config.filter_size_compression, stride=1, 
              weight_name="conv_2" + weight_name)
  
    # calc new f
    self.out_tensors[out_name] = tf.nn.l2_normalize(self.out_tensors[out_name], dim=-1) 
                                  # - tf.reduce_mean(self.out_tensors[out_name], axis=-1, keep_dims=True), dim=-1)
  
    """
    # apply pressure and velocity boundary conditions
   
    # split tensor into boundary and none boundary piece
    self.out_tensors[in_cstate_name + '_boundary'] = (self.out_tensors[in_cstate_name]
                                                    * self.out_tensors[in_cboundary_mask_name])
  
    # apply boundary mask
    self.concat_tensors(in_names=[in_cstate_name + '_boundary', in_cboundary_f_name],
                        out_name=in_cstate_name + '_boundary', axis=-1) # concat on feature
    self.conv(in_name=in_cstate_name + "_boundary", out_name=in_cstate_name + "_boundary", 
              kernel_size=1, filter_size=128, stride=1, 
              nonlinearity=nonlinearity, weight_name="bconv_0")
    self.conv(in_name=in_cstate_name + "_boundary", out_name=in_cstate_name + "_boundary", 
              kernel_size=1, filter_size=64, stride=1, 
              nonlinearity=nonlinearity, weight_name="bconv_1")
    self.conv(in_name=in_cstate_name + "_boundary", out_name=in_cstate_name + "_boundary", 
              kernel_size=1, filter_size=self.config.filter_size_compression, stride=1, 
              weight_name="bconv_2")
  
    # calc Feq
    self.conv(in_name=in_cstate_name, out_name=in_cstate_name + '_feq', 
              kernel_size=1, filter_size=128, stride=1, 
              nonlinearity=nonlinearity, weight_name="conv_0")
    self.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
              kernel_size=1, filter_size=64, stride=1, 
              nonlinearity=nonlinearity, weight_name="conv_1")
    self.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
              kernel_size=1, filter_size=self.config.filter_size_compression, stride=1, 
              weight_name="conv_2")
  
    # calc new f
    self.out_tensors['tau'] = tf.get_variable('tau', [1], initializer=tf.constant_initializer(0.1))
    self.out_tensors[in_cstate_name + '_f'] = (self.out_tensors[in_cstate_name]
             - (tf.abs(self.out_tensors['tau'])*(self.out_tensors[in_cstate_name] - self.out_tensors[in_cstate_name + '_feq'])))
    self.out_tensors[in_cstate_name + '_f'] = (self.out_tensors[in_cstate_name + '_f']
                                      * (1.0 - self.out_tensors[in_cboundary_mask_name]))
    self.out_tensors[out_name] = (self.out_tensors[in_cstate_name + '_f']
                                + self.out_tensors[in_cstate_name + '_boundary'])
  
    # apply pressure and velocity boundary conditions
    #self.out_tensors[out_name] = (self.out_tensors[in_cstate_name + '_f']
    """
  
  def stream(self, in_cstate_name, in_cboundary_stream_name, out_name):
    STREAM = np.zeros((3,3,self.config.filter_size_compression,self.config.filter_size_compression))
    for i in xrange(self.config.filter_size_compression/9):
      layer = i * 9
      STREAM[1,1,0+layer,0+layer] = 1.0
      STREAM[1,0,1+layer,1+layer] = 1.0
      STREAM[0,1,2+layer,2+layer] = 1.0
      STREAM[1,2,3+layer,3+layer] = 1.0
      STREAM[2,1,4+layer,4+layer] = 1.0
      STREAM[0,0,5+layer,5+layer] = 1.0
      STREAM[0,2,6+layer,6+layer] = 1.0
      STREAM[2,2,7+layer,7+layer] = 1.0
      STREAM[2,0,8+layer,8+layer] = 1.0
    STREAM = tf.constant(STREAM, dtype=1)
  
    # stream f
    self.simple_conv(in_name=in_cstate_name, out_name=out_name, kernel=STREAM)
  
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

    self.conv(in_name=out_name, out_name=out_name,
              kernel_size=1, stride=1,
              filter_size=9,
              weight_name="conv_0")
  

  # compression mapping
  def compression_mapping(self, in_cstate_name, in_cboundary_name, out_name):
   
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)

    # lattice steps
    for i in xrange(1):
      # residual block
      self.res_block(in_name=in_cstate_name, out_name=out_name, 
                     filter_size=32,
                     nonlinearity=nonlinearity, 
                     kernel_size=3,
                     weight_name="res_" + str(i))
      in_cstate_name = out_name
 
      # trim cboundary
      self.trim_tensor(in_name=in_cboundary_name, 
                      out_name=in_cboundary_name, 
                      trim=2)
  
      # just concat tensors
      self.concat_tensors(in_names=[in_cstate_name, in_cboundary_name], 
                         out_name=out_name, axis=-1)
   
    self.res_block(in_name=in_cstate_name, out_name=out_name, 
                   filter_size=9,
                   nonlinearity=nonlinearity, 
                   kernel_size=3,
                   weight_name="res_last")
 
    # trim cboundary
    self.trim_tensor(in_name=in_cboundary_name, 
                    out_name=in_cboundary_name, 
                    trim=2)
  

  # decoder state
  def decoder_state(self, in_name, out_name, lattice_size=9):
 
    for i in xrange(self.config.nr_downsamples): 
      self.upsample(in_name=in_name, out_name=out_name)
      in_name = out_name
    
  
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
  
