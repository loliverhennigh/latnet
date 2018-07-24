
import tensorflow as tf
import numpy as np

import sys
sys.path.append('../')

from network_architecture import NetArch
from nn import *

class GeneralizedLB(NetArch):
  # This network is ment to replicate the standard lattice boltzmann method but
  # with trainable parameters (work in progress)

  # network name for saving
  network_name = "generalized_lb"
  
  # network self.config
  @classmethod
  def add_options(cls, group):
    group.add_argument('--nonlinearity', help='network config', type=str,
                           default='relu')
    group.add_argument('--gated', help='network config', type=bool,
                           default=False)
    group.add_argument('--nr_downsamples', help='network config', type=int,
                           default=2)
    group.add_argument('--filter_size_compression', help='network config', type=int,
                           default=2)
  
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
  
    #self.out_tensors[in_cstate_name + "_trim"] = self.out_tensors[in_cstate_name]
  
    # stream f
    self.simple_conv(in_name=in_cstate_name, out_name=out_name, kernel=STREAM)
  
    # dont let stream on boundarys
    #self.trim_tensor(in_name=in_cstate_name + "_trim", 
    #                 out_name=in_cstate_name + "_trim", 
    #                 trim=1)
  
    # prevent streaming
    #self.out_tensors[out_name] = (self.out_tensors[out_name]
    #                     * self.out_tensors[in_cboundary_stream_name])
    #self.out_tensors[out_name] += (self.out_tensors[in_cstate_name + "_trim"]
    #                     * (1.0 - self.out_tensors[in_cboundary_stream_name]))
  
  
  # encoder state
  def _encoder_state(self, in_name, out_name):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
  
    # encoding peice
    self.res_block(in_name=in_name, out_name=out_name,
                   filter_size=32,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=self.config.gated, 
                   begin_nonlinearity=False,
                   weight_name="down_sample_res_" + str(0))
    self.res_block(in_name=out_name, out_name=out_name,
                   filter_size=64,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=self.config.gated, 
                   weight_name="down_sample_res_" + str(1))
   
    self.conv(in_name=out_name, out_name=out_name,
              filter_size=self.config.filter_size_compression,
              kernel_size=1, stride=1,
              weight_name="final_conv")
  
  # encoder boundary
  def _encoder_boundary(self, in_name, out_name):
   
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
  
    # encoding peice
    self.res_block(in_name=in_name, out_name=out_name,
                   filter_size=32,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=self.config.gated, 
                   begin_nonlinearity=False,
                   weight_name="down_sample_res_" + str(0))
    self.res_block(in_name=out_name, out_name=out_name,
                   filter_size=64,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=self.config.gated, 
                   weight_name="down_sample_res_" + str(1))
    self.res_block(in_name=out_name, out_name=out_name,
                   filter_size=128,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=self.config.gated, 
                   begin_nonlinearity=False,
                   weight_name="down_sample_res_" + str(2))
    self.res_block(in_name=out_name, out_name=out_name,
                   filter_size=128,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=self.config.gated, 
                   weight_name="down_sample_res_" + str(3))
   
    self.conv(in_name=out_name, out_name=out_name,
              filter_size=2*self.config.filter_size_compression,
              kernel_size=1, stride=1,
              weight_name="final_conv")
  
  # compression mapping
  def _compression_mapping(self, in_cstate_name, in_cboundary_name, out_name):
  
    # lattice steps
    for i in xrange(4):
      self.trim_tensor(in_name=in_cboundary_name, 
                      out_name=in_cboundary_name, 
                      trim=1)
      # split tensor
      self.split_tensor(in_name=in_cboundary_name, 
                        out_names=[in_cboundary_name + "_f", 
                                   in_cboundary_name + "_stream"],
                        num_split=2, axis=3)
  
      # normalize cboundary_mask between 0 and 1
      self.nonlinearity(name=in_cboundary_name + '_stream', 
                        nonlinearity_name="sigmoid")
  
      # stream and collide
      self.stream(in_cstate_name, in_cboundary_name + '_stream', out_name)
      in_cstate_name = out_name
      self.collide(in_cstate_name, in_cboundary_name + '_f', out_name, weight_name=str(i))
  
  # decoder state
  def _decoder_state(self, in_cstate_name, in_boundary_name, out_name, lattice_size=9):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
  
    # image resize network
    self.upsample(in_name=in_cstate_name, out_name=out_name)
    self.conv(in_name=out_name, out_name=out_name,
            kernel_size=3, stride=1,
            filter_size=8,
            weight_name="conv_" + str(0))
    self.upsample(in_name=out_name, out_name=out_name)
  
    self.res_block(in_name=out_name, out_name=out_name, 
                   filter_size=32,
                   kernel_size=5,
                   nonlinearity=nonlinearity,
                   stride=1,
                   begin_nonlinearity=False, 
                   gated=self.config.gated,
                   weight_name="res_" + str(0))
  
    self.res_block(in_name=out_name, out_name=out_name, 
                   filter_size=128,
                   kernel_size=5,
                   nonlinearity=nonlinearity,
                   stride=1,
                   gated=self.config.gated,
                   weight_name="res_" + str(1))
  
    self.res_block(in_name=out_name, out_name=out_name, 
                   filter_size=32,
                   kernel_size=5,
                   nonlinearity=nonlinearity,
                   stride=1,
                   gated=self.config.gated,
                   weight_name="res_" + str(2))
  
    self.res_block(in_name=out_name, out_name=out_name, 
                   filter_size=32,
                   kernel_size=5,
                   nonlinearity=nonlinearity,
                   stride=1,
                   gated=self.config.gated,
                   weight_name="res_" + str(3))
  
    self.concat_tensors(in_names=[in_boundary_name, out_name],
                        out_name=out_name, axis=-1) # concat on feature
  
    self.conv(in_name=out_name, out_name=out_name,
              kernel_size=3, stride=1,
              nonlinearity=nonlinearity,
              filter_size=64,
              weight_name="boundary_last_conv")
  
    self.conv(in_name=out_name, out_name=out_name,
              kernel_size=3, stride=1,
              filter_size=lattice_size,
              weight_name="last_conv")
  
  # discriminator
  def _discriminator_conditional(self, in_boundary_name, in_state_name, in_seq_state_names, out_name):
  
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
  
  def _discriminator_unconditional(self, in_seq_state_names, out_layer, out_class):
  
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
  
