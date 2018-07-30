
import tensorflow as tf
import numpy as np

import sys
sys.path.append('../')

from network_architecture import NetArch
from nn import *

class StandardSailfishArch(NetArch):
  # This network is from the paper "Lat-Net: Compressing Lattice Boltzmann 
  # Flow Simulations using Deep Neural Networks"
  # network name for saving
  network_name = "standard_network"

  def __init__(self, config):
    super(StandardSailfishArch, self).__init__(config)

    self.nr_residual_compression = config.nr_residual_compression
    self.nr_residual_encoder = config.nr_residual_encoder
    self.nr_downsamples = config.nr_downsamples
    self.nonlinearity = config.nonlinearity
    self.gated = config.gated
    self.filter_size = config.filter_size
    self.upsampling = config.upsampling
    self.filter_size_compression = config.filter_size_compression
    self.cstate_depth = config.cstate_depth
    self.cboundary_depth = config.cboundary_depth

  # network self.config
  @classmethod
  def add_options(cls, group):
    group.add_argument('--nr_residual_compression', help='network config', type=int,
                           default=4)
    group.add_argument('--nr_residual_encoder', help='network config', type=int,
                           default=2)
    group.add_argument('--nr_downsamples', help='network config', type=int,
                           default=2)
    group.add_argument('--nonlinearity', help='network config', type=str,
                           default='relu')
    group.add_argument('--gated', help='network config', type=bool,
                           default=False)
    group.add_argument('--filter_size', help='network config', type=int,
                           default=16)
    group.add_argument('--upsampling', help='network config', type=str,
                           default='trans_conv') # or upsample
    group.add_argument('--filter_size_compression', help='network config', type=int,
                           default=64)
  
  # encoder state
  def _encoder_state(self, in_name, out_name):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.nonlinearity)
  
    # encoding peice
    for i in xrange(self.nr_downsamples):
      filter_size = self.filter_size*(pow(2,i))
      self.res_block(in_name=in_name, out_name=out_name,
                     filter_size=filter_size,
                     nonlinearity=nonlinearity, 
                     stride=2, 
                     gated=self.gated, 
                     begin_nonlinearity=False,
                     weight_name="down_sample_res_" + str(i))
      in_name=out_name
  
      for j in xrange(self.nr_residual_encoder - 1):
        self.res_block(in_name=in_name, out_name=out_name,
                       filter_size=filter_size,
                       nonlinearity=nonlinearity, 
                       stride=1, 
                       gated=self.gated, 
                       weight_name="res_" + str(i) + '_' + str(j))
  
  
    self.res_block(in_name=in_name, out_name=out_name,
                   filter_size=self.cstate_depth,
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=self.gated, 
                   weight_name="final_res")

    self.out_tensors[out_name] = tf.nn.l2_normalize(self.out_tensors[out_name], dim=-1) 
  
  # encoder boundary
  def _encoder_boundary(self, in_name, out_name):
   
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.nonlinearity)
  
    # encoding peice
    for i in xrange(self.nr_downsamples):
      filter_size = self.filter_size*(pow(2,i))
      self.res_block(in_name=in_name, out_name=out_name,
                     filter_size=filter_size,
                     nonlinearity=nonlinearity, 
                     stride=2, 
                     gated=self.gated, 
                     begin_nonlinearity=False,
                     weight_name="down_sample_res_" + str(i))
      in_name=out_name
  
      for j in xrange(self.nr_residual_encoder - 1):
        self.res_block(in_name=in_name, out_name=out_name,
                       filter_size=filter_size,
                       nonlinearity=nonlinearity, 
                       stride=1, 
                       gated=self.gated, 
                       weight_name="res_" + str(i) + '_' + str(j))
  
  
    self.res_block(in_name=in_name, out_name=out_name,
                   filter_size=self.cboundary_depth,
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=self.gated, 
                   weight_name="final_res")

  # compression mapping
  def _compression_mapping(self, in_cstate_name, in_cboundary_name, out_name, start_apply_boundary=False):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.nonlinearity)
  
    # apply boundary
    if start_apply_boundary:
      self.split_tensor(in_name=in_cboundary_name,
                        out_names=[in_cboundary_name + "_apply",
                                   in_cboundary_name + "_mask"],
                        num_split=2, axis=-1)
      self.image_combine(a_name=in_cstate_name, 
                         b_name=in_cboundary_name + "_apply",
                         mask_name=in_cboundary_name + "_mask",
                         out_name=out_name)
    else:
      self.rename_tensor(old_name=in_cstate_name,
                         new_name=out_name)

    # satart with a 1x1 residual block
    self.res_block(in_name=out_name, out_name=out_name, 
                   kernel_size=1,
                   filter_size=self.filter_size_compression, 
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=self.gated, 
                   weight_name="first_res")

    # apply residual blocks
    for i in xrange(self.nr_residual_compression):
      self.res_block(in_name=out_name, out_name=out_name, 
                     filter_size=self.filter_size_compression, 
                     nonlinearity=nonlinearity, 
                     stride=1, 
                     gated=self.gated, 
                     weight_name="res_" + str(i+1))
 
    # 1x1 res block
    self.res_block(in_name=out_name, out_name=out_name, 
                   kernel_size=1,
                   filter_size=self.filter_size_compression, 
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=self.gated, 
                   weight_name="final_res")

    # final conv to correct filter size 
    self.conv(in_name=out_name, out_name=out_name,
            kernel_size=1, stride=1,
            filter_size=self.cstate_depth,
            weight_name="fc_last")
  
    # trim cboundary
    self.trim_tensor(in_name=in_cboundary_name, 
                    out_name=in_cboundary_name, 
                    trim=self.nr_residual_compression*2)

    # apply boundary
    self.split_tensor(in_name=in_cboundary_name,
                      out_names=[in_cboundary_name + "_apply",
                                 in_cboundary_name + "_mask"],
                      num_split=2, axis=-1)
    self.image_combine(a_name=out_name, 
                       b_name=in_cboundary_name + "_apply",
                       mask_name=in_cboundary_name + "_mask",
                       out_name=out_name)

    self.out_tensors[out_name] = tf.nn.l2_normalize(self.out_tensors[out_name], dim=-1) 

  # decoder state
  def _decoder_state(self, in_name, out_name, lattice_size=9):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.nonlinearity)
   
    # just concat tensors
    self.rename_tensor(old_name=in_name,
                       new_name=out_name)
    

    for i in xrange(self.nr_downsamples-1):
      filter_size = int(self.filter_size*pow(2,self.nr_downsamples-i-2))
      self.trans_conv(in_name=out_name, out_name=out_name,
                      kernel_size=4, stride=2, 
                      filter_size=filter_size,
                      weight_name="up_conv_" + str(i))
      for j in xrange(self.nr_residual_encoder):
        self.res_block(in_name=out_name, out_name=out_name, 
                       filter_size=filter_size,
                       nonlinearity=nonlinearity,
                       stride=1,
                       gated=self.gated,
                       weight_name="res_" + str(i) + '_' + str(j))
  
    self.trans_conv(in_name=out_name, out_name=out_name,
                    kernel_size=4, stride=2,
                    filter_size=lattice_size,
                    weight_name="last_up_conv")
  
  # discriminator
  def _discriminator_conditional(self, in_boundary_name, in_state_name, in_seq_state_names, out_name):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity('leaky_relu')
  
    #self.concat_tensors(in_names=[in_boundary_name]
    self.concat_tensors(in_names=[in_state_name] 
                                + in_seq_state_names, 
                        out_name=out_name, axis=-1) # concat on feature
    filter_size=self.filter_size
    for i in xrange(self.nr_residual_compression):
      begin_nonlinearity = True
      if i == 0:
        begin_nonlinearity = False
      self.res_block(in_name=out_name, out_name=out_name, 
                     filter_size=filter_size,
                     nonlinearity=nonlinearity,
                     stride=2,
                     gated=self.gated,
                     begin_nonlinearity=begin_nonlinearity,
                     weight_name="res_" + str(i))
      filter_size = filter_size*2
  
    self.conv(in_name=out_name, out_name=out_name,
            kernel_size=1, stride=1,
            filter_size=256,
            nonlinearity=nonlinearity,
            weight_name="fc_1")
  
    self.conv(in_name=out_name, out_name=out_name,
            kernel_size=1, stride=1,
            filter_size=1,
            weight_name="fc_2")
  
    self.nonlinearity(name=out_name, nonlinearity_name='sigmoid')
  
  def _discriminator_unconditional(self, in_seq_state_names, out_layer, out_class):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity('leaky_relu')
  
    self.concat_tensors(in_names=in_seq_state_names, out_name=out_class, axis=0) # concat on batch
    filter_size=self.filter_size
    for i in xrange(self.nr_residual_compression):
      begin_nonlinearity = True
      if i == 0:
        begin_nonlinearity = False
      self.res_block(in_name=out_class, out_name=out_class, 
                     filter_size=filter_size,
                     nonlinearity=nonlinearity,
                     stride=2,
                     gated=self.gated,
                     begin_nonlinearity=begin_nonlinearity,
                     weight_name="res_" + str(i))
      if i == 0:
        # for layer loss as seen in tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow
        self.rename_tensor(out_class, out_layer)
      filter_size = filter_size*2
  
    self.conv(in_name=out_class, out_name=out_class,
            kernel_size=1, stride=1,
            filter_size=256,
            nonlinearity=nonlinearity,
            weight_name="fc_1")
  
    self.conv(in_name=out_class, out_name=out_class,
            kernel_size=1, stride=1,
            filter_size=1,
            weight_name="fc_2")
  
    self.nonlinearity(name=out_class, nonlinearity_name='sigmoid')
  
