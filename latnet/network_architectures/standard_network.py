
import tensorflow as tf
import numpy as np

import sys
sys.path.append('../')

from latnetwork import LatNet
from nn import *

class StandardNetwork(LatNet):
  # This network is from the paper "Lat-Net: Compressing Lattice Boltzmann 
  # Flow Simulations using Deep Neural Networks"

  # network name for saving
  network_name = "standard_network"

  # network self.config
  @classmethod
  def add_options(cls, group):
    group.add_argument('--nr_residual_compression', help='network config', type=int,
                           default=2)
    group.add_argument('--nr_residual_encoder', help='network config', type=int,
                           default=1)
    group.add_argument('--nr_downsamples', help='network config', type=int,
                           default=4)
    group.add_argument('--nonlinearity', help='network config', type=str,
                           default='relu')
    group.add_argument('--gated', help='network config', type=bool,
                           default=False)
    group.add_argument('--filter_size', help='network config', type=int,
                           default=16)
    group.add_argument('--upsampling', help='network config', type=str,
                           default='trans_conv') # or upsample
    group.add_argument('--filter_size_compression', help='network config', type=int,
                           default=128)
  
  # encoder state
  def encoder_state(self, in_name, out_name):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
  
    # encoding peice
    for i in xrange(self.config.nr_downsamples):
      filter_size = self.config.filter_size*(pow(2,i))
      self.res_block(in_name=in_name, out_name=out_name,
                     filter_size=filter_size,
                     nonlinearity=nonlinearity, 
                     stride=2, 
                     gated=self.config.gated, 
                     begin_nonlinearity=False,
                     weight_name="down_sample_res_" + str(i))
      in_name=out_name
  
      for j in xrange(self.config.nr_residual_encoder - 1):
        self.res_block(in_name=in_name, out_name=out_name,
                       filter_size=filter_size,
                       nonlinearity=nonlinearity, 
                       stride=1, 
                       gated=self.config.gated, 
                       weight_name="res_" + str(i) + '_' + str(j))
  
  
    self.res_block(in_name=in_name, out_name=out_name,
                   filter_size=self.config.filter_size_compression,
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=self.config.gated, 
                   weight_name="final_res")
  
  # encoder boundary
  def encoder_boundary(self, in_name, out_name):
   
    # just do the same as encoder state
    self.encoder_state(in_name, out_name)
  
  # compression mapping
  def compression_mapping(self, in_cstate_name, in_cboundary_name, out_name):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
  
    # just concat tensors
    self.concat_tensors(in_names=[in_cstate_name, in_cboundary_name], 
                       out_name=out_name, axis=-1)
  
    # apply residual blocks
    for i in xrange(self.config.nr_residual_compression):
      self.res_block(in_name=out_name, out_name=out_name, 
                     filter_size=self.config.filter_size_compression, 
                     nonlinearity=nonlinearity, 
                     stride=1, 
                     gated=self.config.gated, 
                     weight_name="res_" + str(i+1))
  
    # trim cboundary
    self.trim_tensor(in_name=in_cboundary_name, 
                    out_name=in_cboundary_name, 
                    trim=self.config.nr_residual_compression*2)
  
  # decoder state
  def decoder_state(self, in_cstate_name, in_cboundary_name, out_name, lattice_size=9):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
   
    # just concat tensors
    self.concat_tensors(in_names=[in_cstate_name, in_cboundary_name], 
                       out_name=out_name, axis=-1)

    for i in xrange(self.config.nr_downsamples-1):
      filter_size = int(self.config.filter_size*pow(2,self.config.nr_downsamples-i-2))
      self.trans_conv(in_name=out_name, out_name=out_name,
                      kernel_size=4, stride=2, 
                      filter_size=filter_size,
                      weight_name="up_conv_" + str(i))
      for j in xrange(self.config.nr_residual_encoder):
        self.res_block(in_name=out_name, out_name=out_name, 
                       filter_size=filter_size,
                       nonlinearity=nonlinearity,
                       stride=1,
                       gated=self.config.gated,
                       weight_name="res_" + str(i) + '_' + str(j))
  
    self.trans_conv(in_name=out_name, out_name=out_name,
                    kernel_size=4, stride=2,
                    filter_size=lattice_size,
                    weight_name="last_up_conv")
  
    """ 
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
    """ 
  
  # discriminator
  def discriminator_conditional(self, in_boundary_name, in_state_name, in_seq_state_names, out_name):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity('leaky_relu')
  
    self.concat_tensors(in_names=[in_boundary_name]
                               + [in_state_name] 
                               + in_seq_state_names, 
                        out_name=out_name, axis=-1) # concat on feature
    filter_size=self.config.filter_size
    for i in xrange(self.config.nr_residual_compression):
      begin_nonlinearity = True
      if i == 0:
        begin_nonlinearity = False
      self.res_block(in_name=out_name, out_name=out_name, 
                     filter_size=filter_size,
                     nonlinearity=nonlinearity,
                     stride=2,
                     gated=self.config.gated,
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
  
  def discriminator_unconditional(self, in_seq_state_names, out_layer, out_class):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity('leaky_relu')
  
    self.concat_tensors(in_names=in_seq_state_names, out_name=out_class, axis=0) # concat on batch
    filter_size=self.config.filter_size
    for i in xrange(self.config.nr_residual_compression):
      begin_nonlinearity = True
      if i == 0:
        begin_nonlinearity = False
      self.res_block(in_name=out_class, out_name=out_class, 
                     filter_size=filter_size,
                     nonlinearity=nonlinearity,
                     stride=2,
                     gated=self.config.gated,
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
  
