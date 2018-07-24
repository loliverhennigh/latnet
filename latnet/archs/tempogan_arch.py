
import tensorflow as tf
import numpy as np

import sys
sys.path.append('../')

from network_architecture import NetArch
from nn import *

class TempoGAN(NetArch):
  # This network was inspired by the paper "tempoGAN: A Temporally Coherent, 
  # Volumetric GAN for Super-resolution Fluid Flow"

  # network name for saving
  network_name = "tempo_gan"

  # network configs
  @classmethod
  def add_options(cls, group):
    group.add_argument('--nonlinearity', help='network config', type=str,
                           default='relu')
    group.add_argument('--gated', help='network config', type=bool,
                           default=False)
    group.add_argument('--nr_downsamples', help='network config', type=int,
                           default=2)
    group.add_argument('--filter_size_compression', help='network config', type=int,
                           default=16)
  
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
                   filter_size=128,
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
   
    # just do the same as encoder state
    self.encoder_state(in_name, out_name)
  
  # compression mapping
  def _compression_mapping(self, in_cstate_name, in_cboundary_name, out_name):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
  
    # just concat tensors
    self.concat_tensors(in_names=[in_cstate_name, in_cboundary_name], 
                       out_name=out_name, axis=-1)
  
    # residual block
    self.fast_res_block(in_name=out_name, out_name=out_name, 
                   filter_size=self.config.filter_size_compression, 
                   filter_size_conv=self.config.filter_size_compression/4, 
                   nonlinearity=nonlinearity, 
                   kernel_size=7,
                   weight_name="res_" + str(0))
  
    # trim cboundary
    self.trim_tensor(in_name=in_cboundary_name, 
                    out_name=in_cboundary_name, 
                    trim=6)
  
   # just concat tensors
    self.concat_tensors(in_names=[out_name, in_cboundary_name], 
                       out_name=out_name, axis=-1)
  
    # residual block
    self.fast_res_block(in_name=out_name, out_name=out_name, 
                   filter_size=self.config.filter_size_compression, 
                   filter_size_conv=self.config.filter_size_compression/4, 
                   nonlinearity=nonlinearity, 
                   kernel_size=7,
                   weight_name="res_" + str(1))
  
    # trim cboundary
    self.trim_tensor(in_name=in_cboundary_name, 
                    out_name=in_cboundary_name, 
                    trim=6)
  
    # just concat tensors
    self.concat_tensors(in_names=[out_name, in_cboundary_name], 
                       out_name=out_name, axis=-1)
  
    # residual block
    self.fast_res_block(in_name=out_name, out_name=out_name, 
                   filter_size=self.config.filter_size_compression, 
                   filter_size_conv=self.config.filter_size_compression/2, 
                   nonlinearity=nonlinearity, 
                   kernel_size=5,
                   weight_name="res_" + str(2))
  
    # trim cboundary
    self.trim_tensor(in_name=in_cboundary_name, 
                    out_name=in_cboundary_name, 
                    trim=4)
  
  # decoder state
  def _decoder_state(self, in_cstate_name, in_cboundary_name, out_name, lattice_size=9):
  
    # set nonlinearity
    nonlinearity = set_nonlinearity(self.config.nonlinearity)
   
    # just concat tensors
    self.concat_tensors(in_names=[in_cstate_name, in_cboundary_name], 
                       out_name=out_name, axis=-1)

    # image resize network
    self.upsample(in_name=out_name, out_name=out_name)
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
  
    #self.concat_tensors(in_names=[in_boundary_name, out_name],
    #                    out_name=out_name, axis=-1) # concat on feature
  
    #self.conv(in_name=out_name, out_name=out_name,
    #          kernel_size=3, stride=1,
    #          nonlinearity=nonlinearity,
    #          filter_size=64,
    #          weight_name="boundary_last_conv")
  
    self.conv(in_name=out_name, out_name=out_name,
              kernel_size=1, stride=1,
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
  
    #self.out_tensors[out_class] = tf.reduce_mean(self.out_tensors[out_class], axis=[1,2], keep_dims=True)
  
    self.conv(in_name=out_class, out_name=out_class,
            kernel_size=1, stride=1,
            filter_size=1,
            weight_name="fc_0")
  
    self.nonlinearity(name=out_class, nonlinearity_name='sigmoid')
  
