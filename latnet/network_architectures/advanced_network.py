
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

# network configs
def add_options(group):
  group.add_argument('--nr_residual_compression', help='network config', type=int,
                         default=3)
  group.add_argument('--nr_residual_encoder', help='network config', type=int,
                         default=1)
  group.add_argument('--nr_downsamples', help='network config', type=int,
                         default=3)
  group.add_argument('--nonlinearity', help='network config', type=str,
                         default='relu')
  group.add_argument('--gated', help='network config', type=bool,
                         default=True)
  group.add_argument('--filter_size', help='network config', type=int,
                         default=16)
  group.add_argument('--upsampling', help='network config', type=str,
                         default='trans_conv') # or upsample
  group.add_argument('--filter_size_compression', help='network config', type=int,
                         default=64)

# encoder state
def encoder_state(pipe, configs, in_name, out_name):

  # set nonlinearity
  nonlinearity = set_nonlinearity(configs.nonlinearity)

  # encoding peice
  for i in xrange(configs.nr_downsamples):
    filter_size = configs.filter_size*(pow(2,i))
    pipe.res_block(in_name=in_name, out_name=out_name,
                   filter_size=filter_size,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=configs.gated, 
                   begin_nonlinearity=False,
                   weight_name="down_sample_res_" + str(i))
    in_name=out_name

    for j in xrange(configs.nr_residual_encoder - 1):
      pipe.res_block(in_name=in_name, out_name=out_name,
                     filter_size=filter_size,
                     nonlinearity=nonlinearity, 
                     stride=1, 
                     gated=configs.gated, 
                     weight_name="res_" + str(i) + '_' + str(j))


  pipe.res_block(in_name=in_name, out_name=out_name,
                 filter_size=configs.filter_size_compression,
                 nonlinearity=nonlinearity, 
                 stride=1, 
                 gated=configs.gated, 
                 weight_name="final_res")

# encoder boundary
def encoder_boundary(pipe, configs, in_name, out_name):

  # set nonlinearity
  nonlinearity = set_nonlinearity(configs.nonlinearity)

  # encoding peice
  for i in xrange(configs.nr_downsamples):
    filter_size = configs.filter_size*(pow(2,i))
    pipe.res_block(in_name=in_name, out_name=out_name,
                   filter_size=filter_size,
                   nonlinearity=nonlinearity, 
                   stride=2, 
                   gated=configs.gated, 
                   begin_nonlinearity=False,
                   weight_name="down_sample_res_" + str(i))
    in_name=out_name

    for j in xrange(configs.nr_residual_encoder - 1):
      pipe.res_block(in_name=in_name, out_name=out_name,
                     filter_size=filter_size,
                     nonlinearity=nonlinearity, 
                     stride=1, 
                     gated=configs.gated, 
                     weight_name="res_" + str(i) + '_' + str(j))


  pipe.res_block(in_name=in_name, out_name=out_name,
                 filter_size=configs.filter_size_compression,
                 nonlinearity=nonlinearity, 
                 stride=1, 
                 gated=configs.gated, 
                 weight_name="final_res")

# compression mapping
def compression_mapping(pipe, configs, in_cstate_name, in_cboundary_name, out_name):

  # set nonlinearity
  nonlinearity = set_nonlinearity(configs.nonlinearity)

  # just concat tensors
  pipe.concat_tensors(in_names=[in_cstate_name, in_cboundary_name], 
                     out_name=out_name, axis=-1)

  # apply residual blocks
  for i in xrange(configs.nr_residual_compression):
    pipe.res_block(in_name=out_name, out_name=out_name, 
                   filter_size=configs.filter_size_compression, 
                   nonlinearity=nonlinearity, 
                   stride=1, 
                   gated=configs.gated, 
                   weight_name="res_" + str(i+1))

  # trim cboundary
  pipe.trim_tensor(in_name=in_cboundary_name, 
                  out_name=in_cboundary_name, 
                  trim=configs.nr_residual_compression*2)

# decoder state
def decoder_state(pipe, configs, in_name, out_name, lattice_size=9):

  # set nonlinearity
  nonlinearity = set_nonlinearity(configs.nonlinearity)

  # transconv network
  if configs.upsampling == 'trans_conv':
    for i in xrange(configs.nr_downsamples-1):
      filter_size = int(configs.filter_size*pow(2,configs.nr_downsamples-i-2))
      pipe.trans_conv(in_name=in_name, out_name=out_name,
                      kernel_size=4, stride=2, 
                      filter_size=filter_size,
                      weight_name="up_conv_" + str(i))
      in_name=out_name
      for j in xrange(configs.nr_residual_encoder):
        pipe.res_block(in_name=in_name, out_name=out_name, 
                       filter_size=filter_size,
                       nonlinearity=nonlinearity,
                       stride=1,
                       gated=configs.gated,
                       weight_name="res_" + str(i) + '_' + str(j))

    pipe.trans_conv(in_name=in_name, out_name=out_name,
                    kernel_size=4, stride=2,
                    filter_size=lattice_size,
                    weight_name="last_up_conv")

  # image resize network
  elif configs.upsampling == 'image_resize':
    for i in xrange(configs.nr_downsamples):
      filter_size = int(configs.filter_size*pow(2,configs.nr_downsamples-i-1))
      pipe.upsample(in_name=in_name, out_name=out_name)
      in_name=out_name
      pipe.conv(in_name=in_name, out_name=out_name,
              kernel_size=3, stride=1,
              filter_size=filter_size,
              weight_name="conv_" + str(i))

    for i in xrange(5):
      pipe.res_block(in_name=in_name, out_name=out_name, 
                     filter_size=filter_size,
                     nonlinearity=nonlinearity,
                     stride=1,
                     gated=configs.gated,
                     weight_name="res_" + str(i))
    pipe.res_block(in_name=in_name, out_name=out_name, 
                   filter_size=lattice_size,
                   nonlinearity=nonlinearity,
                   stride=1,
                   gated=configs.gated,
                   weight_name="last_res")

  pipe.nonlinearity(name=out_name, nonlinearity_name='tanh')

# discriminator
def discriminator_conditional(pipe, configs, in_boundary_name, in_state_name, in_seq_state_names, out_name):

  # set nonlinearity
  nonlinearity = set_nonlinearity('leaky_relu')

  pipe.concat_tensors(in_names=[in_boundary_name]
                             + [in_state_name] 
                             + in_seq_state_names, 
                      out_name=out_name, axis=-1) # concat on feature
  filter_size=configs.filter_size
  for i in xrange(configs.nr_residual_compression):
    begin_nonlinearity = True
    if i == 0:
      begin_nonlinearity = False
    pipe.res_block(in_name=out_name, out_name=out_name, 
                   filter_size=filter_size,
                   nonlinearity=nonlinearity,
                   stride=2,
                   gated=configs.gated,
                   begin_nonlinearity=begin_nonlinearity,
                   weight_name="res_" + str(i))
    filter_size = filter_size*2

  pipe.conv(in_name=out_name, out_name=out_name,
          kernel_size=1, stride=1,
          filter_size=256,
          nonlinearity=nonlinearity,
          weight_name="fc_1")

  pipe.conv(in_name=out_name, out_name=out_name,
          kernel_size=1, stride=1,
          filter_size=1,
          weight_name="fc_2")

  pipe.nonlinearity(name=out_name, nonlinearity_name='sigmoid')

def discriminator_unconditional(pipe, configs, in_seq_state_names, out_layer, out_class):

  # set nonlinearity
  nonlinearity = set_nonlinearity('leaky_relu')

  pipe.concat_tensors(in_names=in_seq_state_names, out_name=out_class, axis=0) # concat on batch
  filter_size=configs.filter_size
  for i in xrange(configs.nr_residual_compression):
    begin_nonlinearity = True
    if i == 0:
      begin_nonlinearity = False
    pipe.res_block(in_name=out_class, out_name=out_class, 
                   filter_size=filter_size,
                   nonlinearity=nonlinearity,
                   stride=2,
                   gated=configs.gated,
                   begin_nonlinearity=begin_nonlinearity,
                   weight_name="res_" + str(i))
    if i == 0:
      # for layer loss as seen in tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow
      pipe.rename_tensor(out_class, out_layer)
    filter_size = filter_size*2

  pipe.conv(in_name=out_class, out_name=out_class,
          kernel_size=1, stride=1,
          filter_size=256,
          nonlinearity=nonlinearity,
          weight_name="fc_1")

  pipe.conv(in_name=out_class, out_name=out_class,
          kernel_size=1, stride=1,
          filter_size=1,
          weight_name="fc_2")

  pipe.nonlinearity(name=out_class, nonlinearity_name='sigmoid')

