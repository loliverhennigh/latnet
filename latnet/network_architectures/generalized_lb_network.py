
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
  group.add_argument('--nonlinearity', help='network config', type=str,
                         default='relu')
  group.add_argument('--gated', help='network config', type=bool,
                         default=False)
  group.add_argument('--normalize', help='network config', type=str,
                         default='False')
  group.add_argument('--nr_downsamples', help='network config', type=int,
                         default=2)
  group.add_argument('--lat_q_to_net_q', help='network config', type=int,
                         default=1)

def collide(pipe, configs, in_cstate_name, in_cboundary_name, out_name):

  nonlinearity = set_nonlinearity(configs.nonlinearity)

  # boundary bounce piece
  pipe.concat_tensors(in_names=[in_cboundary_name, in_cstate_name], 
                     out_name=in_cstate_name + "_boundary", axis=-1)
  pipe.conv(in_name=in_cstate_name + "_boundary", out_name=in_cstate_name + "_boundary", 
            kernel_size=1, filter_size=128, stride=1, nonlinearity=nonlinearity, weight_name="bconv_0")
  pipe.conv(in_name=in_cstate_name + "_boundary", out_name=in_cstate_name + "_boundary", 
            kernel_size=1, filter_size=64, stride=1, nonlinearity=nonlinearity, weight_name="bconv_1")
  pipe.conv(in_name=in_cstate_name + "_boundary", out_name=in_cstate_name + "_boundary", 
            kernel_size=1, filter_size=configs.lat_q_to_net_q*9, stride=1, nonlinearity=nonlinearity, weight_name="bconv_2")

  # calc Feq
  pipe.conv(in_name=in_cstate_name, out_name=in_cstate_name + '_feq', 
            kernel_size=1, filter_size=128, stride=1, nonlinearity=nonlinearity, weight_name="conv_0")
  pipe.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
            kernel_size=1, filter_size=64, stride=1, nonlinearity=nonlinearity, weight_name="conv_1")
  pipe.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
            kernel_size=1, filter_size=configs.lat_q_to_net_q*9, stride=1, nonlinearity=nonlinearity, weight_name="conv_2")

  # collision calc
  pipe.out_tensors[out_name] = pipe.out_tensors[in_cstate_name] + pipe.out_tensors[in_cstate_name + '_feq'] + pipe.out_tensors[in_cstate_name + '_boundary']

collide_template = tf.make_template('collide', collide) 

def stream(pipe, configs, in_cstate_name, out_name):
  STREAM = np.zeros((3,3,9,9))
  STREAM[1,1,0,0] = 1.0
  STREAM[1,0,1,1] = 1.0
  STREAM[0,1,2,2] = 1.0
  STREAM[1,2,3,3] = 1.0
  STREAM[2,1,4,4] = 1.0
  STREAM[0,0,5,5] = 1.0
  STREAM[0,2,6,6] = 1.0
  STREAM[2,2,7,7] = 1.0
  STREAM[2,0,8,8] = 1.0
  STREAM = tf.constant(STREAM, dtype=1)

  # stream f
  pipe.simple_conv(in_name=in_cstate_name, out_name=out_name, kernel=tf.concat(configs.lat_q_to_net_q * [STREAM], axis=-2))

stream_template = tf.make_template('stream', stream) 

# encoder state
def encoder_state(pipe, configs, in_name, out_name):

  # set nonlinearity
  nonlinearity = set_nonlinearity(configs.nonlinearity)

  # encoding peice
  pipe.res_block(in_name=in_name, out_name=out_name,
                 filter_size=32,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=configs.gated, 
                 begin_nonlinearity=False,
                 weight_name="down_sample_res_" + str(0))
  pipe.res_block(in_name=out_name, out_name=out_name,
                 filter_size=128,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=configs.gated, 
                 #normalize=configs.normalize,
                 weight_name="down_sample_res_" + str(1))
  pipe.conv(in_name=out_name, out_name=out_name,
            filter_size=configs.lat_q_to_net_q*9,
            kernel_size=1, stride=1,
            weight_name="final_conv")

# encoder boundary
def encoder_boundary(pipe, configs, in_name, out_name):
 
  # just do the same as encoder state
  encoder_state(pipe, configs, in_name, out_name)

# compression mapping
def compression_mapping(pipe, configs, in_cstate_name, in_cboundary_name, out_name):

  # lattice steps
  for i in xrange(5):
    stream_template(pipe, configs, in_cstate_name, out_name)
    in_cstate_name = out_name
    pipe.trim_tensor(in_name=in_cboundary_name, 
                    out_name=in_cboundary_name, 
                    trim=1)
    collide_template(pipe, configs, in_cstate_name, in_cboundary_name, out_name)
    print(pipe.out_tensors[in_cboundary_name].get_shape())

# decoder state
def decoder_state(pipe, configs, in_boundary_name, in_name, out_name, lattice_size=9):

  # set nonlinearity
  nonlinearity = set_nonlinearity(configs.nonlinearity)

  # image resize network
  pipe.upsample(in_name=in_name, out_name=out_name)
  pipe.conv(in_name=out_name, out_name=out_name,
          kernel_size=3, stride=1,
          filter_size=8,
          #nonlinearity=nonlinearity,
          weight_name="conv_" + str(0))
  pipe.upsample(in_name=out_name, out_name=out_name)

  pipe.res_block(in_name=out_name, out_name=out_name, 
                 filter_size=32,
                 kernel_size=5,
                 nonlinearity=nonlinearity,
                 stride=1,
                 begin_nonlinearity=False, 
                 gated=configs.gated,
                 #normalize=configs.normalize,
                 weight_name="res_" + str(0))

  pipe.res_block(in_name=out_name, out_name=out_name, 
                 filter_size=128,
                 kernel_size=5,
                 nonlinearity=nonlinearity,
                 stride=1,
                 gated=configs.gated,
                 #normalize=configs.normalize,
                 weight_name="res_" + str(1))

  pipe.res_block(in_name=out_name, out_name=out_name, 
                 filter_size=32,
                 kernel_size=5,
                 nonlinearity=nonlinearity,
                 stride=1,
                 gated=configs.gated,
                 #normalize=configs.normalize,
                 weight_name="res_" + str(2))

  pipe.res_block(in_name=out_name, out_name=out_name, 
                 filter_size=32,
                 kernel_size=5,
                 nonlinearity=nonlinearity,
                 stride=1,
                 gated=configs.gated,
                 weight_name="res_" + str(3))

  pipe.concat_tensors(in_names=[in_boundary_name, out_name],
                      out_name=out_name, axis=-1) # concat on feature

  pipe.conv(in_name=out_name, out_name=out_name,
            kernel_size=3, stride=1,
            nonlinearity=nonlinearity,
            filter_size=64,
            weight_name="boundary_last_conv")

  pipe.conv(in_name=out_name, out_name=out_name,
            kernel_size=3, stride=1,
            filter_size=lattice_size,
            weight_name="last_conv")

  #pipe.nonlinearity(name=out_name, nonlinearity_name='tanh')

# discriminator
def discriminator_conditional(pipe, configs, in_boundary_name, in_state_name, in_seq_state_names, out_name):

  # set nonlinearity
  nonlinearity = set_nonlinearity('leaky_relu')

  # concat tensors
  pipe.concat_tensors(in_names=[in_boundary_name]
                             + [in_state_name] 
                             + in_seq_state_names, 
                      out_name=out_name, axis=-1) # concat on feature

  pipe.conv(in_name=out_name, out_name=out_name,
            kernel_size=4, stride=2,
            filter_size=32,
            nonlinearity=nonlinearity,
            weight_name="conv_0")

  pipe.conv(in_name=out_name, out_name=out_name,
            kernel_size=4, stride=2,
            filter_size=64,
            nonlinearity=nonlinearity,
            normalize=configs.normalize,
            weight_name="conv_1")

  pipe.conv(in_name=out_name, out_name=out_name,
            kernel_size=4, stride=2,
	    filter_size=128,
	    nonlinearity=nonlinearity,
            normalize=configs.normalize,
	    weight_name="conv_2")

  pipe.conv(in_name=out_name, out_name=out_name,
            kernel_size=4, stride=1,
            filter_size=256,
            nonlinearity=nonlinearity,
            normalize=configs.normalize,
            weight_name="conv_3")

  #pipe.out_tensors[out_name] = tf.reduce_mean(pipe.out_tensors[out_name], axis=[1,2], keep_dims=True)

  pipe.conv(in_name=out_name, out_name=out_name,
          kernel_size=1, stride=1,
          filter_size=1,
          weight_name="fc_0")

  pipe.nonlinearity(name=out_name, nonlinearity_name='sigmoid')

def discriminator_unconditional(pipe, configs, in_seq_state_names, out_layer, out_class):

  # set nonlinearity
  #nonlinearity = set_nonlinearity('leaky_relu')
  nonlinearity = set_nonlinearity('relu')

  pipe.concat_tensors(in_names=in_seq_state_names, out_name=out_class, axis=0) # concat on batch

  pipe.conv(in_name=out_class, out_name=out_class,
            kernel_size=4, stride=2,
            filter_size=32,
            nonlinearity=nonlinearity,
            weight_name="conv_0")

  pipe.rename_tensor(out_class, out_layer)

  pipe.conv(in_name=out_class, out_name=out_class,
            kernel_size=4, stride=2,
            filter_size=64,
            nonlinearity=nonlinearity,
            normalize=configs.normalize,
            weight_name="conv_1")

  pipe.conv(in_name=out_class, out_name=out_class,
            kernel_size=4, stride=2,
	    filter_size=128,
	    nonlinearity=nonlinearity,
            normalize=configs.normalize,
	    weight_name="conv_2")

  pipe.conv(in_name=out_class, out_name=out_class,
            kernel_size=4, stride=1,
            filter_size=256,
            nonlinearity=nonlinearity,
            weight_name="conv_3")

  #pipe.out_tensors[out_class] = tf.reduce_mean(pipe.out_tensors[out_class], axis=[1,2], keep_dims=True)

  pipe.conv(in_name=out_class, out_name=out_class,
          kernel_size=1, stride=1,
          filter_size=1,
          weight_name="fc_0")

  pipe.nonlinearity(name=out_class, nonlinearity_name='sigmoid')

