
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
                         default=4)
  group.add_argument('--lat_q_to_net_q', help='network config', type=int,
                         default=10)

#def collide(pipe, configs, in_cstate_name, in_cboundary_f_name, in_cboundary_mask_name, out_name):
def collide(pipe, configs, in_cstate_name, in_cboundary_f_name, out_name, weight_name):

  nonlinearity = set_nonlinearity(configs.nonlinearity)

  # apply boundary mask
  pipe.concat_tensors(in_names=[in_cstate_name, in_cboundary_f_name],
                      out_name=in_cstate_name + '_feq', axis=-1) # concat on feature
  pipe.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
            kernel_size=1, filter_size=256, stride=1, 
            nonlinearity=nonlinearity, weight_name="conv_0" + weight_name)
  pipe.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
            kernel_size=1, filter_size=256, stride=1, 
            nonlinearity=nonlinearity, weight_name="conv_1" + weight_name)
  pipe.conv(in_name=in_cstate_name + '_feq', out_name=out_name, 
            kernel_size=1, filter_size=configs.lat_q_to_net_q*9, stride=1, 
            weight_name="conv_2" + weight_name)

  # calc new f
  pipe.out_tensors[out_name] = tf.nn.l2_normalize(pipe.out_tensors[out_name], dim=-1) 
                                # - tf.reduce_mean(pipe.out_tensors[out_name], axis=-1, keep_dims=True), dim=-1)

  """
  # apply pressure and velocity boundary conditions
 
  # split tensor into boundary and none boundary piece
  pipe.out_tensors[in_cstate_name + '_boundary'] = (pipe.out_tensors[in_cstate_name]
                                                  * pipe.out_tensors[in_cboundary_mask_name])

  # apply boundary mask
  pipe.concat_tensors(in_names=[in_cstate_name + '_boundary', in_cboundary_f_name],
                      out_name=in_cstate_name + '_boundary', axis=-1) # concat on feature
  pipe.conv(in_name=in_cstate_name + "_boundary", out_name=in_cstate_name + "_boundary", 
            kernel_size=1, filter_size=128, stride=1, 
            nonlinearity=nonlinearity, weight_name="bconv_0")
  pipe.conv(in_name=in_cstate_name + "_boundary", out_name=in_cstate_name + "_boundary", 
            kernel_size=1, filter_size=64, stride=1, 
            nonlinearity=nonlinearity, weight_name="bconv_1")
  pipe.conv(in_name=in_cstate_name + "_boundary", out_name=in_cstate_name + "_boundary", 
            kernel_size=1, filter_size=configs.lat_q_to_net_q*9, stride=1, 
            weight_name="bconv_2")

  # calc Feq
  pipe.conv(in_name=in_cstate_name, out_name=in_cstate_name + '_feq', 
            kernel_size=1, filter_size=128, stride=1, 
            nonlinearity=nonlinearity, weight_name="conv_0")
  pipe.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
            kernel_size=1, filter_size=64, stride=1, 
            nonlinearity=nonlinearity, weight_name="conv_1")
  pipe.conv(in_name=in_cstate_name + '_feq', out_name=in_cstate_name + '_feq', 
            kernel_size=1, filter_size=configs.lat_q_to_net_q*9, stride=1, 
            weight_name="conv_2")

  # calc new f
  pipe.out_tensors['tau'] = tf.get_variable('tau', [1], initializer=tf.constant_initializer(0.1))
  pipe.out_tensors[in_cstate_name + '_f'] = (pipe.out_tensors[in_cstate_name]
           - (tf.abs(pipe.out_tensors['tau'])*(pipe.out_tensors[in_cstate_name] - pipe.out_tensors[in_cstate_name + '_feq'])))
  pipe.out_tensors[in_cstate_name + '_f'] = (pipe.out_tensors[in_cstate_name + '_f']
                                    * (1.0 - pipe.out_tensors[in_cboundary_mask_name]))
  pipe.out_tensors[out_name] = (pipe.out_tensors[in_cstate_name + '_f']
                              + pipe.out_tensors[in_cstate_name + '_boundary'])

  # apply pressure and velocity boundary conditions
  #pipe.out_tensors[out_name] = (pipe.out_tensors[in_cstate_name + '_f']
  """

collide_template = tf.make_template('collide', collide) 

def stream(pipe, configs, in_cstate_name, in_cboundary_stream_name, out_name):
  STREAM = np.zeros((3,3,configs.lat_q_to_net_q*9,configs.lat_q_to_net_q*9))
  for i in xrange(configs.lat_q_to_net_q):
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

  #pipe.out_tensors[in_cstate_name + "_trim"] = pipe.out_tensors[in_cstate_name]

  # stream f
  pipe.simple_conv(in_name=in_cstate_name, out_name=out_name, kernel=STREAM)

  # dont let stream on boundarys
  #pipe.trim_tensor(in_name=in_cstate_name + "_trim", 
  #                 out_name=in_cstate_name + "_trim", 
  #                 trim=1)

  # prevent streaming
  #pipe.out_tensors[out_name] = (pipe.out_tensors[out_name]
  #                     * pipe.out_tensors[in_cboundary_stream_name])
  #pipe.out_tensors[out_name] += (pipe.out_tensors[in_cstate_name + "_trim"]
  #                     * (1.0 - pipe.out_tensors[in_cboundary_stream_name]))

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
                 filter_size=64,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=configs.gated, 
                 weight_name="down_sample_res_" + str(1))
  pipe.res_block(in_name=out_name, out_name=out_name,
                 filter_size=128,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=configs.gated, 
                 begin_nonlinearity=False,
                 weight_name="down_sample_res_" + str(2))
  pipe.res_block(in_name=out_name, out_name=out_name,
                 filter_size=128,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=configs.gated, 
                 weight_name="down_sample_res_" + str(3))
 
  pipe.conv(in_name=out_name, out_name=out_name,
            filter_size=configs.lat_q_to_net_q*9,
            kernel_size=1, stride=1,
            weight_name="final_conv")

# encoder boundary
def encoder_boundary(pipe, configs, in_name, out_name):
 
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
                 filter_size=64,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=configs.gated, 
                 weight_name="down_sample_res_" + str(1))
  pipe.res_block(in_name=out_name, out_name=out_name,
                 filter_size=128,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=configs.gated, 
                 begin_nonlinearity=False,
                 weight_name="down_sample_res_" + str(2))
  pipe.res_block(in_name=out_name, out_name=out_name,
                 filter_size=128,
                 nonlinearity=nonlinearity, 
                 stride=2, 
                 gated=configs.gated, 
                 weight_name="down_sample_res_" + str(3))
 
  pipe.conv(in_name=out_name, out_name=out_name,
            filter_size=2*configs.lat_q_to_net_q*9,
            kernel_size=1, stride=1,
            weight_name="final_conv")

# compression mapping
def compression_mapping(pipe, configs, in_cstate_name, in_cboundary_name, out_name):

  # lattice steps
  for i in xrange(4):
    pipe.trim_tensor(in_name=in_cboundary_name, 
                    out_name=in_cboundary_name, 
                    trim=1)
    # split tensor
    pipe.split_tensor(in_name=in_cboundary_name, 
                      out_names=[in_cboundary_name + "_f", 
                                 in_cboundary_name + "_stream"],
                      num_split=2, axis=3)

    # normalize cboundary_mask between 0 and 1
    pipe.nonlinearity(name=in_cboundary_name + '_stream', 
                      nonlinearity_name="sigmoid")

    # stream and collide
    stream_template(pipe, configs, in_cstate_name, in_cboundary_name + '_stream', out_name)
    in_cstate_name = out_name
    collide(pipe, configs, in_cstate_name, in_cboundary_name + '_f', out_name, weight_name=str(i))

# decoder state
def decoder_state(pipe, configs, in_boundary_name, in_name, out_name, lattice_size=9):

  # set nonlinearity
  nonlinearity = set_nonlinearity(configs.nonlinearity)

  # image resize network
  pipe.upsample(in_name=in_name, out_name=out_name)
  pipe.conv(in_name=out_name, out_name=out_name,
          kernel_size=3, stride=1,
          filter_size=64,
          nonlinearity=nonlinearity,
          weight_name="conv_" + str(0))
  pipe.upsample(in_name=out_name, out_name=out_name)
  pipe.conv(in_name=out_name, out_name=out_name,
          kernel_size=3, stride=1,
          filter_size=64,
          nonlinearity=nonlinearity,
          weight_name="conv_" + str(1))
  pipe.upsample(in_name=out_name, out_name=out_name)
  pipe.conv(in_name=out_name, out_name=out_name,
          kernel_size=3, stride=1,
          filter_size=64,
          nonlinearity=nonlinearity,
          weight_name="conv_" + str(2))
  pipe.upsample(in_name=out_name, out_name=out_name)

  pipe.res_block(in_name=out_name, out_name=out_name, 
                 filter_size=32,
                 kernel_size=5,
                 nonlinearity=nonlinearity,
                 stride=1,
                 begin_nonlinearity=False, 
                 gated=configs.gated,
                 weight_name="res_" + str(0))

  pipe.res_block(in_name=out_name, out_name=out_name, 
                 filter_size=128,
                 kernel_size=5,
                 nonlinearity=nonlinearity,
                 stride=1,
                 gated=configs.gated,
                 weight_name="res_" + str(1))

  pipe.res_block(in_name=out_name, out_name=out_name, 
                 filter_size=128,
                 kernel_size=5,
                 nonlinearity=nonlinearity,
                 stride=1,
                 gated=configs.gated,
                 weight_name="res_" + str(2))

  pipe.res_block(in_name=out_name, out_name=out_name, 
                 filter_size=128,
                 kernel_size=5,
                 nonlinearity=nonlinearity,
                 stride=1,
                 gated=configs.gated,
                 weight_name="res_" + str(3))


  pipe.res_block(in_name=out_name, out_name=out_name, 
                 filter_size=32,
                 kernel_size=5,
                 nonlinearity=nonlinearity,
                 stride=1,
                 gated=configs.gated,
                 weight_name="res_" + str(4))

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

