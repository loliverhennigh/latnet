

from copy import copy
import tensorflow as tf

import lattice as lat
import nn as nn

from shape_converter import ShapeConverter
from optimizer import Optimizer
from saver import Saver

class LatNet:
  def __init__(self, config):
    # in and out tensors
    self.in_tensors = {}
    self.out_tensors = {}

    # shape converter from in_tensor to out_tensor
    self.shape_converters = {}

    # needed configs
    self.config = config # TODO remove this when config is corrected
    self.dims = 2
    self.lattice_size = 9
    self.network_dir  = config.latnet_network_dir
    self.network_name = config.network_name
    self.seq_length = config.seq_length

    if self.network_name == "basic_network":
      import network_architectures.basic_network as net
    elif self.network_name == "advanced_network":
      import network_architectures.advanced_network as net
    else:
      print("network name not found")
      exit()

    # piecese of network
    self.encoder_state                = tf.make_template('encoder_state', net.encoder_state)
    self.encoder_boundary             = tf.make_template('encoder_boundary', net.encoder_boundary)
    self.compression_mapping_boundary = tf.make_template('compression_mapping_boundary', net.compression_mapping_boundary)
    self.compression_mapping          = tf.make_template('compression_mapping', net.compression_mapping)
    self.decoder_state                = tf.make_template('decoder_state', net.decoder_state)
    self.network_config               = net.CONFIGS
    self.padding                      = net.PADDING

  def train_unroll(self):

    # graph
    with tf.Graph().as_default():

      ###### Inputs to Graph ######
      # global step counter
      self.add_tensor('global_step', tf.get_variable('global_step', [], 
                       initializer=tf.constant_initializer(0), trainable=False))
      # make input state and boundary
      self.add_tensor('state', tf.placeholder(tf.float32, (1 + self.dims) * [None] + [self.lattice_size]))
      tf.summary.image('state', lat.lattice_to_norm(self.in_tensors['state']))
      self.add_tensor('boundary', tf.placeholder(tf.float32, (1 + self.dims) * [None] + [4]))
      tf.summary.image('boundary', self.in_tensors['boundary'][...,0:1])
      # make seq of output states
      for i in xrange(self.seq_length):
        self.add_tensor('true_state_' + str(i), tf.placeholder(tf.float32, (2 + self.dims) * [None]))
        tf.summary.image('true_state_' + str(i), lat.lattice_to_norm(self.in_tensors['true_state_' + str(i)]))
  
      ###### Unroll Graph ######
      # encode
      self.encoder_state(self, in_name="state", out_name="cstate_0")
      self.encoder_boundary(self, in_name="boundary", out_name="cboundary")
  
      # unroll all
      for i in xrange(self.seq_length):
        # decode and add to list
        self.out_tensors['cstate_00'] = self.out_tensors['cstate_0']
        self.decoder_state(self, in_name="cstate_" + str(i), out_name="pred_state_" + str(i))
  
        # compression mapping
        self.compression_mapping(self, in_name="cstate_" + str(i), out_name="cstate_" + str(i))
  
        # apply boundary
        self.out_tensors['cboundary'] = self.out_tensors['cboundary'][:,4:-4,4:-4] # TODO fix this
        self.compression_mapping_boundary(self, in_cstate_name="cstate_" + str(i), 
                                                in_cboundary_name="cboundary", 
                                                out_name="cstate_" + str(i+1))
  
        # make image summary
        tf.summary.image('predicted_state_vel_', lat.lattice_to_norm(self.out_tensors['pred_state_' + str(i)]))
  
      ###### Loss Operation ######
      # define mse loss
      self.out_tensors["loss"] = 0.0
      for i in xrange(self.seq_length):
        self.mse(true_name='true_state_' + str(i),
                 pred_name='pred_state_' + str(i),
                 loss_name='loss_' + str(i))
        self.out_tensors['loss'] += self.out_tensors['loss_' + str(i)]
      tf.summary.scalar('loss', self.out_tensors['loss'])
  
      ###### Train Operation ######
      all_params = tf.trainable_variables()
      self.optimizer = Optimizer(self.config)
      self.optimizer.compute_gradients(self.out_tensors['loss'], all_params)
      self.out_tensors['train_op'] = self.optimizer.train_op(all_params, self.out_tensors['global_step'])
  
      ###### Start Session ######
      self.sess = self.start_session()
  
      ###### Saver Operation ######
      graph_def = self.sess.graph.as_graph_def(add_shapes=True)
      self.saver = Saver(self.config, self.network_config, graph_def)
      self.saver.load_checkpoint(self.sess)

  def train_shape_converter(self):
    shape_converters = {}
    for i in xrange(self.seq_length):
      name = ("state", "pred_state_" + str(i))
      shape_converters[name] = self.shape_converters[name]
    return shape_converters

  def train_step(self, feed_dict):

    # perform train step
    tf_feed_dict = {}
    for name in feed_dict.keys():
      tf_feed_dict[self.in_tensors[name]] = feed_dict[name]
    _, l = self.sess.run([self.out_tensors['train_op'], self.out_tensors['loss']], feed_dict=tf_feed_dict)
   
    # print required data and save
    step = self.sess.run(self.out_tensors['global_step'])
    if step % 10 == 0:
      print("current loss is " + str(l))
      print("current step is " + str(step))
    if step % self.config.save_freq == 0:
      print("saving...")
      self.saver.save_summary(self.sess, tf_feed_dict, int(self.sess.run(self.out_tensors['global_step'])))
      self.saver.save_checkpoint(self.sess, int(self.sess.run(self.out_tensors['global_step'])))
 
  def eval_unroll(self):

    # graph
    with tf.Graph().as_default():

      ###### Inputs to Graph ######
      # make input state and boundary
      self.add_tensor('state',     tf.placeholder(tf.float32, (1 + self.dims) * [None] + [self.lattice_size]))
      self.add_tensor('boundary',  tf.placeholder(tf.float32, (1 + self.dims) * [None] + [4]))
      self.add_tensor('cstate',    tf.placeholder(tf.float32, (1 + self.dims) * [None] + [self.network_config['filter_size_compression']]))
      self.add_tensor('cboundary', tf.placeholder(tf.float32, (1 + self.dims) * [None] + [2*self.network_config['filter_size_compression']]))
  
      ###### Unroll Graph ######
      # encoders
      self.encoder_state(self, in_name="state", out_name="cstate_from_state")
      self.encoder_boundary(self, in_name="boundary", out_name="cboundary_from_boundary")
  
      # compression mapping
      self.compression_mapping(self, in_name="cstate", out_name="cstate_from_cstate")
      self.compression_mapping_boundary(self, in_cstate_name="cstate_from_cstate", 
                                              in_cboundary_name="cboundary", 
                                              out_name="cstate_from_cstate")
  
      # decoder
      self.decoder_state(self, in_name="cstate", out_name="state_from_cstate")
  
      ###### Start Session ######
      self.sess = self.start_session()
  
      ###### Saver Operation ######
      graph_def = self.sess.graph.as_graph_def(add_shapes=True)
      self.saver = Saver(self.config, self.network_config, graph_def)
      self.saver.load_checkpoint(self.sess)
  
    ###### Function Wrappers ######
    # network functions
    state_encoder    = lambda x: self.sess.run(self.out_tensors['cstate_from_state'], 
                                 feed_dict={self.in_tensors['state']:x})
    boundary_encoder = lambda x: self.sess.run(self.out_tensors['cboundary_from_boundary'], 
                                 feed_dict={self.in_tensors['boundary']:x})
    cmapping         = lambda x, y: self.sess.run(self.out_tensors['cstate_from_cstate'], 
                                 feed_dict={self.in_tensors['cstate']:x,
                                            self.in_tensors['cboundary']:y})
    decoder           = lambda x: self.sess.run(self.out_tensors['state_from_cstate'], 
                                 feed_dict={self.in_tensors['cstate']:x})
    # shape converters
    encoder_shape_converter = self.shape_converters['state', 'cstate_from_state']
    cmapping_shape_converter = self.shape_converters['cstate', 'cstate_from_cstate']
    decoder_shape_converter = self.shape_converters['cstate', 'state_from_cstate']

    return (state_encoder, boundary_encoder, cmapping, decoder,
            encoder_shape_converter, cmapping_shape_converter, 
            decoder_shape_converter) # This should probably be cleaned up

  def conv(self, in_name, out_name,
           kernel_size, stride, filter_size, 
           weight_name="conv", nonlinearity=None):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.conv_layer(self.out_tensors[in_name],
                                              kernel_size, stride, filter_size, 
                                              name=weight_name, nonlinearity=None)

    # add conv to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_conv(kernel_size, stride)

    # rename tensor
    #self.rename_out_tensor(in_name, out_name)

  def trans_conv(self, in_name, out_name,
                 kernel_size, stride, filter_size, 
                 weight_name="trans_conv", nonlinearity=None):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.transpose_conv_layer(self.out_tensors[in_name],
                                                        kernel_size, stride, filter_size, 
                                                        name=weight_name, nonlinearity=None)

    # add conv to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_trans_conv(kernel_size, stride)

    # rename tensor
    #self.rename_out_tensor(in_name, out_name)

  def res_block(self, in_name, out_name,
                filter_size=16, 
                nonlinearity=nn.concat_elu, 
                keep_p=1.0, stride=1, 
                gated=True, weight_name="resnet", 
                begin_nonlinearity=True, 
                normalize=None):

    # add res block to tensor computation
    self.out_tensors[out_name] = nn.res_block(self.out_tensors[in_name], a=None,
                                            filter_size=filter_size, 
                                            nonlinearity=nonlinearity, 
                                            keep_p=keep_p, stride=stride, 
                                            gated=gated, name=weight_name, 
                                            begin_nonlinearity=begin_nonlinearity, 
                                            normalize=normalize)

    # add res block to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_res_block(stride)

    # rename tensor
    #self.rename_out_tensor(in_name, out_name)

  def split_tensor(self, in_name,
                   a_out_name, b_out_name,
                   num_split, axis):

    # perform split on tensor
    self.out_tensors[a_out_name], self.out_tensors[b_out_name]  = tf.split(self.out_tensors[in_name],
                                                                           num_split, axis)
    # add to shape converters
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], a_out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], b_out_name] = copy(self.shape_converters[name])

    # rm old tensor
    #self.rm_tensor(in_name)

  def image_combine(self, a_name, b_name, mask_name, out_name):
    # as seen in "Generating Videos with Scene Dynamics" figure 1
    self.out_tensors[out_name] = ((self.out_tensors[a_name] *      self.out_tensors[mask_name] )
                                + (self.out_tensors[b_name] * (1 - self.out_tensors[mask_name])))

    # take shape converters from a_name
    # TODO add tools to how shape converters are merged to make safer
    for name in self.shape_converters.keys():
      if name[1] == a_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == b_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == mask_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])

    # rm old tensors
    #self.rm_tensor(   a_name)
    #self.rm_tensor(   b_name)
    #self.rm_tensor(mask_name)

  def nonlinearity(self, name, nonlinearity_name):
    nonlin = nn.set_nonlinearity(nonlinearity_name)
    self.out_tensors[name] = nonlin(self.out_tensors[name])

  def mse(self, true_name, pred_name, loss_name):
    self.out_tensors[loss_name] = tf.nn.l2_loss(self.in_tensors[ true_name] 
                                              - self.out_tensors[pred_name])
    tf.summary.scalar('loss_' + true_name + "_and_" + pred_name, self.out_tensors[loss_name])

  def combine_pipe(self, other_pipe):
    self.in_tensors.update(other_pipe.in_tensors)
    self.out_tensors.update(other_pipe.out_tensors)
    self.shape_converters.update(other_pipe.shape_converters)

  def split_pipe(self, old_name, new_name):
    self.out_tensors[new_name] = self.out_tensors[old_name]
    for name in self.shape_converters.keys():
      if name[1] == old_name:
        self.shape_converters[name[0],new_name] = copy(self.shape_converters[name])

  def remove_tensor(self, rm_name):
    self.out_tensors.pop(rm_name)
    for name in self.shape_converters.keys():
      if name[1] == rm_name:
        self.shape_converters.pop(name)
 
  def add_tensor(self, name, tensor):
    self.in_tensors[name] = tensor
    self.out_tensors[name] = tensor
    self.shape_converters[name,name] = ShapeConverter()
      
  def rename_out_tensor(self, old_name, new_name):
    self.out_tensors[new_name] = self.out_tensors.pop(old_name)
    for name in self.shape_converters.keys():
      if name[1] == old_name:
        self.shape_converters[name[0],new_name] = self.shape_converters.pop(name)

  def start_session(self):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess




