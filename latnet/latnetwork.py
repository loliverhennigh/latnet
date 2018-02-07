

import time
from copy import copy
import os
from termcolor import colored, cprint
import tensorflow as tf
import numpy as np

import lattice
import nn as nn

from shape_converter import ShapeConverter
from optimizer import Optimizer
from shape_converter import SubDomain
from network_saver import NetworkSaver

class LatNet(object):
  # default network name
  network_name = 'advanced_network'

  def __init__(self, config, network_name, script_name):
    # in and out tensors
    self.in_tensors = {}
    self.out_tensors = {}

    # shape converter from in_tensor to out_tensor
    self.shape_converters = {}

    # needed configs
    self.config = config # TODO remove this when config is corrected
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.network_dir  = config.latnet_network_dir
    self.network_name = network_name
    self.script_name = script_name
    self.seq_length = config.seq_length
    self.gan = config.gan
    gpus = config.gpus.split(',')
    self.gpus = map(int, gpus)
    self.loss_stats = {}
    self.time_stats = {}
    self.start_time = time.time()
    self.tic = 0
    self.tic = 0
    self.stats_history_length = 300

    if self.network_name == "test_network":
      import network_architectures.test_network as net
    elif self.network_name == "advanced_network":
      import network_architectures.advanced_network as net
    else:
      print("network name not found")
      exit()

    # piecese of network
    self.encoder_state                = tf.make_template('encoder_state', net.encoder_state)
    self.encoder_boundary             = tf.make_template('encoder_boundary', net.encoder_boundary)
    self.compression_mapping          = tf.make_template('compression_mapping', net.compression_mapping)
    self.decoder_state                = tf.make_template('decoder_state', net.decoder_state)
    self.discriminator_conditional    = tf.make_template('discriminator_conditional', net.discriminator_conditional)
    self.discriminator_unconditional  = tf.make_template('discriminator_unconditional', net.discriminator_unconditional)

  @classmethod
  def add_options(cls, group, network_name):
    if network_name == "test_network":
      from network_architectures.test_network import add_options
    elif network_name == "advanced_network":
      from network_architectures.advanced_network import add_options
    add_options(group)

  def train_unroll(self):

    # graph
    with tf.Graph().as_default():

      ###### Inputs to Graph ######
      # global step counter
      self.add_step_counter('gen_global_step') 
      if self.gan:
        self.add_step_counter('disc_global_step') 

      for i in xrange(len(self.gpus)):
        gpu_str = '_gpu_' + str(self.gpus[i])
        seq_str = lambda x: '_' + str(x) + gpu_str
        with tf.device('/gpu:%d' % self.gpus[i]):
          # make input state and boundary
          self.add_tensor('state' + gpu_str, (1 + self.DxQy.dims) * [None] + [self.DxQy.Q])
          self.add_tensor('boundary' + gpu_str, (1 + self.DxQy.dims) * [None] + [4])
          self.add_tensor('mask' + gpu_str, (1 + self.DxQy.dims) * [None] + [1])
          if i == 0:
            with tf.device('/cpu:0'):
              self.lattice_summary(in_name='state' + gpu_str, summary_name='state')
              tf.summary.image('boundary', self.in_tensors['boundary' + gpu_str][...,0:1])
              tf.summary.image('mask', self.in_tensors['mask' + gpu_str])
          # make seq of output states
          for j in xrange(self.seq_length):
            self.add_tensor('true_state' + seq_str(j), (1 + self.DxQy.dims) * [None] + [self.DxQy.Q])
            if i == 0:
              with tf.device('/cpu:0'):
                self.lattice_summary(in_name='true_state_' + str(j), summary_name='true_state')
      
          ###### Unroll Graph ######
          ### encode ###
          self.encoder_state(self, self.config, 
                             in_name="state" + gpu_str, 
                             out_name="cstate" + seq_str(0))
          self.encoder_boundary(self, self.config, 
                                in_name="boundary" + gpu_str, 
                                out_name="cboundary" + gpu_str)
      
          ### unroll on the compressed state ###
          for j in xrange(self.seq_length):
            # compression mapping
            self.add_shape_converter("cstate" + seq_str(j))
            self.compression_mapping(self, self.config,
                                     in_cstate_name="cstate" + seq_str(j),
                                     in_cboundary_name="cboundary" + gpu_str,
                                     out_name="cstate" + seq_str(j+1))

          ### decode all compressed states ###
          for j in xrange(self.seq_length):
            self.match_trim_tensor(in_name="cstate" + seq_str(j), 
                                   match_name="cstate" + seq_str(self.seq_length-1), 
                                   out_name="cstate" + seq_str(j))
            self.decoder_state(self, self.config, 
                               in_name="cstate" + seq_str(j), 
                               out_name="pred_state" + seq_str(j))
            self.out_tensors["pred_state" + seq_str(j)] = (self.out_tensors['mask' + gpu_str]
                                                         * self.out_tensors["pred_state" + seq_str(j)])
  
            if i == 0:
              with tf.device('/cpu:0'):
                # make image summary
                tf.lattice_summary(in_name='pred_state' + seq_str(j), out_name='pred_vel_' + str(j))
 
          ### unconditional discriminator of gan ###
          if self.gan is not None:
            seq_pred_state_names = []
            seq_true_state_names = []
            for j in xrange(self.seq_length):
              seq_pred_state_names.append("pred_state" + seq_str(j))
              seq_true_state_names.append("true_state" + seq_str(j))
            self.discriminator_unconditional(self, self.config, 
                                             in_seq_state_names=seq_pred_state_names,
                                             out_layer='D_un_layer_pred' + gpu_str)
                                             out_class='D_un_class_pred' + gpu_str)
            self.discriminator_unconditional(self, self.config, 
                                             in_seq_state_names=seq_true_state_names,
                                             out_layer='D_un_layer_true' + gpu_str)
                                             out_class='D_un_class_true' + gpu_str)
      
          ###### Loss Operation ######
          num_samples = (self.seq_length * self.config.batch_size * len(self.config.gpus))

          ### L1 loss ###
          self.out_tensors["loss_l1" + gpu_str] = 0.0
          for j in xrange(self.seq_length):
            # normalize loss to 256 by 256 state for now
            pred_shape = tf.shape(self.out_tensors['true_state' + seq_str(j)])
            num_cells = tf.cast(tf.reduce_prod(pred_shape[1:]), dtype=tf.float32)
            l1_factor = (256.0*256.0*9.0) / (num_cells*num_samples)
            self.l1_loss(true_name='true_state' + seq_str(j),
                         pred_name='pred_state' + seq_str(j),
                         loss_name='loss_l1' + seq_str(j), 
                         factor=l1_factor)
            # add up losses
            self.out_tensors['loss_l1' + gpu_str] += self.out_tensors['loss_l1' + seq_str(j)] 

          ### Unconditional GAN loss ###
          if self.gan:
            # normal gan losses
            gan_factor = 1.0 / num_samples
            self.gen_loss(class_name='D_un_class_pred' + gpu_str,
                          loss_name='loss_gen_un_class' + gpu_str,
                          factor=gan_factor)
            self.disc_loss(true_class_name='D_un_class_true' + gpu_str,
                           pred_class_name='D_un_class_pred' + gpu_str,
                           loss_name='loss_disc_un_class' + gpu_str,
                           factor=gan_factor)
            # layer loss as seen in tempoGAN: A Temporally Coherent, Volumetric GAN for
            # Super-resolution Fluid Flow
            pred_shape = tf.shape(self.out_tensors['D_un_layer_pred' + gpu_str])
            num_cells = tf.cast(tf.reduce_prod(pred_shape[1:]), dtype=tf.float32)
            l2_layer_factor = (256.0*256.0*9.0) / (num_cells*num_samples)
            self.l2_loss(true_name='D_un_layer_true' + gpu_str,
                         pred_name='D_un_layer_pred' + gpu_str,
                         loss_name='loss_layer_l2' + seq_str(i),
                         factor=l2_layer_factor)

          ### Conditional GAN loss ###
          # TODO add cond gan

          #### add all losses together ###
          self.out_tensors['loss_gen' + gpu_str] =  0.0
          self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_l1' + gpu_str]
          if self.gan:
            self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_gen_un_class' + gpu_str]
            self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_layer_l2' + gpu_str]
            self.out_tensors['loss_disc' + gpu_str] = self.out_tensors['loss_disc_un_class' + gpu_str]
 
          ###### Grad Operation ######
          if i == 0:
            all_params = tf.trainable_variables()
            gen_params = [v for i, v in enumerate(all_params) if "discriminator" not in v.name[:v.name.index(':')]]
            disc_params = [v for i, v in enumerate(all_params) if "discriminator" in v.name[:v.name.index(':')]]
          self.out_tensors['gen_grads' + gpu_str] = tf.gradients(self.out_tensors['loss' + gpu_str], gen_params)
          if self.gan:
            self.out_tensors['disc_grads' + gpu_str] = tf.gradients(self.out_tensors['disc_loss' + gpu_str], disc_params)

      ###### Round up losses and Gradients on gpu:0 ######
      ### Round up losses ###
      with tf.device('/gpu:%d' % self.gpus[0]):
        gpu_str = lambda x: '_gpu_' + str(self.gpus[x])
        for i in range(1, len(self.gpus)):
          self.out_tensors['loss_gen' + gpu_str(0)] += self.out_tensors['loss_gen' + gpu_str(i)]
          self.out_tensors['loss_l1' + gpu_str(0)] += self.out_tensors['loss_l1' + gpu_str(i)]
          if self.gan:
            self.out_tensors['loss_gen_un_class' + gpu_str(0)] += self.out_tensors['loss_gen_un_class' + gpu_str(i)]
            self.out_tensors['loss_layer_l2' + gpu_str(0)] += self.out_tensors['loss_layer_l2' + gpu_str(i)]
            self.out_tensors['loss_disc_un_class' + gpu_str(0)] += self.out_tensors['loss_disc_un_class' + gpu_str(i)]
            self.out_tensors['loss_disc' + gpu_str(0)] += self.out_tensors['loss_disc' + gpu_str(i)]
      self.rename_tensor(old_name='loss_gen' + gpu_str(0), new_name='loss_gen')
      self.rename_tensor(old_name='loss_l1' + gpu_str(0), new_name='loss_l1')
      if self.gan:
        self.rename_tensor(old_name='loss_gen_un_class' + gpu_str(0), new_name='loss_gen_un_class')
        self.rename_tensor(old_name='loss_layer_l2' + gpu_str(0), new_name='loss_layer_l2')
        self.rename_tensor(old_name='loss_disc_un_class' + gpu_str(0), new_name='loss_disc_un_class')
        self.rename_tensor(old_name='loss_disc' + gpu_str(0), new_name='loss_disc')

      ### Round up gradients ###
      with tf.device('/gpu:%d' % self.gpus[0]):
        gpu_str = lambda x: '_gpu_' + str(self.gpus[x])
        for i in range(1, len(self.gpus)):
          # gradients 
          for j in range(len(self.out_tensors['gen_grads' + gpu_str(0)])):
              self.out_tensors['gen_grads' + gpu_str(0)][j] += self.out_tensors['gen_grads' + gpu_str(i)][j]
          if self.gan:
            for j in range(len(self.out_tensors['disc_grads' + gpu_str(0)])):
              self.out_tensors['disc_grads' + gpu_str(0)][j] += self.out_tensors['disc_grads' + gpu_str(i)][j]

      ### add loss summary ###
      tf.summary.scalar('loss_l1', self.out_tensors['loss_l1'])
      if self.gan:
        tf.summary.scalar('loss_gen_un_class', self.out_tensors['loss_gen_un_class'])
        tf.summary.scalar('loss_layer_l2', self.out_tensors['loss_layer_l2'])
        tf.summary.scalar('loss_disc', self.out_tensors['loss_disc'])

      ###### Train Operation ######
      self.gen_optimizer = Optimizer(self.config)
      self.out_tensors['gen_train_op'] = self.optimizer.train_op(gen_params, 
                                               self.out_tensors['gen_grads' + gpu_str(0)], 
                                               self.out_tensors['gen_global_step'])
      if self.gan:
        self.disc_optimizer = Optimizer(self.config)
        self.out_tensors['disc_train_op'] = self.disc_optimizer.train_op(disc_params, 
                                                   self.out_tensors['disc_grads' + gpu_str(0)], 
                                                   self.out_tensors['disc_global_step'])
  
      ###### Start Session ######
      self.sess = self.start_session()
  
      ###### Saver Operation ######
      graph_def = self.sess.graph.as_graph_def(add_shapes=True)
      self.saver = NetworkSaver(self.config, self.network_name, self.script_name, graph_def)
      self.saver.load_checkpoint(self.sess)

  def train_shape_converter(self):
    shape_converters = {}
    for i in xrange(len(self.gpus)):
      gpu_str = '_gpu_' + str(self.gpus[i])
      for j in xrange(self.seq_length):
        name = ("state" + gpu_str, "pred_state_" + str(j) + gpu_str)
        shape_converters[name] = self.shape_converters[name]
        name = ("state" + gpu_str, "cstate_" + str(j) + gpu_str)
        shape_converters[name] = self.shape_converters[name]
    return shape_converters

  def train(self, dataset):
  
    # steps per print (hard set for now untill done debugging)
    steps_per_print = 20
 
    while True: 
      # get batch of data
      feed_dict = dataset.minibatch()

      # perform optimization step for gen
      gen_output = self.run(['gen_train_op', 'loss_gen'], feed_dict=feed_dict, return_dict=True)
      if self.gan:
        disc_output = self.run(['disc_train_op', 'loss_disc'], feed_dict=feed_dict, return_dict=True)
        gen_output.update(disc_output)

      # update loss summary
      self.update_loss_stats(gen_output)
    
      # update time summary
      self.update_time_stats()

      # print required data and save
      step = self.run('gen_global_step')
      if step % steps_per_print == 0:
        self.saver.save_summary(self.sess, feed_dict, int(self.run('gen_global_step')))
        self.print_stats(self.loss_stats, self.time_stats, dataset.queue_stats())

      if step % self.config.save_network_freq == 0:
        self.saver.save_checkpoint(self.sess, int(self.run('gen_global_step')))

  def eval_unroll(self):

    # graph
    with tf.Graph().as_default():

      ###### Inputs to Graph ######
      # make input state and boundary
      self.add_tensor('state',     tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [self.DxQy.Q]))
      self.add_tensor('boundary',  tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [4]))
      self.add_tensor('cstate',    tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [self.config.filter_size_compression]))
      self.add_tensor('cboundary', tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [self.config.filter_size_compression]))
  
      ###### Unroll Graph ######
      # encoders
      self.encoder_state(self, self.config, in_name="state", out_name="cstate_from_state")
      self.encoder_boundary(self, self.config, in_name="boundary", out_name="cboundary_from_boundary")
  
      # compression mapping
      self.compression_mapping(self, self.config, 
                               in_cstate_name="cstate", 
                               in_cboundary_name="cboundary",
                               out_name="cstate_from_cstate")
  
      # decoder
      self.decoder_state(self, self.config, in_name="cstate", out_name="state_from_cstate")
      self.out_tensors['vel_from_cstate'] = self.DxQy.lattice_to_vel(self.out_tensors['state_from_cstate'])
      self.out_tensors['rho_from_cstate'] = self.DxQy.lattice_to_rho(self.out_tensors['state_from_cstate'])

      ###### Start Session ######
      self.sess = self.start_session()
  
      ###### Saver Operation ######
      graph_def = self.sess.graph.as_graph_def(add_shapes=True)
      self.saver = NetworkSaver(self.config, self.network_name, self.script_name, graph_def)
      self.saver.load_checkpoint(self.sess)
  
    ###### Function Wrappers ######
    # network functions
    state_encoder    = lambda x: self.run('cstate_from_state', 
                                 feed_dict={'state':x})
    boundary_encoder = lambda x: self.run('cboundary_from_boundary', 
                                 feed_dict={'boundary':x})
    cmapping         = lambda x, y: self.run(self.out_tensors['cstate_from_cstate'], 
                                 feed_dict={'cstate':x,
                                            'cboundary':y})
    decoder           = lambda x: self.run(['vel_from_cstate', 
                                            'rho_from_cstate'], 
                                 feed_dict={'cstate':x})
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

  def trans_conv(self, in_name, out_name,
                 kernel_size, stride, filter_size, 
                 weight_name="trans_conv", nonlinearity=None):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.transpose_conv_layer(self.out_tensors[in_name],
                                                        kernel_size, stride, filter_size, 
                                                        name=weight_name, nonlinearity=nonlinearity)

    # add conv to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_trans_conv(kernel_size, stride)

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

  def concat_tensors(self, in_names, out_name, axis=-1):
    in_tensors = [self.out_tensors[name] for name in in_names]
    self.out_tensors[out_name] = tf.concat(in_tensors, 
                                           axis=axis)

    # add to shape converters
    for name in self.shape_converters.keys():
      if name[1] in in_names:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])

  def trim_tensor(self, in_name, out_name, trim):
    if trim > 0:
      if len(self.out_tensors[in_name].get_shape()) == 4:
        self.out_tensors[out_name] = self.out_tensors[in_name][:,trim:-trim, trim:-trim]
      elif len(self.out_tensors[in_name].get_shape()) == 5:
        self.out_tensors[out_name] = self.out_tensors[in_name][:,trim:-trim, trim:-trim, trim:-trim]

  def match_trim_tensor(self, in_name, match_name, out_name):

    subdomain = SubDomain(self.DxQy.dims*[0], self.DxQy.dims*[1])
    new_subdomain = self.shape_converters[in_name, match_name].out_in_subdomain(subdomain)
    self.trim_tensor(in_name, out_name, abs(new_subdomain.pos[0]))

  def image_combine(self, a_name, b_name, mask_name, out_name):
    # as seen in "Generating Videos with Scene Dynamics" figure 1
    self.out_tensors[out_name] = ((self.out_tensors[a_name] *      self.out_tensors[mask_name] )
                                + (self.out_tensors[b_name] * (1 - self.out_tensors[mask_name])))

    # take shape converters from a_name
    for name in self.shape_converters.keys():
      if name[1] == a_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == b_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == mask_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])

  def add_shape_converter(self, name):
    self.shape_converters[name, name] = ShapeConverter()

  def nonlinearity(self, name, nonlinearity_name):
    nonlin = nn.set_nonlinearity(nonlinearity_name)
    self.out_tensors[name] = nonlin(self.out_tensors[name])

  def l1_loss(self, true_name, pred_name, loss_name, factor=None):
    self.out_tensors[loss_name] = tf.abs(self.in_tensors[ true_name] 
                                       - self.out_tensors[pred_name])
    if factor is not None:
      self.out_tensors[loss_name] = factor * self.out_tensors[loss_name]

  def l2_loss(self, true_name, pred_name, loss_name, factor=None):
    self.out_tensors[loss_name] = tf.nn.l2_loss(self.in_tensors[ true_name] 
                                              - self.out_tensors[pred_name])
    if factor is not None:
      self.out_tensors[loss_name] = factor * self.out_tensors[loss_name]

  def gen_loss(self, class_name, loss_name, factor=None):
    self.out_tensors[loss_name] = -tf.reduce_mean(tf.log(self.out_tensors[class_name]))
    if factor is not None:
      self.out_tensors[loss_name] = factor * self.out_tensors[loss_name]

  def disc_loss(self, true_class_name, pred_class_name, loss_name, factor=None):
    self.out_tensors[loss_name] = -tf.reduce_mean(tf.log(self.out_tensors[true_class_name])
                                          + tf.log(1.0 - self.out_tensors[pred_class_name]))
    if factor is not None:
      self.out_tensors[loss_name] = factor * self.out_tensors[loss_name]

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

  def add_step_counter(self, name):
    counter = tf.get_variable(name, [], 
              initializer=tf.constant_initializer(0), trainable=False))
    self.in_tensors[name] = counter
    self.out_tensors[name] = counter
    self.shape_converters[name,name] = ShapeConverter()
 
  def add_tensor(self, name, shape):
    tensor = tf.placeholder(tf.float32, shape)
    self.in_tensors[name] = tensor
    self.out_tensors[name] = tensor
    self.shape_converters[name,name] = ShapeConverter()
      
  def rename_tensor(self, old_name, new_name):
    # this may need to be handled with more care
    self.out_tensors[new_name] = self.out_tensors[old_name]

  def lattice_summary(self, in_name, summary_name, 
                      display_norm=True):
    # TODO add more display options
    if display_norm:
      tf.summary.image(summary_name, self.DxQy.lattice_to_norm(self.in_tensors[in_name]))

  def run(self, out_names, feed_dict=None, return_dict=False):
    # convert out_names to tensors 
    if type(out_names) is list:
      out_tensors = [self.out_tensors[x] for x in out_names]
    else:
      out_tensors = out_names

    # convert feed_dict to tensorflow version
    tf_feed_dict = {}
    for name in feed_dict.keys():
      tf_feed_dict[self.in_tensors[name]] = feed_dict[name]

    # run with tensorflow
    tf_output = self.sess.run(out_tensors feed_dict=tf_feed_dict)
    
    # possibly convert output to dictionary 
    if return_dict:
      output = {}
      if type(out_names) is list:
        for i in xrange(len(out_names)):
          output[out_names[i]] = tf_output[i]
      else:
        output[out_names] = tf_output
    else:
      output = tf_output
    return output

  def update_loss_stats(self, output):
    for name in output.keys():
      if 'loss' in name:
        # update loss history
        if name not in self.loss_stats.keys():
          self.loss_stats[name + '_history'] = []
        self.loss_stats[name + '_history'].append(output[name])
        if self.loss_stats[name + '_history'] > self.stats_history_length:
          self.loss_stats[name + '_history'].pop(0)
        # update loss
        self.loss_stats[name] = output[name]
        # update ave loss
        self.loss_stats[name + '_ave'] = (np.sum(np.array(self.loss_stats[name + '_history']))
                                         / len(self.loss_stats[name + '_history']))

  def update_time_stats(self):
    # stop timer
    self.toc = time.time()
    # update total run time
    self.time_stats['run_time'] = time.time() - self.start_time
    # update total step time
    self.time_stats['step_time'] = self.toc - self.tic
    # update time history
    if 'step_time_history' not in self.time_stats.keys():
      self.time_stats['step_time_history'] = []
    self.time_stats['step_time_history'].append(self.time_stats['step_time'])
    if self.time_stats['step_time_history'] > self.stats_history_length:
      self.time_stats['step_time_history'].pop(0)
    # update time ave
    self.time_stats['step_time_ave'] = (np.sum(np.array(self.time_stats['step_time_history']))
                   / (len(self.time_stats['step_time_history']) * self.config.batch_size * len(self.config.gpus))
    # start timer
    self.tick = time.time()

  def print_train_info(self, loss_stats, time_stats, queue_stats, step):

    print_string = ''
    print(
    print_string  = (colored('time per sample ', 'green') + str(round(time_per_sample, 3)) + '\n')
    print_string += (colored('loss ', 'blue') + str(round(loss, 3)) + '\n')
    print_string += (colored('ave loss ', 'blue') + str(round(ave_loss, 3)) + '\n')
    print_string += (colored('step ', 'yellow') + str(step) + '\n')
    for k in queue_stats.keys():
      print_string += (colored(k + ' ', 'magenta') + str(queue_stats[k]) + '\n')
    os.system('clear')
    print("TRAIN INFO")
    print(print_string)

  def start_session(self):
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess

