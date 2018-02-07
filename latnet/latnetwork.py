

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
    self.gan_loss = config.gan_loss
    gpus = config.gpus.split(',')
    self.gpus = map(int, gpus)

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
    self.discriminator                = tf.make_template('discriminator', net.discriminator)

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
      self.add_tensor('global_step', tf.get_variable('global_step', [], 
                       initializer=tf.constant_initializer(0), trainable=False))

      for i in xrange(len(self.gpus)):
        gpu_str = '_gpu_' + str(self.gpus[i])
        seq_str = lambda x: '_' + str(x) + gpu_str
        with tf.device('/gpu:%d' % self.gpus[i]):
          # make input state and boundary
          self.add_tensor('state' + gpu_str, tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [self.DxQy.Q]))
          self.add_tensor('boundary' + gpu_str, tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [4]))
          self.add_tensor('mask' + gpu_str, tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [1]))
          if i == 0:
            with tf.device('/cpu:0'):
              tf.summary.image('state', self.DxQy.lattice_to_norm(self.in_tensors['state' + gpu_str]))
              tf.summary.image('boundary', self.in_tensors['boundary' + gpu_str][...,0:1])
              tf.summary.image('mask', self.in_tensors['mask' + gpu_str])
          # make seq of output states
          for j in xrange(self.seq_length):
            self.add_tensor('true_state' + seq_str(j), tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [self.DxQy.Q]))
            if i == 0:
              with tf.device('/cpu:0'):
                tf.summary.image('true_state_' + str(j), self.DxQy.lattice_to_norm(self.in_tensors['true_state' + seq_str(j)]))
      
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
                self.out_tensors['pred_norm' + seq_str(j)] = self.DxQy.lattice_to_norm(self.out_tensors['pred_state' + seq_str(j)])
                tf.summary.image('pred_vel_' + str(j), self.out_tensors['pred_norm' + seq_str(j)])
 
          ### discriminator of gan ###
          if self.gan_loss is not None:
            seq_pred_state_names = []
            seq_true_state_names = []
            for j in xrange(self.seq_length):
              seq_pred_state_names.append("pred_state" + seq_str(j))
              seq_true_state_names.append("true_state" + seq_str(j))
            self.discriminator(self, self.config, 
                               in_boundary_name=None,
                               in_state_name=None,
                               in_seq_state_names=seq_pred_state_names,
                               out_name='D_pred' + gpu_str)
            self.discriminator(self, self.config, 
                               in_boundary_name=None,
                               in_state_name=None,
                               in_seq_state_names=seq_true_state_names,
                               out_name='D_true' + gpu_str)
      
          ###### Loss Operation ######
          num_samples = (self.seq_length * self.config.batch_size * len(self.config.gpus))

          ### MSE loss ###
          self.out_tensors["loss_mse" + gpu_str] = 0.0
          for j in xrange(self.seq_length):
            # normalize loss to 256 by 256 state for now
            pred_shape = tf.shape(self.out_tensors['true_state' + seq_str(j)])
            num_cells = tf.cast(tf.reduce_prod(pred_shape[1:3]), dtype=tf.float32)
            mse_factor = ((256.0*256.0) / num_cells)
            self.mse(true_name='true_state' + seq_str(j),
                     pred_name='pred_state' + seq_str(j),
                     loss_name='loss_mse' + seq_str(j), 
                     factor=mse_factor)

            # add up losses
            self.out_tensors['loss_mse' + gpu_str] += self.out_tensors['loss_mse' + seq_str(j)] 

          # factor out batch size and num gpus and seq length
          self.out_tensors['loss_mse' + gpu_str] = self.out_tensors['loss_mse' + gpu_str]/(num_samples)

          ### GAN loss ###
          if self.gan_loss is not None:
            self.out_tensors['gen_loss' + gpu_str] = -tf.reduce_mean(tf.log(self.out_tensors['D_pred' + gpu_str]))
            self.out_tensors['disc_loss' + gpu_str] = -tf.reduce_mean(tf.log(self.out_tensors['D_true' + gpu_str])
                                                              + tf.log(1.0 - self.out_tensors['D_pred' + gpu_str]))

          #### add all losses ###
          #self.out_tensors['loss' + gpu_str] =  self.out_tensors['loss_mse' + gpu_str]
          self.out_tensors['loss' + gpu_str] =  0.0
          if self.gan_loss is not None:
            self.out_tensors['loss' + gpu_str] += self.out_tensors['gen_loss' + gpu_str]
 
          ###### Grad Operation ######
          if i == 0:
            all_params = tf.trainable_variables()
            if self.gan_loss is not None:
              gen_params = [v for i, v in enumerate(all_params) if "discriminator" not in v.name[:v.name.index(':')]]
              disc_params = [v for i, v in enumerate(all_params) if "discriminator" in v.name[:v.name.index(':')]]

          if self.gan_loss is None:
            self.out_tensors['grads' + gpu_str] = tf.gradients(self.out_tensors['loss' + gpu_str], all_params)
          else:
            self.out_tensors['gen_grads' + gpu_str] = tf.gradients(self.out_tensors['loss' + gpu_str], gen_params)
            self.out_tensors['disc_grads' + gpu_str] = tf.gradients(self.out_tensors['disc_loss' + gpu_str], disc_params)

      ###### Round up losses and Gradients on gpu:0 ######
      with tf.device('/gpu:%d' % self.gpus[0]):
        gpu_str = lambda x: '_gpu_' + str(self.gpus[x])
        for i in range(1, len(self.gpus)):
          # losses
          self.out_tensors['loss_mse' + gpu_str(0)] += self.out_tensors['loss_mse' + gpu_str(i)]
          if self.gan_loss is not None:
            self.out_tensors['gen_loss' + gpu_str(0)] += self.out_tensors['gen_loss' + gpu_str(i)]
            self.out_tensors['disc_loss' + gpu_str(0)] += self.out_tensors['disc_loss' + gpu_str(i)]
          # gradients 
          if self.gan_loss is None:
            for j in range(len(self.out_tensors['grads' + gpu_str(0)])):
                self.out_tensors['grads' + gpu_str(0)][j] += self.out_tensors['grads' + gpu_str(i)][j]
          else:
            for j in range(len(self.out_tensors['gen_grads' + gpu_str(0)])):
              self.out_tensors['gen_grads' + gpu_str(0)][j] += self.out_tensors['gen_grads' + gpu_str(i)][j]
            for j in range(len(self.out_tensors['disc_grads' + gpu_str(0)])):
              self.out_tensors['disc_grads' + gpu_str(0)][j] += self.out_tensors['disc_grads' + gpu_str(i)][j]

      ### add loss summary ### 
      tf.summary.scalar('loss_mse', self.out_tensors['loss_mse' + gpu_str(0)])
      if self.gan_loss is not None:
        tf.summary.scalar('gen_loss', self.out_tensors['gen_loss' + gpu_str(0)])
        tf.summary.scalar('disc_loss', self.out_tensors['disc_loss' + gpu_str(0)])

      ###### Train Operation ######
      if self.gan_loss is None:
        self.optimizer = Optimizer(self.config)
        self.out_tensors['train_op'] = self.optimizer.train_op(all_params, 
                                               self.out_tensors['grads' + gpu_str(0)], 
                                               self.out_tensors['global_step'])
      else:
        self.gen_optimizer = Optimizer(self.config)
        self.disc_optimizer = Optimizer(self.config)
        self.out_tensors['gen_train_op'] = self.gen_optimizer.train_op(gen_params, 
                                                   self.out_tensors['gen_grads' + gpu_str(0)], 
                                                   self.out_tensors['global_step'])
        self.out_tensors['disc_train_op'] = self.disc_optimizer.train_op(disc_params, 
                                                   self.out_tensors['disc_grads' + gpu_str(0)], 
                                                   self.out_tensors['global_step'])
  
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
  
    # steps per print (hard set for now)
    steps_per_print = 1
    ave_loss_length = 300
 
    # start timer
    t = time.time()

    prev_losses = []
    while True: 
      # get batch of data
      feed_dict = dataset.minibatch()
      tf_feed_dict = {}
      for name in feed_dict.keys():
        tf_feed_dict[self.in_tensors[name]] = feed_dict[name]

      # perform optimization step
      if self.gan_loss is None:
        output = [self.out_tensors['train_op'], 
                  self.out_tensors['loss_gpu_' + str(self.gpus[0])]]
        _, loss = self.sess.run(output, feed_dict=tf_feed_dict)
      else:
        #output = [self.out_tensors['gen_train_op'], 
        #          self.out_tensors['disc_train_op'], 
        #          self.out_tensors['gen_loss_gpu_' + str(self.gpus[0])],
        #          self.out_tensors['disc_loss_gpu_' + str(self.gpus[0])]]
        output = [self.out_tensors['gen_train_op'], 
                  self.out_tensors['gen_loss_gpu_' + str(self.gpus[0])],
                  self.out_tensors['disc_loss_gpu_' + str(self.gpus[0])]]
        #_, _, gen_loss, disc_loss = self.sess.run(output, feed_dict=tf_feed_dict)
        #for i in xrange(10):
        #  _ = self.sess.run(self.out_tensors['gen_train_op'], feed_dict=tf_feed_dict)
        _, gen_loss, disc_loss = self.sess.run(output, feed_dict=tf_feed_dict)
        loss = gen_loss
        #loss = disc_loss

      # calc ave loss
      prev_losses.append(loss)
      if len(prev_losses) > ave_loss_length:
        prev_losses.pop(0)
 
      # print required data and save
      step = self.sess.run(self.out_tensors['global_step'])
      if step % steps_per_print == 0:
        queue_stats = dataset.queue_stats()
        elapsed = time.time() - t
        t = time.time()
        time_per_sample = elapsed/(steps_per_print * self.config.batch_size * len(self.config.gpus))
        ave_loss = np.sum(np.array(prev_losses))/len(prev_losses)
        self.saver.save_summary(self.sess, tf_feed_dict, int(self.sess.run(self.out_tensors['global_step'])))
        self.print_train_info(time_per_sample, loss, ave_loss, step, queue_stats)

      if step % self.config.save_network_freq == 0:
        self.saver.save_summary(self.sess, tf_feed_dict, int(self.sess.run(self.out_tensors['global_step'])))
        self.saver.save_checkpoint(self.sess, int(self.sess.run(self.out_tensors['global_step'])))

  def print_train_info(self, time_per_sample, loss, ave_loss, step, queue_stats):
    print_string  = (colored('time per sample ', 'green') + str(round(time_per_sample, 3)) + '\n')
    print_string += (colored('loss ', 'blue') + str(round(loss, 3)) + '\n')
    print_string += (colored('ave loss ', 'blue') + str(round(ave_loss, 3)) + '\n')
    print_string += (colored('step ', 'yellow') + str(step) + '\n')
    for k in queue_stats.keys():
      print_string += (colored(k + ' ', 'magenta') + str(queue_stats[k]) + '\n')
    os.system('clear')
    print("TRAIN INFO")
    print(print_string)

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
    state_encoder    = lambda x: self.sess.run(self.out_tensors['cstate_from_state'], 
                                 feed_dict={self.in_tensors['state']:x})
    boundary_encoder = lambda x: self.sess.run(self.out_tensors['cboundary_from_boundary'], 
                                 feed_dict={self.in_tensors['boundary']:x})
    cmapping         = lambda x, y: self.sess.run(self.out_tensors['cstate_from_cstate'], 
                                 feed_dict={self.in_tensors['cstate']:x,
                                            self.in_tensors['cboundary']:y})
    decoder           = lambda x: self.sess.run([self.out_tensors['vel_from_cstate'], 
                                                 self.out_tensors['rho_from_cstate']], 
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
                                                        name=weight_name, nonlinearity=nonlinearity)

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

  def mse(self, true_name, pred_name, loss_name, factor):
    self.out_tensors[loss_name] = factor * tf.nn.l2_loss(self.in_tensors[ true_name] 
                                                       - self.out_tensors[pred_name])
    #tf.summary.scalar('loss_' + true_name + "_and_" + pred_name, self.out_tensors[loss_name])
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
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess


  """
  def gradient_difference(self, true_name, pred_name, loss_name, factor):
    true = self.in_tensors[ true_name] 
    generated = self.out_tensors[ pred_name] 

    # seen in here https://arxiv.org/abs/1511.05440
    if len(true.get_shape()) == 4:
      true_x_shifted_right = true[:,1:,:,:]
      true_x_shifted_left = true[:,:-1,:,:]
      true_x_gradient = tf.abs(true_x_shifted_right - true_x_shifted_left)
  
      generated_x_shifted_right = generated[:,1:,:,:]
      generated_x_shifted_left = generated[:,:-1,:,:]
      generated_x_gradient = tf.abs(generated_x_shifted_right - generated_x_shifted_left)
  
      loss_x_gradient = tf.nn.l2_loss(true_x_gradient - generated_x_gradient)
  
      true_y_shifted_right = true[:,:,1:,:]
      true_y_shifted_left = true[:,:,:-1,:]
      true_y_gradient = tf.abs(true_y_shifted_right - true_y_shifted_left)
  
      generated_y_shifted_right = generated[:,:,1:,:]
      generated_y_shifted_left = generated[:,:,:-1,:]
      generated_y_gradient = tf.abs(generated_y_shifted_right - generated_y_shifted_left)
      
      loss_y_gradient = tf.nn.l2_loss(true_y_gradient - generated_y_gradient)
  
      loss = loss_x_gradient + loss_y_gradient
  
    else:
      true_x_shifted_right = true[:,1:,:,:,:]
      true_x_shifted_left = true[:,:-1,:,:,:]
      true_x_gradient = tf.abs(true_x_shifted_right - true_x_shifted_left)
  
      generated_x_shifted_right = generated[:,1:,:,:,:]
      generated_x_shifted_left = generated[:,:-1,:,:,:]
      generated_x_gradient = tf.abs(generated_x_shifted_right - generated_x_shifted_left)
  
      loss_x_gradient = tf.nn.l2_loss(true_x_gradient - generated_x_gradient)
  
      true_y_shifted_right = true[:,:,1:,:,:]
      true_y_shifted_left = true[:,:,:-1,:,:]
      true_y_gradient = tf.abs(true_y_shifted_right - true_y_shifted_left)
  
      generated_y_shifted_right = generated[:,:,1:,:,:]
      generated_y_shifted_left = generated[:,:,:-1,:,:]
      generated_y_gradient = tf.abs(generated_y_shifted_right - generated_y_shifted_left)
      
      loss_y_gradient = tf.nn.l2_loss(true_y_gradient - generated_y_gradient)
  
      true_z_shifted_right = true[:,:,:,1:,:]
      true_z_shifted_left = true[:,:,:,:-1,:]
      true_z_gradient = tf.abs(true_z_shifted_right - true_z_shifted_left)
  
      generated_z_shifted_right = generated[:,:,:,1:,:]
      generated_z_shifted_left = generated[:,:,:,:-1,:]
      generated_z_gradient = tf.abs(generated_z_shifted_right - generated_z_shifted_left)
      
      loss_z_gradient = tf.nn.l2_loss(true_z_gradient - generated_z_gradient)
  
      loss = loss_x_gradient + loss_y_gradient + loss_z_gradient
  
    self.out_tensors[loss_name] = factor * loss
    return loss
  """



