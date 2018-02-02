

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
    self.compression_mapping_boundary = tf.make_template('compression_mapping_boundary', net.compression_mapping_boundary)
    if self.seq_length >= 2:
      self.compression_mapping          = tf.make_template('compression_mapping', net.compression_mapping)
    self.decoder_state                = tf.make_template('decoder_state', net.decoder_state)

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
        with tf.device('/gpu:%d' % self.gpus[i]):
          # make input state and boundary
          self.add_tensor('state' + gpu_str, tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [self.DxQy.Q]))
          self.add_tensor('boundary' + gpu_str, tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [4]))
          if i == 0:
            with tf.device('/cpu:0'):
              tf.summary.image('state', self.DxQy.lattice_to_norm(self.in_tensors['state' + gpu_str]))
              tf.summary.image('boundary', self.in_tensors['boundary' + gpu_str][...,0:1])
          # make seq of output states
          for j in xrange(self.seq_length):
            self.add_tensor('true_state_' + str(j) + gpu_str, tf.placeholder(tf.float32, (2 + self.DxQy.dims) * [None]))
            if i == 0:
              with tf.device('/cpu:0'):
                tf.summary.image('true_state_' + str(j), self.DxQy.lattice_to_norm(self.in_tensors['true_state_' + str(j) + gpu_str]))
      
          ###### Unroll Graph ######
          # encode
          self.encoder_state(self, self.config, in_name="state" + gpu_str, out_name="cstate_0" + gpu_str)
          self.encoder_boundary(self, self.config, in_name="boundary" + gpu_str, out_name="cboundary" + gpu_str)
      
          # unroll all
          for j in xrange(self.seq_length):
            # apply boundary
            self.compression_mapping_boundary(self, self.config, in_cstate_name="cstate_" + str(j) + gpu_str, 
                                                    in_cboundary_name="cboundary" + gpu_str, 
                                                    out_name="cstate_" + str(j) + gpu_str)
            # decode and add to list
            self.decoder_state(self, self.config, in_name="cstate_" + str(j) + gpu_str, out_name="pred_state_" + str(j) + gpu_str)
   
            if self.seq_length >= 2: 
              # compression mapping
              self.compression_mapping(self, self.config, in_name="cstate_" + str(j) + gpu_str, out_name="cstate_" + str(j+1) + gpu_str)
              self.out_tensors['cboundary' + gpu_str] = self.out_tensors['cboundary' + gpu_str][:,6:-6,6:-6] # TODO fix this
   
            if i == 0:
              with tf.device('/cpu:0'):
                # make image summary
                tf.summary.image('predicted_state_vel_' + str(j), self.DxQy.lattice_to_norm(self.out_tensors['pred_state_' + str(j) + gpu_str]))
      
          ###### Loss Operation ######
          # define mse loss
          self.out_tensors["loss" + gpu_str] = 0.0
          for j in xrange(self.seq_length):
            #factor = np.power(2.0, j)
            # normalize loss to 256 by 256 state for now
            factor = ((256.0*256.0)
                    / tf.cast(tf.reduce_prod(tf.shape(self.out_tensors['true_state_' + str(j) + gpu_str])[1:3]), dtype=tf.float32))
            self.mse(true_name='true_state_' + str(j) + gpu_str,
                     pred_name='pred_state_' + str(j) + gpu_str,
                     loss_name='loss_mse_' + str(j) + gpu_str, factor=factor)
            self.gradient_difference(true_name='true_state_' + str(j) + gpu_str,
                     pred_name='pred_state_' + str(j) + gpu_str,
                     loss_name='loss_gd_' + str(j) + gpu_str, factor=factor)

            # add up losses
            self.out_tensors['loss' + gpu_str] +=       self.out_tensors['loss_mse_' + str(j) + gpu_str] 
            self.out_tensors['loss' + gpu_str] += 0.2 * self.out_tensors['loss_gd_' + str(j) + gpu_str] 

          # factor out batch size and num gpus and seq length
          self.out_tensors['loss' + gpu_str] = self.out_tensors['loss' + gpu_str]/(self.seq_length * self.config.batch_size * len(self.config.gpus))
 
          ###### Grad Operation ######
          if i == 0:
            all_params = tf.trainable_variables()
          self.out_tensors['grads' + gpu_str] = tf.gradients(self.out_tensors['loss' + gpu_str], all_params)

      # store up the loss and gradients on gpu:0
      with tf.device('/gpu:0'):
        for i in range(1, len(self.gpus)):
          self.out_tensors['loss_gpu_' + str(self.gpus[0])] += self.out_tensors['loss_gpu_' + str(self.gpus[i])]
          for j in range(len(self.out_tensors['grads_gpu_' + str(self.gpus[0])])):
            self.out_tensors['grads_gpu_' + str(self.gpus[0])][j] += self.out_tensors['grads_gpu_' + str(self.gpus[i])][j]
      tf.summary.scalar('total_loss', self.out_tensors['loss_gpu_' + str(self.gpus[0])])

      ###### Train Operation ######
      self.optimizer = Optimizer(self.config)
      self.out_tensors['train_op'] = self.optimizer.train_op(all_params, self.out_tensors['grads_gpu_' + str(self.gpus[0])], self.out_tensors['global_step'])
  
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
    return shape_converters

  def train(self, dataset):
  
    # steps per print (hard set for now)
    steps_per_print = 10
    ave_loss_length = 300
 
    # start timer
    t = time.time()

    prev_losses = []
    while True: 
      # get batch of data
      feed_dict = dataset.minibatch()

      # perform train step
      tf_feed_dict = {}
      for name in feed_dict.keys():
        tf_feed_dict[self.in_tensors[name]] = feed_dict[name]
      _, loss = self.sess.run([self.out_tensors['train_op'], self.out_tensors['loss_gpu_' + str(self.gpus[0])]], feed_dict=tf_feed_dict)

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
      self.add_tensor('cboundary', tf.placeholder(tf.float32, (1 + self.DxQy.dims) * [None] + [2*self.config.filter_size_compression]))
  
      ###### Unroll Graph ######
      # encoders
      self.encoder_state(self, self.config, in_name="state", out_name="cstate_from_state")
      self.encoder_boundary(self, self.config, in_name="boundary", out_name="cboundary_from_boundary")
  
      # compression mapping
      self.compression_mapping_boundary(self, self.config, in_cstate_name="cstate", 
                                              in_cboundary_name="cboundary", 
                                              out_name="cstate_from_cstate")
      self.compression_mapping(self, self.config, in_name="cstate_from_cstate", out_name="cstate_from_cstate")
  
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
    #self.out_tensors[out_name] = self.out_tensors[a_name] + self.out_tensors[b_name]

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

  def mse(self, true_name, pred_name, loss_name, factor):
    self.out_tensors[loss_name] = factor * tf.nn.l2_loss(self.in_tensors[ true_name] 
                                                       - self.out_tensors[pred_name])
    #tf.summary.scalar('loss_' + true_name + "_and_" + pred_name, self.out_tensors[loss_name])

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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess




