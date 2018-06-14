

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
  #network_name = 'advanced_network'

  def __init__(self, config):
    # in and out tensors
    self.in_tensors = {}
    self.out_tensors = {}
    self.in_pad_tensors = {}
    self.out_pad_tensors = {}

    # shape converter from in_tensor to out_tensor
    self.shape_converters = {}

    # needed configs
    self.config = config # TODO remove this when config is corrected
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.network_dir  = config.latnet_network_dir
    self.seq_length = config.seq_length
    self.train_autoencoder = config.train_autoencoder
    self.gan = config.gan
    self.train_iters = config.train_iters
    gpus = config.gpus.split(',')
    self.gpus = map(int, gpus)
    self.loss_stats = {}
    self.time_stats = {}
    self.start_time = time.time()
    self.tic = time.time()
    self.toc = time.time()
    self.stats_history_length = 300

    # 
    self.make_network_templates()

  @classmethod
  def add_options(cls, group):
    pass

  def make_network_templates(self):
    # piecese of network
    self._encoder_state                = tf.make_template('encoder_state', self.encoder_state)
    self._encoder_boundary             = tf.make_template('encoder_boundary', self.encoder_boundary)
    self._compression_mapping          = tf.make_template('compression_mapping', self.compression_mapping)
    self._decoder_state                = tf.make_template('decoder_state', self.decoder_state)
    if self.gan:
      self._discriminator_conditional    = tf.make_template('discriminator_conditional', self.discriminator_conditional)
      self._discriminator_unconditional  = tf.make_template('discriminator_unconditional', self.discriminator_unconditional)

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
          self.add_tensor('boundary' + gpu_str, (1 + self.DxQy.dims) * [None] + [6])
          self.add_tensor('boundary_small' + gpu_str, (1 + self.DxQy.dims) * [None] + [6])
          self.add_phase() 
          if i == 0:
            with tf.device('/cpu:0'):
              self.lattice_summary(in_name='state' + gpu_str, summary_name='true')
              self.boundary_summary(in_name='boundary' + gpu_str, summary_name='boundary')
          # make seq of output states
          for j in xrange(self.seq_length):
            self.add_tensor('true_state' + seq_str(j), (1 + self.DxQy.dims) * [None] + [self.DxQy.Q])
            self.out_tensors["true_state" + seq_str(j)] = self.out_tensors["true_state" + seq_str(j)]
            if i == 0:
              with tf.device('/cpu:0'):
                self.lattice_summary(in_name='true_state' + seq_str(j), summary_name='true_' + str(j))
      
          ###### Unroll Graph ######
          ### encode ###
          self._encoder_state(in_name="state" + gpu_str, 
                              out_name="cstate" + seq_str(0))
          self._encoder_boundary(in_name="boundary" + gpu_str, 
                                out_name="cboundary" + gpu_str)
      
          ### unroll on the compressed state ###
          for j in xrange(self.seq_length):
            # compression mapping
            self.add_shape_converter("cstate" + seq_str(j))
            if j == 0:
              self._compression_mapping(in_cstate_name="cstate" + seq_str(j),
                                        in_cboundary_name="cboundary" + gpu_str,
                                        out_name="cstate" + seq_str(j+1), 
                                        start_apply_boundary=True)
            if (j < self.seq_length - 1) and (j != 0):
              self._compression_mapping(in_cstate_name="cstate" + seq_str(j),
                                        in_cboundary_name="cboundary" + gpu_str,
                                        out_name="cstate" + seq_str(j+1))

          ### decode all compressed states ###
          for j in xrange(self.seq_length):
            self.match_trim_tensor(in_name="cstate" + seq_str(j), 
                                   match_name="cstate" + seq_str(self.seq_length-1), 
                                   out_name="cstate" + seq_str(j))
            self._decoder_state(in_cstate_name="cstate" + seq_str(j), 
                                in_cboundary_name="cboundary" + gpu_str, 
                                out_name="pred_state" + seq_str(j))


            ### encode seq state  
            if not self.train_autoencoder:
              self._encoder_state(in_name="true_state" + seq_str(j),
                                  out_name="true_cstate" + seq_str(j))

            if i == 0:
              with tf.device('/cpu:0'):
                # make image summary
                self.lattice_summary(in_name='pred_state' + seq_str(j), summary_name='pred_' + str(j))

 
          ### unconditional discriminator of gan ###
          if self.gan:
            seq_pred_state_names = []
            seq_true_state_names = []
            for j in range(1, self.seq_length):
              seq_pred_state_names.append("pred_state" + seq_str(j))
              seq_true_state_names.append("true_state" + seq_str(j))
            self._discriminator_unconditional(in_seq_state_names=seq_pred_state_names,
                                              out_layer='D_un_layer_pred' + gpu_str,
                                              out_class='D_un_class_pred' + gpu_str)
            self._discriminator_unconditional(in_seq_state_names=seq_true_state_names,
                                              out_layer='D_un_layer_true' + gpu_str,
                                              out_class='D_un_class_true' + gpu_str)
      
          ### conditional discriminator of gan ###
          if self.gan:
            seq_pred_state_names = []
            seq_true_state_names = []
            for j in range(1, self.seq_length):
              seq_pred_state_names.append("pred_state" + seq_str(j))
              seq_true_state_names.append("true_state" + seq_str(j))
            self._discriminator_conditional(in_boundary_name='boundary_small' + gpu_str,
                                            in_state_name='true_state' + seq_str(0),
                                            in_seq_state_names=seq_pred_state_names,
                                            out_name='D_con_class_pred' + gpu_str)
            self._discriminator_conditional(in_boundary_name='boundary_small' + gpu_str,
                                            in_state_name='true_state' + seq_str(0),
                                            in_seq_state_names=seq_true_state_names,
                                            out_name='D_con_class_true' + gpu_str)
      

          ###### Loss Operation ######
          num_samples = (self.seq_length * self.config.batch_size * len(self.config.gpus))

          ### L2 loss ###
          if not self.gan:
            self.out_tensors["loss_l2" + gpu_str] = 0.0

            # factor to account for how much the sim is changing
            seq_factor = tf.nn.l2_loss(self.out_tensors['true_state' + seq_str(0)]
                                     - self.out_tensors['true_state' + seq_str(self.seq_length-1)])
            if self.train_autoencoder:
              for j in range(0, self.seq_length):
                # normalize loss to make comparable for diffrent input sizes
                # TODO remove 100.0 (only in to make comparable to previous code)
                #l2_factor = 1.0*(256.0*256.0)/self.num_lattice_cells('pred_state' + seq_str(j), return_float=True)
                l2_factor = 1.0*(256.0*256.0)/self.num_lattice_cells('pred_state' + seq_str(j), return_float=True)
                self.l2_loss(true_name='true_state' + seq_str(j),
                             pred_name='pred_state' + seq_str(j),
                             loss_name='loss_l2' + seq_str(j),
                             factor=l2_factor/1.0)
                             #factor=l2_factor/num_samples)
                # add up losses
                self.out_tensors['loss_l2' + gpu_str] += self.out_tensors['loss_l2' + seq_str(j)] 
            else:
              for j in range(1, self.seq_length):
                # normalize loss to make comparable for diffrent input sizes
                #l2_factor = 1.0*(256.0*256.0)/self.num_lattice_cells('pred_state' + seq_str(j), return_float=True)
                l2_factor = (256.0*256.0)/(len(self.gpus)*self.config.batch_size*self.num_lattice_cells('pred_state' + seq_str(j), return_float=True))
                self.l2_loss(true_name='true_cstate' + seq_str(j),
                             pred_name='cstate' + seq_str(j),
                             loss_name='loss_l2' + seq_str(j),
                             factor=l2_factor/1.0)
                             #factor=l2_factor/num_samples)
                # add up losses
                self.out_tensors['loss_l2' + gpu_str] += self.out_tensors['loss_l2' + seq_str(j)] 
 


          ### L1 loss ###
          if self.gan:
            self.out_tensors["loss_l1" + gpu_str] = 0.0
            for j in range(1, self.seq_length):
              l1_factor = self.config.l1_factor
              self.l1_loss(true_name='true_state' + seq_str(j),
                           pred_name='pred_state' + seq_str(j),
                           loss_name='loss_l1' + seq_str(j),
                           factor=l1_factor/len(self.gpus))
              # add up losses
              self.out_tensors['loss_l1' + gpu_str] += self.out_tensors['loss_l1' + seq_str(j)] 
          ### Unconditional GAN loss ###
          if self.gan:
            # normal gan losses
            self.gen_loss(class_name='D_un_class_pred' + gpu_str,
                          loss_name='loss_gen_un_class' + gpu_str,
                          factor=1.0/len(self.gpus))
            self.disc_loss(true_class_name='D_un_class_true' + gpu_str,
                           pred_class_name='D_un_class_pred' + gpu_str,
                           loss_name='loss_disc_un_class' + gpu_str,
                           factor=1.0/len(self.gpus))
            # layer loss as seen in tempoGAN: A Temporally Coherent, Volumetric GAN for
            # Super-resolution Fluid Flow
            l2_layer_factor = 1e-5 # from paper TODO make config param
            self.l2_loss(true_name='D_un_layer_true' + gpu_str,
                         pred_name='D_un_layer_pred' + gpu_str,
                         loss_name='loss_layer_l2' + gpu_str,
                         factor=l2_layer_factor/len(self.gpus))

          ### Conditional GAN loss ###
          if self.gan:
            # normal gan losses
            self.gen_loss(class_name='D_con_class_pred' + gpu_str,
                          loss_name='loss_gen_con_class' + gpu_str,
                          factor=1.0/len(self.gpus))
            self.disc_loss(true_class_name='D_con_class_true' + gpu_str,
                           pred_class_name='D_con_class_pred' + gpu_str,
                           loss_name='loss_disc_con_class' + gpu_str,
                           factor=1.0/len(self.gpus))

          #### add all losses together ###
          self.out_tensors['loss_gen' + gpu_str] =  0.0
          if not self.gan:
            self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_l2' + gpu_str]
          if self.gan:
            self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_l1' + gpu_str]
            self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_gen_un_class' + gpu_str]
            self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_gen_con_class' + gpu_str]
            self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_layer_l2' + gpu_str]
            self.out_tensors['loss_disc' + gpu_str] = self.out_tensors['loss_disc_un_class' + gpu_str]
            self.out_tensors['loss_disc' + gpu_str] += self.out_tensors['loss_disc_con_class' + gpu_str]
 
          ###### Grad Operation ######
          if i == 0:
            all_params = tf.trainable_variables()
            gen_params = [v for i, v in enumerate(all_params) if "discriminator" not in v.name[:v.name.index(':')]]
            disc_params = [v for i, v in enumerate(all_params) if "discriminator" in v.name[:v.name.index(':')]]
            if not self.train_autoencoder:
              gen_params = [v for i, v in enumerate(gen_params) if "compression_mapping" in v.name[:v.name.index(':')]]
          self.out_tensors['gen_grads' + gpu_str] = tf.gradients(self.out_tensors['loss_gen' + gpu_str], gen_params)
          if self.gan:
            self.out_tensors['disc_grads' + gpu_str] = tf.gradients(self.out_tensors['loss_disc' + gpu_str], disc_params)

      ###### Round up losses and Gradients on gpu:0 ######
      ### Round up losses ###
      with tf.device('/gpu:%d' % self.gpus[0]):
        gpu_str = lambda x: '_gpu_' + str(self.gpus[x])
        for i in range(1, len(self.gpus)):
          self.out_tensors['loss_gen' + gpu_str(0)] += self.out_tensors['loss_gen' + gpu_str(i)]
          if not self.gan:
            self.out_tensors['loss_l2' + gpu_str(0)] += self.out_tensors['loss_l2' + gpu_str(i)]
          if self.gan:
            self.out_tensors['loss_l1' + gpu_str(0)] += self.out_tensors['loss_l1' + gpu_str(i)]
            self.out_tensors['loss_gen_un_class' + gpu_str(0)] += self.out_tensors['loss_gen_un_class' + gpu_str(i)]
            self.out_tensors['loss_gen_con_class' + gpu_str(0)] += self.out_tensors['loss_gen_con_class' + gpu_str(i)]
            self.out_tensors['loss_layer_l2' + gpu_str(0)] += self.out_tensors['loss_layer_l2' + gpu_str(i)]
            self.out_tensors['loss_disc_un_class' + gpu_str(0)] += self.out_tensors['loss_disc_un_class' + gpu_str(i)]
            self.out_tensors['loss_disc_con_class' + gpu_str(0)] += self.out_tensors['loss_disc_con_class' + gpu_str(i)]
            self.out_tensors['loss_disc' + gpu_str(0)] += self.out_tensors['loss_disc' + gpu_str(i)]
      self.rename_tensor(old_name='loss_gen' + gpu_str(0), new_name='loss_gen')
      if not self.gan:
        self.rename_tensor(old_name='loss_l2' + gpu_str(0), new_name='loss_l2')
      if self.gan:
        self.rename_tensor(old_name='loss_l1' + gpu_str(0), new_name='loss_l1')
        self.rename_tensor(old_name='loss_gen_un_class' + gpu_str(0), new_name='loss_gen_un_class')
        self.rename_tensor(old_name='loss_gen_con_class' + gpu_str(0), new_name='loss_gen_con_class')
        self.rename_tensor(old_name='loss_layer_l2' + gpu_str(0), new_name='loss_layer_l2')
        self.rename_tensor(old_name='loss_disc_un_class' + gpu_str(0), new_name='loss_disc_un_class')
        self.rename_tensor(old_name='loss_disc_con_class' + gpu_str(0), new_name='loss_disc_con_class')
        self.rename_tensor(old_name='loss_disc' + gpu_str(0), new_name='loss_disc')

      ### Round up gradients ###
      with tf.device('/gpu:%d' % self.gpus[0]):
        gpu_str = lambda x: '_gpu_' + str(self.gpus[x])
        for i in range(1, len(self.gpus)):
          # gradients 
          for j in range(len(self.out_tensors['gen_grads' + gpu_str(0)])):
              if self.out_tensors['gen_grads' + gpu_str(0)][j] is not None:
                self.out_tensors['gen_grads' + gpu_str(0)][j] += self.out_tensors['gen_grads' + gpu_str(i)][j]
          if self.gan:
            for j in range(len(self.out_tensors['disc_grads' + gpu_str(0)])):
              if self.out_tensors['disc_grads' + gpu_str(0)][j] is not None:
                self.out_tensors['disc_grads' + gpu_str(0)][j] += self.out_tensors['disc_grads' + gpu_str(i)][j]

      ### add loss summary ###
      tf.summary.scalar('loss_gen', self.out_tensors['loss_gen'])
      tf.summary.scalar('total_loss', self.out_tensors['loss_gen'])
      if not self.gan:
        tf.summary.scalar('loss_l2', self.out_tensors['loss_l2'])
      if self.gan:
        tf.summary.scalar('loss_l1', self.out_tensors['loss_l1'])
        tf.summary.scalar('loss_gen_un_class', self.out_tensors['loss_gen_un_class'])
        tf.summary.scalar('loss_gen_con_class', self.out_tensors['loss_gen_con_class'])
        tf.summary.scalar('loss_layer_l2', self.out_tensors['loss_layer_l2'])
        tf.summary.scalar('loss_disc_un_class', self.out_tensors['loss_disc_un_class'])
        tf.summary.scalar('loss_disc_con_class', self.out_tensors['loss_disc_con_class'])
        tf.summary.scalar('loss_disc', self.out_tensors['loss_disc'])

      ###### Train Operation ######
      self.gen_optimizer = Optimizer(self.config, name='gen', optimizer_name='adam')
      self.disc_optimizer = Optimizer(self.config, name='disc', optimizer_name='adam')
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.out_tensors['gen_train_op'] = self.gen_optimizer.train_op(gen_params, 
                                               self.out_tensors['gen_grads' + gpu_str(0)], 
                                               self.out_tensors['gen_global_step'],
                                               mom1=self.config.beta1,
                                               other_update=update_ops)
      if self.gan:
        self.out_tensors['disc_train_op'] = self.disc_optimizer.train_op(disc_params, 
                                                     self.out_tensors['disc_grads' + gpu_str(0)], 
                                                     self.out_tensors['disc_global_step'],
                                                     mom1=self.config.beta1,
                                                     other_update=update_ops)
  
      ###### Start Session ######
      self.sess = self.start_session()
  
      ###### Saver Operation ######
      graph_def = self.sess.graph.as_graph_def(add_shapes=True)
      self.saver = NetworkSaver(self.config, self.network_name, graph_def)
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
        if not self.train_autoencoder:
          name = ("true_state_" + str(j) + gpu_str, "true_cstate_" + str(j) + gpu_str)
          shape_converters[name] = self.shape_converters[name]
    print(shape_converters.keys())
    return shape_converters

  def eval_unroll(self):

    # graph
    with tf.Graph().as_default():

      ###### Inputs to Graph ######
      # make input state and boundary
      self.add_tensor('state',     (1 + self.DxQy.dims) * [None] + [self.DxQy.Q])
      self.add_tensor('boundary',  (1 + self.DxQy.dims) * [None] + [6])
      self.add_tensor('cstate',    (1 + self.DxQy.dims) * [None] + [self.config.filter_size_compression])
      self.add_tensor('cboundary_first', (1 + self.DxQy.dims) * [None] + [2*self.config.filter_size_compression])
      self.add_tensor('cboundary', (1 + self.DxQy.dims) * [None] + [2*self.config.filter_size_compression])
      self.add_tensor('cboundary_decoder', (1 + self.DxQy.dims) * [None] + [2*self.config.filter_size_compression])
      self.add_tensor('cboundary_decoder', (1 + self.DxQy.dims) * [None] + [2*self.config.filter_size_compression])
      self.add_phase() 
  
      ###### Unroll Graph ######
      # encoders
      self._encoder_state(in_name="state", out_name="cstate_from_state")
      self._encoder_boundary(in_name="boundary", out_name="cboundary_from_boundary")
  
   
      # compression mapping first step
      self._compression_mapping(in_cstate_name="cstate", 
                                in_cboundary_name="cboundary_first",
                                out_name="cstate_from_cstate_first",
                                start_apply_boundary=True)
  
      # compression mapping
      self._compression_mapping(in_cstate_name="cstate", 
                                in_cboundary_name="cboundary",
                                out_name="cstate_from_cstate")
  
 
      # decoder
      self._decoder_state(in_cstate_name="cstate", in_cboundary_name="cboundary_decoder", out_name="state_from_cstate")
      self.out_tensors['vel_from_cstate'] = self.DxQy.lattice_to_vel(self.out_tensors['state_from_cstate'])
      self.out_tensors['rho_from_cstate'] = self.DxQy.lattice_to_rho(self.out_tensors['state_from_cstate'])

      ###### Start Session ######
      self.sess = self.start_session()
  
      ###### Saver Operation ######
      graph_def = self.sess.graph.as_graph_def(add_shapes=True)
      self.saver = NetworkSaver(self.config, self.network_name, graph_def)
      self.saver.load_checkpoint(self.sess)
  
    ###### Function Wrappers ######
    # network functions
    state_encoder    = lambda x: self.run('cstate_from_state', 
                                 feed_dict={'state':x})
    boundary_encoder = lambda x: self.run('cboundary_from_boundary', 
                                 feed_dict={'boundary':x})
    cmapping         = lambda x, y: self.run('cstate_from_cstate', 
                                 feed_dict={'cstate':x,
                                            'cboundary':y})
    cmapping_first   = lambda x, y: self.run('cstate_from_cstate_first', 
                                 feed_dict={'cstate':x,
                                            'cboundary_first':y})
    decoder_vel_rho  = lambda x, y: self.run(['vel_from_cstate', 
                                              'rho_from_cstate'], 
                                 feed_dict={'cstate':x,
                                            'cboundary_decoder':y})
    decoder_state    = lambda x, y: self.run('state_from_cstate',
                                 feed_dict={'cstate':x,
                                            'cboundary_decoder':y})

    # shape converters
    encoder_shape_converter = self.shape_converters['state', 'cstate_from_state']
    cmapping_shape_converter = self.shape_converters['cstate', 'cstate_from_cstate']
    decoder_shape_converter = self.shape_converters['cstate', 'state_from_cstate']

    return (state_encoder, boundary_encoder, cmapping, cmapping_first, decoder_vel_rho,
            decoder_state, encoder_shape_converter, cmapping_shape_converter, 
            decoder_shape_converter) # TODO This should probably be cleaned up

  def fc(self, in_name, out_name,
         hidden, weight_name='fc', 
         nonlinearity=None, flat=False):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.fc_layer(self.out_tensors[in_name],
                                              hidden, name=weight_name, 
                                              nonlinearity=nonlinearity, flat=flat)

  def simple_conv(self, in_name, out_name, kernel):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.simple_conv_2d(self.out_tensors[in_name], k=kernel)

    # add conv to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_conv(3, 1)

  def conv(self, in_name, out_name,
           kernel_size, stride, filter_size, 
           weight_name="conv", nonlinearity=None,
           normalize=None):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.conv_layer(self.out_tensors[in_name],
                                              kernel_size, stride, filter_size, 
                                              name=weight_name, nonlinearity=nonlinearity,
                                              phase=self.out_tensors['phase'],
                                              normalize=normalize)

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_conv_pad(self.out_pad_tensors[in_name], kernel_size, stride)

    # ensure zeros padding
    self.out_tensors[out_name] = nn.apply_pad(self.out_tensors[out_name], self.out_pad_tensors[out_name])

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

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_trans_conv_pad(self.out_pad_tensors[in_name], kernel_size, stride)

    # ensure zeros padding
    self.out_tensors[out_name] = nn.apply_pad(self.out_tensors[out_name], self.out_pad_tensors[out_name])

    # add conv to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_trans_conv(kernel_size, stride)

  def upsample(self, in_name, out_name):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.upsampleing_resize(self.out_tensors[in_name])

    # add conv to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_trans_conv(0, 2)

  def downsample(self, in_name, out_name, sampling='ave'):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.downsample(self.out_tensors[in_name], sampling=sampling)

    # add conv to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_conv(1, 2)

  def res_block(self, in_name, out_name,
                a_name = None,
                filter_size=16, 
                kernel_size=3, 
                nonlinearity=nn.concat_elu, 
                keep_p=1.0, stride=1, 
                gated=True, weight_name="resnet", 
                begin_nonlinearity=True, 
                normalize=None):
                #normalize='batch_norm'):

    # add res block to tensor computation
    self.out_tensors[out_name] = nn.res_block(self.out_tensors[in_name],
                                            filter_size=filter_size, 
                                            kernel_size=kernel_size, 
                                            nonlinearity=nonlinearity, 
                                            keep_p=keep_p, stride=stride, 
                                            gated=gated, name=weight_name, 
                                            begin_nonlinearity=begin_nonlinearity, 
                                            phase=self.out_tensors['phase'],
                                            normalize=normalize)

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_res_pad(self.out_pad_tensors[in_name], kernel_size, stride)

    # ensure zeros padding
    self.out_tensors[out_name] = nn.apply_pad(self.out_tensors[out_name], self.out_pad_tensors[out_name])

    # add res block to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_res_block(kernel_size, stride)

  def fast_res_block(self, in_name, out_name,
                     filter_size=16, 
                     filter_size_conv=4, 
                     kernel_size=7, 
                     nonlinearity=nn.concat_elu, 
                     weight_name="resnet", 
                     begin_nonlinearity=True):

    # add res block to tensor computation
    self.out_tensors[out_name] = nn.fast_res_block(self.out_tensors[in_name],
                                                   filter_size=filter_size, 
                                                   filter_size_conv=filter_size_conv, 
                                                   kernel_size=kernel_size, 
                                                   nonlinearity=nonlinearity, 
                                                   name=weight_name, 
                                                   begin_nonlinearity=begin_nonlinearity)

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_res_pad(self.out_pad_tensors[in_name], kernel_size, stride)

    # ensure zeros padding
    self.out_tensors[out_name] = nn.apply_pad(self.out_tensors[out_name], self.out_pad_tensors[out_name])

    # add res block to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_res_block(kernel_size, 1)


  def split_tensor(self, in_name,
                   out_names,
                   num_split, axis):

    # perform split on tensor
    splited_tensors  = tf.split(self.out_tensors[in_name],
                                num_split, axis)
    for i in xrange(len(out_names)):
      self.out_tensors[out_names[i]] = splited_tensors[i]

    # add to shape converters
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        for i in xrange(len(out_names)):
          self.shape_converters[name[0], out_names[i]] = copy(self.shape_converters[name])

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
        self.out_pad_tensors[out_name] = self.out_pad_tensors[in_name][:,trim:-trim, trim:-trim]
      elif len(self.out_tensors[in_name].get_shape()) == 5:
        self.out_tensors[out_name] = self.out_tensors[in_name][:,trim:-trim, trim:-trim, trim:-trim]
        self.out_pad_tensors[out_name] = self.out_pad_tensors[in_name][:,trim:-trim, trim:-trim, trim:-trim]

  def match_trim_tensor(self, in_name, match_name, out_name):

    subdomain = SubDomain(self.DxQy.dims*[0], self.DxQy.dims*[1])
    new_subdomain = self.shape_converters[in_name, match_name].out_in_subdomain(subdomain)
    self.trim_tensor(in_name, out_name, abs(new_subdomain.pos[0]))

  def image_combine(self, a_name, b_name, mask_name, out_name):
    # as seen in "Generating Videos with Scene Dynamics" figure 1
    self.out_tensors[out_name] = ((self.out_tensors[a_name] *      self.out_tensors[mask_name] )
                                + self.out_tensors[b_name])
                                #+ (self.out_tensors[b_name] * (1 - self.out_tensors[mask_name])))

    # update padding name
    self.out_pad_tensors[out_name] = self.out_pad_tensors[a_name]

    # take shape converters from a_name
    for name in self.shape_converters.keys():
      if name[1] == a_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == b_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == mask_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])

  def lattice_shape(self, in_name):
    shape = tf.shape(self.out_tensors[in_name])[1:-1]
    return shape

  def num_lattice_cells(self, in_name, return_float=False):
    lattice_shape = self.lattice_shape(in_name)
    lattice_cells = tf.reduce_prod(lattice_shape)
    if return_float:
      lattice_cells = tf.cast(lattice_cells, tf.float32)
    return lattice_cells

  def add_shape_converter(self, name):
    self.shape_converters[name, name] = ShapeConverter()

  def nonlinearity(self, name, nonlinearity_name):
    nonlin = nn.set_nonlinearity(nonlinearity_name)
    self.out_tensors[name] = nonlin(self.out_tensors[name])

  def l1_loss(self, true_name, pred_name, loss_name, factor=None):
    self.out_tensors[loss_name] = tf.reduce_mean(tf.abs(tf.stop_gradient(self.out_tensors[ true_name])
                                              - self.out_tensors[pred_name]))
    if factor is not None:
      self.out_tensors[loss_name] = factor * self.out_tensors[loss_name]

  def l2_loss(self, true_name, pred_name, loss_name, factor=None, normalize=None):

    with tf.device('/cpu:0'):
      self.out_tensors[true_name + '_' + pred_name] = tf.abs(self.out_tensors[true_name]
                                                - self.out_tensors[pred_name])
      #self.lattice_summary(in_name=true_name + '_' + pred_name, summary_name='loss_image')

    if normalize == 'std':
      mean, var = tf.nn.moments(self.out_tensors[true_name], axes=[1,2], keep_dims=True)
      std = tf.sqrt(var)
      self.out_tensors[true_name] = self.out_tensors[true_name] / (1000.0 * std + 0.0001) # TODO take out 10.0, only in to compare with previous code
      self.out_tensors[pred_name] = self.out_tensors[pred_name] / (1000.0 * std + 0.0001)
      self.out_tensors[loss_name] = tf.nn.l2_loss(tf.stop_gradient(self.out_tensors[ true_name]) 
                                                - self.out_tensors[pred_name])
    elif normalize == 'vel':
      vel_true = self.DxQy.lattice_to_vel(self.out_tensors[true_name])
      vel_pred = self.DxQy.lattice_to_vel(self.out_tensors[pred_name])
      rho_true = self.DxQy.lattice_to_rho(self.out_tensors[true_name])[...,0]
      rho_pred = self.DxQy.lattice_to_rho(self.out_tensors[pred_name])[...,0]

      vel_true_x = vel_true[...,0]
      vel_true_y = vel_true[...,1]
      vel_pred_x = vel_true[...,0]
      vel_pred_y = vel_true[...,1]
       
      vel_true_x_min = tf.reduce_min(vel_true_x, axis=[1,2], keep_dims=True)
      vel_true_x_max = tf.reduce_max(vel_true_x, axis=[1,2], keep_dims=True)
      #vel_true_x = (vel_true_x - vel_true_x_min)/(vel_true_x_max - vel_true_x_min + 1e-3)
      #vel_pred_x = (vel_pred_x - vel_true_x_min)/(vel_true_x_max - vel_true_x_min + 1e-3)
      #vel_true_x = (vel_true_x)/(vel_true_x_max - vel_true_x_min + 1e-2)
      #vel_pred_x = (vel_pred_x)/(vel_true_x_max - vel_true_x_min + 1e-2)

      vel_true_y_min = tf.reduce_min(vel_true_y, axis=[1,2], keep_dims=True)
      vel_true_y_max = tf.reduce_max(vel_true_y, axis=[1,2], keep_dims=True)
      #vel_true_y = (vel_true_y - vel_true_y_min)/(vel_true_y_max - vel_true_y_min + 1e-3)
      #vel_pred_y = (vel_pred_y - vel_true_y_min)/(vel_true_y_max - vel_true_y_min + 1e-3)
      #vel_true_y = (vel_true_y)/(vel_true_y_max - vel_true_y_min + 1e-2)
      #vel_pred_y = (vel_pred_y)/(vel_true_y_max - vel_true_y_min + 1e-2)

      rho_true_min = tf.reduce_min(rho_true, axis=[1,2], keep_dims=True)
      rho_true_max = tf.reduce_max(rho_true, axis=[1,2], keep_dims=True)
      #rho_true = (rho_true - rho_true_min)/(rho_true_max - rho_true_min + 1e-3)
      #rho_pred = (rho_pred - rho_true_min)/(rho_true_max - rho_true_min + 1e-3)
      #rho_true = (rho_true)/(rho_true_max - rho_true_min + 1e-2)
      #rho_pred = (rho_pred)/(rho_true_max - rho_true_min + 1e-2)

      self.out_tensors[loss_name] = 0.0
      self.out_tensors[loss_name] += tf.nn.l2_loss(vel_true_x - vel_pred_x)
      self.out_tensors[loss_name] += tf.nn.l2_loss(vel_true_y - vel_pred_y)
      self.out_tensors[loss_name] += tf.nn.l2_loss(rho_true - rho_pred)
      #self.out_tensors[loss_name] += tf.reduce_mean(tf.abs(vel_true_x - vel_pred_x))
      #self.out_tensors[loss_name] += tf.reduce_mean(tf.abs(vel_true_y - vel_pred_y))
      #self.out_tensors[loss_name] += tf.reduce_mean(tf.abs(rho_true - rho_pred))
    elif normalize is None:
      self.out_tensors[loss_name] = tf.nn.l2_loss(tf.stop_gradient(self.out_tensors[ true_name]) 
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
              initializer=tf.constant_initializer(0), trainable=False)
    self.in_tensors[name] = counter
    self.out_tensors[name] = counter
    self.shape_converters[name,name] = ShapeConverter()
 
  def add_tensor(self, name, shape):
    tensor     = tf.placeholder(tf.float32, shape, name=name)
    pad_tensor = tf.placeholder(tf.float32, shape[:-1] + [1], name=name + "_pad")
    self.in_tensors[name] = tensor
    self.out_tensors[name] = tensor
    self.in_pad_tensors[name] = pad_tensor
    self.out_pad_tensors[name] = pad_tensor
    self.shape_converters[name,name] = ShapeConverter()
     
  def add_phase(self): 
    self.in_tensors['phase'] = tf.placeholder(tf.bool, name='phase')
    self.out_tensors['phase'] = self.in_tensors['phase']

  def rename_tensor(self, old_name, new_name):
    # this may need to be handled with more care
    self.out_tensors[new_name] = self.out_tensors[old_name]
    if old_name in self.out_pad_tensors.keys():
      self.out_pad_tensors[new_name] = self.out_pad_tensors[old_name]
    
    # add to shape converter
    for name in self.shape_converters.keys():
      if name[1] == old_name:
        self.shape_converters[name[0], new_name] = copy(self.shape_converters[name])



  def lattice_summary(self, in_name, summary_name, 
                      display_norm=True, display_vel=True, display_pressure=True):
    if display_norm:
      tf.summary.image(summary_name + '_norm', self.DxQy.lattice_to_norm(self.out_tensors[in_name]))
    if display_pressure:
      tf.summary.image(summary_name + '_rho', self.DxQy.lattice_to_rho(self.out_tensors[in_name]))
    if display_vel:
      vel = self.DxQy.lattice_to_vel(self.out_tensors[in_name])
      tf.summary.image(summary_name + '_vel_x', vel[...,0:1])
      tf.summary.image(summary_name + '_vel_y', vel[...,1:2])

  def boundary_summary(self, in_name, summary_name):
    tf.summary.image('physical_boundary', self.out_tensors[in_name][...,0:1])
    tf.summary.image('vel_x_boundary', self.out_tensors[in_name][...,1:2])
    tf.summary.image('vel_y_boundary', self.out_tensors[in_name][...,2:3])
    if len(self.out_tensors[in_name].get_shape()) == 5:
      tf.summary.image('vel_z_boundary', self.out_tensors[in_name][...,3:4])
    tf.summary.image('density_boundary', self.out_tensors[in_name][...,-1:])

  def run(self, out_names, feed_dict=None, return_dict=False):
    # convert out_names to tensors 
    if type(out_names) is list:
      out_tensors = [self.out_tensors[x] for x in out_names]
    else:
      out_tensors = self.out_tensors[out_names]

    # convert feed_dict to tensorflow version
    if feed_dict is not None:
      tf_feed_dict = {}
      for name in feed_dict.keys():
        if type(feed_dict[name]) is tuple:
          tf_feed_dict[self.in_tensors[name]] = feed_dict[name][0]
          tf_feed_dict[self.in_pad_tensors[name]] = feed_dict[name][1]
        else:
          tf_feed_dict[self.in_tensors[name]] = feed_dict[name]
    else:
      tf_feed_dict=None

    # run with tensorflow
    tf_output = self.sess.run(out_tensors, feed_dict=tf_feed_dict)
    
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

  def start_session(self):
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess
