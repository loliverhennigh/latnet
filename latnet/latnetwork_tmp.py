

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
  def __init__(self, config):
    # needed configs
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.network_dir  = config.latnet_network_dir
    self.seq_length = config.seq_length
    gpus = config.gpus.split(',')
    self.gpus = map(int, gpus)

  @classmethod
  def add_options(cls, group):
    pass

  def unroll_global_step(self, name):
    # global step counter
    self.add_step_counter(name) 
    global_step = lambda : self.run(name, feed_dict={})
    return global_step

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
        #print(tf_feed_dict[self.in_tensors[name]].shape)
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


class TrainLatNet(LatNet):

  def __init__(self, config):
    super(LatNet, self).__init__(config)

    # shape converter from in_tensor to out_tensor
    self.train_shape_converters = {}

    # needed configs
    self.train_mode = config.train_mode
    self.train_iters = config.train_iters

  @classmethod
  def add_options(cls, group):
    pass

  def unroll_train_full(self):
    for i in xrange(len(self.gpus)):
      # make input names and output names
      gpu_str = '_gpu_' + str(self.gpus[i])
      state_name = 'state' + gpu_str
      boundary_name = 'boundary' + gpu_str
      pred_state_names = ['pred_state' + gpu_str + '_' + str(x) for x in range(self.seq_length)]
      true_state_names = ['true_state' + gpu_str + '_' + str(x) for x in range(self.seq_length)]
      loss_name = 'l2_loss' + gpu_str
      grad_name = 'gradient' + gpu_str

      with tf.device('/gpu:%d' % self.gpus[i]):
        # make input state and boundary
        self.network_arch.add_lattice(state_name, gpu_id=i)
        self.network_arch.add_boundary(boundary_name, gpu_id=i)
        for j in xrange(self.seq_length):
          self.network_arch.add_lattice(true_state_names[j], gpu_id=i)
      
        # unroll network
        self.network_arch.unroll_full(in_name_state=state_name,
                                      in_name_boundary=boundary_name,
                                      out_names=pred_state_names,
                                      gpu_id=i)
    
        # calc loss
        self.network_arch.l2_loss(true_name=true_state_names,
                                  pred_name=pred_state_names,
                                  loss_name=loss_name,
                                  normalize=True)
 
        # calc grad
        if i == 0:
          all_params = tf.trainable_variables()
        self.network_arch.gradients(params=all_params, out_name=gradient_name)

    ###### Round up losses and Gradients on gpu:0 ######
    with tf.device('/gpu:%d' % self.gpus[0]):
      # round up losses
      loss_names = ['l2_loss_gpu_' + str(x) for x in xrange(self.seq_length)]
      loss_name = 'l2_loss'
      self.network_arch.sum_losses(in_names=loss_names, out_name=loss_name)
      # round up gradients
      gradient_names = ['gradient_gpu_' + str(x) for x in xrange(self.seq_length)]
      gradient_name = 'gradient'
      self.network_arch.sum_gradients(in_names=gradient_names, out_name=gradient_name)

    ###### Train Operation ######
    self.optimizer = Optimizer(self.config, name='opt', optimizer_name='adam')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self.out_tensors['gen_train_op'] = self.gen_optimizer.train_op(gen_params, 
                                             self.out_tensors['gen_grads' + gpu_str(0)], 
                                             self.out_tensors['gen_global_step'],
                                             mom1=self.config.beta1,
                                             other_update=update_ops)

    def train_step(feed_dict):
      self.run('loss', 
    train_step = lambda x: self.run('cstate_from_state', 
                             feed_dict={'state':x})

  def train_unroll(self):

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
          self.add_tensor('boundary' + gpu_str, (1 + self.DxQy.dims) * [None] + [self.DxQy.boundary_dims])
          self.add_tensor('boundary_small' + gpu_str, (1 + self.DxQy.dims) * [None] + [self.DxQy.boundary_dims])
          self.add_tensor('cstate' + gpu_str, (1 + self.DxQy.dims) * [None] + [self.config.filter_size_compression])
          self.add_tensor('cboundary' + gpu_str, (1 + self.DxQy.dims) * [None] + [self.config.filter_size_compression])
          self.add_phase() 
          if i == 0:
            with tf.device('/cpu:0'):
              self.lattice_summary(in_name='state' + gpu_str, summary_name='true')
              self.boundary_summary(in_name='boundary' + gpu_str, summary_name='boundary')
          # make seq of output states
          for j in xrange(self.seq_length):
            self.add_tensor('true_state' + seq_str(j), (1 + self.DxQy.dims) * [None] + [self.DxQy.Q])
            self.add_tensor('true_comp_cstate' + seq_str(j), (1 + self.DxQy.dims) * [None] + [self.config.filter_size_compression])
            #self.out_tensors["true_state" + seq_str(j)] = self.out_tensors["true_state" + seq_str(j)]
            if i == 0:
              with tf.device('/cpu:0'):
                self.lattice_summary(in_name='true_state' + seq_str(j), summary_name='true_' + str(j))
      
          ###### Unroll Graph ######
          ### encode ###
          self._encoder_state(in_name="state" + gpu_str, 
                              out_name="cstate" + seq_str(0))
          self._encoder_boundary(in_name="boundary" + gpu_str, 
                                out_name="cboundary" + gpu_str)
          if not self.train_autoencoder:
            self._encoder_state(in_name="state" + gpu_str, 
                                out_name="cstate_auto")
            self.rename_tensor(old_name='cstate' + gpu_str,
                               new_name="cstate" + seq_str(0))
      
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
                                out_name="pred_state" + seq_str(j),
                                lattice_size=self.DxQy.Q)

            ### encode seq state  
            self._encoder_state(in_name="true_state" + seq_str(j),
                                out_name="true_cstate" + seq_str(j))
            self._decoder_state(in_cstate_name="true_cstate" + seq_str(j),
                                in_cboundary_name="cboundary" + gpu_str, 
                                out_name="true_state_compare" + seq_str(j),
                                lattice_size=self.DxQy.Q)
            self.match_trim_tensor(in_name="true_state" + seq_str(j), 
                                   match_name="true_state_compare" + seq_str(j), 
                                   out_name="true_state_compare_" + seq_str(j),
                                   in_out=True)

            if i == 0:
              with tf.device('/cpu:0'):
                # make image summary
                self.lattice_summary(in_name='pred_state' + seq_str(j), summary_name='pred_' + str(j))
                #self.lattice_summary(in_name='true_state_compare_' + seq_str(j), summary_name='true_compare_' + str(j))

 
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
            #self.out_tensors["loss_comp_l2" + gpu_str] = 0.0

            if self.train_autoencoder:
              self.out_tensors["loss_auto_l2" + gpu_str] = 0.0

              for j in range(0, self.seq_length):
                # normalize loss to make comparable for diffrent input sizes
                # TODO remove 100.0 (only in to make comparable to previous code)
                #l2_factor = 1.0*(256.0*256.0)/self.num_lattice_cells('pred_state' + seq_str(j), return_float=True)
                l2_factor = 1.0*(16*16*16.)/(len(self.gpus)*self.config.batch_size*self.num_lattice_cells('pred_state' + seq_str(j), return_float=True))
                self.l2_loss(true_name='true_state_compare_' + seq_str(j),
                             pred_name='pred_state' + seq_str(j),
                             loss_name='loss_auto_l2' + seq_str(j),
                             factor=l2_factor/1.0)
                             #factor=l2_factor/num_samples)
                #l2_factor = (4*4*4.0)/(len(self.gpus)*self.config.batch_size*self.num_lattice_cells('pred_state' + seq_str(j), return_float=True))
                #self.l2_loss(true_name='true_cstate' + seq_str(j),
                #             pred_name='cstate' + seq_str(j),
                #             loss_name='loss_comp_l2' + seq_str(j),
                #             factor=l2_factor/1.0)
                # add up losses
                self.out_tensors['loss_auto_l2' + gpu_str] += self.out_tensors['loss_auto_l2' + seq_str(j)] 
                #self.out_tensors['loss_comp_l2' + gpu_str] += self.out_tensors['loss_comp_l2' + seq_str(j)] 
            else:
              self.out_tensors["loss_comp_l2" + gpu_str] = 0.0

              for j in range(1, self.seq_length):
                # normalize loss to make comparable for diffrent input sizes
                #l2_factor = 1.0*(256.0*256.0)/self.num_lattice_cells('pred_state' + seq_str(j), return_float=True)
                l2_factor = (4*4*4.0)/(len(self.gpus)*self.config.batch_size*self.num_lattice_cells('cstate' + seq_str(j), return_float=True))
                self.l2_loss(true_name='true_comp_cstate' + seq_str(j),
                             pred_name='cstate' + seq_str(j),
                             loss_name='loss_comp_l2' + seq_str(j),
                             factor=l2_factor/1.0)
                # add up losses
                self.out_tensors['loss_comp_l2' + gpu_str] += self.out_tensors['loss_comp_l2' + seq_str(j)] 
 
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
            if self.train_autoencoder:
              self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_auto_l2' + gpu_str]
            else:
              self.out_tensors['loss_gen' + gpu_str] += self.out_tensors['loss_comp_l2' + gpu_str]
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
          print("trainable variables:")
          for v in gen_params:
            print(v.name)
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
            if self.train_autoencoder:
              self.out_tensors['loss_auto_l2' + gpu_str(0)] += self.out_tensors['loss_auto_l2' + gpu_str(i)]
            else:
              self.out_tensors['loss_comp_l2' + gpu_str(0)] += self.out_tensors['loss_comp_l2' + gpu_str(i)]
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
        if self.train_autoencoder:
          self.rename_tensor(old_name='loss_auto_l2' + gpu_str(0), new_name='loss_auto_l2')
        else:
          self.rename_tensor(old_name='loss_comp_l2' + gpu_str(0), new_name='loss_comp_l2')
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
        if self.train_autoencoder: 
          tf.summary.scalar('loss_auto_l2', self.out_tensors['loss_auto_l2'])
        else:
          tf.summary.scalar('loss_comp_l2', self.out_tensors['loss_comp_l2'])
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
        name = ("true_state_" + str(j) + gpu_str, "true_cstate_" + str(j) + gpu_str)
        shape_converters[name] = self.shape_converters[name]
    return shape_converters

  def eval_unroll(self):

    # graph
    with tf.Graph().as_default():

      ###### Inputs to Graph ######
      # make input state and boundary
      self.add_tensor('state',     (1 + self.DxQy.dims) * [None] + [self.DxQy.Q])
      self.add_tensor('boundary',  (1 + self.DxQy.dims) * [None] + [self.DxQy.boundary_dims])
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
      self._decoder_state(in_cstate_name="cstate", in_cboundary_name="cboundary_decoder", out_name="state_from_cstate", lattice_size=self.DxQy.Q)
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

  def encoder_lambda(self, state):
    cstate = self.run('cstate_auto', feed_dict={'state_gpu_0': state})
    return cstate

  def run(self, out_names, feed_dict=None, return_dict=False):
    # convert out_names to tensors 
    if type(out_names) is list:
      out_tensors = [self.network_arch.out_tensors[x] for x in out_names]
    else:
      out_tensors = self.network_arch.out_tensors[out_names]

    # convert feed_dict to tensorflow version
    if feed_dict is not None:
      tf_feed_dict = {}
      for name in feed_dict.keys():
        if type(feed_dict[name]) is tuple:
          tf_feed_dict[self.network_arch.in_tensors[name]] = feed_dict[name][0]
          tf_feed_dict[self.network_arch.in_pad_tensors[name]] = feed_dict[name][1]
        else:
          tf_feed_dict[self.in_tensors[name]] = feed_dict[name]
        #print(tf_feed_dict[self.in_tensors[name]].shape)
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


class TrainLatNet(LatNet):



