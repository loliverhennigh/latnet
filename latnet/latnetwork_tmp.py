

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
from config import NONSAVE_CONFIGS

class LatNet(object):
  def __init__(self, config):
    # needed configs
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.network_dir  = config.latnet_network_dir
    self.seq_length = config.seq_length
    gpus = config.gpus.split(',')
    self.gpus = map(int, gpus)

    # make save and load values options
    self.checkpoint_path = self._make_checkpoint_path(config)

  @classmethod
  def add_options(cls, group):
    pass

  def _make_checkpoint_path(self, config):
    # make checkpoint path with all the flags specifing different directories
 
    # run through all params and add them to the base path
    base_path = self.network_dir + '/' + self.network_name
    for k, v in self.config.__dict__.items():
      if k not in NONSAVE_CONFIGS:
        base_path += '/' + k + '.' + str(v)
    return base_path

  def _make_saver(self):
    variables = tf.global_variables()
    variables_autoencoder = [v for i, v in enumerate(variables) if ("coder" in v.name[:v.name.index(':')]) or ('global' in v.name[:v.name.index(':')])]
    variables_compression = [v for i, v in enumerate(variables) if "compression_mapping" in v.name[:v.name.index(':')]]
    self.saver_all = tf.train.Saver(variables, max_to_keep=1)
    self.saver_autoencoder = tf.train.Saver(variables_autoencoder)
    self.saver_compression = tf.train.Saver(variables_compression)

  def load_checkpoint(self, maybe_remove_prev=False):

    # make saver
    self._make_saver()

    # load everything
    ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
    print("looking for checkpoint in " + self.checkpoint_path) 
    if ckpt is not None:
      print("trying init autoencoder from " + ckpt.model_checkpoint_path)
      try:
        self.saver_autoencoder.restore(self.sess, ckpt.model_checkpoint_path)
      except:
        if maybe_remove_prev:
          tf.gfile.DeleteRecursively(self.checkpoint_path)
          tf.gfile.MakeDirs(self.checkpoint_path)
        print("there was a problem using variables for autoencoder in checkpoint, random init will be used instead")
      print("trying init compression mapping from " + ckpt.model_checkpoint_path)
      try:
        self.saver_compression.restore(self.sess, ckpt.model_checkpoint_path)
      except:
        if maybe_remove_prev:
          tf.gfile.DeleteRecursively(self.checkpoint_path)
          tf.gfile.MakeDirs(self.checkpoint_path)
        print("there was a problem using variables for compression mapping in checkpoint, random init will be used instead")
    else:
      print("using rand init")

  def save_checkpoint(self, global_step):
    save_path = os.path.join(self.checkpoint_path, 'model.ckpt')
    self.saver_all.save(sess, save_path, global_step=global_step)  

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

  def unroll_global_step(self, name):
    # global step counter
    self.add_step_counter(name) 
    global_step = lambda : self.run(name, feed_dict={})
    return global_step

  def unroll_save_summary(self, graph_def):
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(self.checkpoint_path, graph_def=graph_def)
    def save_summary(feed_dict, step):
      summary_str = self.run('summary_op', feed_dict=feed_dict)
      self.summary_writer.add_summary(summary_str, global_step) 
    return save_summary

class TrainLatNet(LatNet):

  def __init__(self, config):
    super(LatNet, self).__init__(config)

    # needed configs
    self.train_mode = config.train_mode
    self.train_iters = config.train_iters

    # make optimizer
    self.optimizer = Optimizer(config, name='opt', optimizer_name='adam')

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
      cstate_names = ['comp_state' + gpu_str + '_' + str(x) for x in range(self.seq_length)]

      loss_name = 'l2_loss' + gpu_str
      grad_name = 'gradient' + gpu_str

      with tf.device('/gpu:%d' % self.gpus[i]):
        # make input state and boundary
        self.add_lattice(state_name, gpu_id=i)
        self.add_boundary(boundary_name, gpu_id=i)
        for j in xrange(self.seq_length):
          self.add_lattice(true_state_names[j], gpu_id=i)
      
        # unroll network
        self.full_seq_pred(in_name_state=state_name,
                           in_name_boundary=boundary_name,
                           out_cstate_names=pred_state_names,
                           out_names=pred_state_names,
                           gpu_id=i)
    
        # calc loss
        self.l2_loss(true_name=true_state_names,
                     pred_name=pred_state_names,
                     loss_name=loss_name,
                     normalize=True)
 
        # calc grad
        if i == 0:
          all_params = tf.trainable_variables()
        self.gradients(loss_name=loss_name, grad_name=grad_name, params=all_params)

    ###### Round up losses and Gradients on gpu:0 ######
    with tf.device('/gpu:%d' % self.gpus[0]):
      # round up losses
      loss_names = ['l2_loss_gpu_' + str(x) for x in xrange(self.seq_length)]
      loss_name = 'l2_loss'
      self.sum_losses(in_names=loss_names, out_name=loss_name)
      # round up gradients
      gradient_names = ['gradient_gpu_' + str(x) for x in xrange(self.seq_length)]
      gradient_name = 'gradient'
      self.sum_gradients(in_names=gradient_names, out_name=gradient_name)

    ###### Train Operation ######
    self.out_tensors['train_op'] = self.optimizer.train_op(all_params, 
                                             self.out_tensors['gradient'], 
                                             self.out_tensors['global_step'],
                                             mom1=self.config.beta1)

    # make a train step
    def train_step(feed_dict):
      loss, _ = self.run(['loss', 'train_op'], feed_dict=feed_dict)
      return {'loss': loss}

    # make shape converter
    shape_converters = {}
    shape_converters['state_shape_converter'] = self.shape_converters[(state_name, cstate_name[-1])]
    shape_converters['seq_state_shape_converter'] = self.shape_converters[(state_name, cstate_name[-1])]

    return train_step, shape_converters

  def unroll_train_comp(self):
    for i in xrange(len(self.gpus)):
      # make input names and output names
      gpu_str = '_gpu_' + str(self.gpus[i])
      cstate_name = 'cstate' + gpu_str
      cboundary_name = 'cboundary' + gpu_str
      pred_cstate_names = ['pred_cstate' + gpu_str + '_' + str(x) for x in range(self.seq_length)]
      true_cstate_names = ['true_cstate' + gpu_str + '_' + str(x) for x in range(self.seq_length)]
      loss_name = 'l2_loss' + gpu_str
      grad_name = 'gradient' + gpu_str

      with tf.device('/gpu:%d' % self.gpus[i]):
        # make input cstate and cboundary
        self.add_cstate(cstate_name, gpu_id=i)
        self.add_cboundary(cboundary_name, gpu_id=i)
        for j in xrange(self.seq_length):
          self.add_lattice(true_cstate_names[j], gpu_id=i)
      
        # unroll network
        self.comp_seq_pred(in_name_cstate=cstate_name,
                           in_name_boundary=boundary_name,
                           out_names=pred_cstate_names,
                           gpu_id=i)
    
        # calc loss
        self.l2_loss(true_name=true_cstate_names,
                     pred_name=pred_cstate_names,
                     loss_name=loss_name,
                     normalize=True)
 
        # calc grad
        if i == 0:
          all_params = tf.trainable_variables()
          comp_params = [v for i, v in enumerate(gen_params) if "compression_mapping" in v.name[:v.name.index(':')]]
        self.gradients(loss_name=loss_name, grad_name=grad_name, params=comp_params)

    ###### Round up losses and Gradients on gpu:0 ######
    with tf.device('/gpu:%d' % self.gpus[0]):
      # round up losses
      loss_names = ['l2_loss_gpu_' + str(x) for x in xrange(self.seq_length)]
      loss_name = 'l2_loss'
      self.sum_losses(in_names=loss_names, out_name=loss_name)
      # round up gradients
      gradient_names = ['gradient_gpu_' + str(x) for x in xrange(self.seq_length)]
      gradient_name = 'gradient'
      self.sum_gradients(in_names=gradient_names, out_name=gradient_name)

    ###### Train Operation ######
    self.out_tensors['train_op'] = self.optimizer.train_op(all_params, 
                                             self.out_tensors['gradient'], 
                                             self.out_tensors['global_step'],
                                             mom1=self.config.beta1)

    # make a train step
    def train_step(feed_dict):
      loss, _ = self.run(['loss', 'train_op'], feed_dict=feed_dict)
      return {'loss': loss}

    # make shape converter
    shape_converters = {}
    shape_converters['cstate_shape_converter'] = self.shape_converters[(state_name, cstate_name[-1])]
    shape_converters['seq_cstate_shape_converter'] = self.shape_converters[(state_name, cstate_name[-1])]

    return train_step, shape_converter

  def make_data_queue(self, config):
    # add script name to domains TODO This is a little weird and might be taken out later
    for domain in self.domains:
      if domain is not None:
        domain.script_name = self.script_name

    data_queue = DataQueue(self.config, 
                           self.config.train_sim_dir,
                           self.domains, 
                           self.train_shape_converter())
    return data_queue

  def train(self):
    # unroll network
    with tf.Graph().as_default():
      get_step = self.unroll_global_step()
      if self.train_mode == "full": 
        train_step, train_shape_converters = self.unroll_train_full()
      elif self.train_mode == "compression":
        encode_state, state_shape_converter = self.unroll_state_encoder()
        encode_boundary, boundary_shape_converters = self.unroll_boundary_encoder()
        train_step, train_shape_converter = self.unroll_train_comp()
      save_sumary = self.unroll_save_summary()
      self.load_checkpoint() 
   
    # get data
    if self.train_mode == "full":
      self.data_queue.add_rand_dp(100, 
                 train_shape_conveters['state_converter'], 
                 train_shape_conveters['seq_state_converter'])
    elif self.train_mode == "compression":
      self.data_queue.add_rand_cdp(100, 
                 train_shape_conveters['cstate_converter'], 
                 train_shape_conveters['seq_cstate_converter'], 
                 encode_state=encode_state, 
                 encode_boundary=encode_boundary)
 
    while True 
      step = get_step()
      if self.train_mode == "full":
        feed_dict = self.data_queue.dp_minibatch()
      elif self.train_mode == "compression":
        feed_dict = self.data_queue.cdp_minibatch()

      # run train op and get loss output
      output = train_step(feed_dict)
          
      # update loss summary
      self.update_loss_stats(output)
    
      # update time summary
      self.update_time_stats()
    
      # print required data and save
      if step % steps_per_print == 0:
        self.print_stats(self.loss_stats, self.time_stats, data_queue_train.queue_stats(), step)
  
      # save in tensorboard 
      if step % 100 == 0:
        save_summary(feed_dict)
    
      # save in network
      if step % self.save_network_freq == 0:
        self.save_checkpoint(step)
   
      # possibly add more data if JHTDB
      #if step % 100 == 0 and self.dataset == "JHTDB":
      #  self.data_queue(
      # possibly get more data
      #if step % 200 == 0:
      #  self.active_data_add()
 
  def update_loss_stats(self, output):
    names = output.keys()
    names.sort()
    for name in names:
      if 'loss' in name:
        # update loss history
        if name + '_history' not in self.loss_stats.keys():
          self.loss_stats[name + '_history'] = []
        self.loss_stats[name + '_history'].append(output[name])
        if len(self.loss_stats[name + '_history']) > self.stats_history_length:
          self.loss_stats[name + '_history'].pop(0)
        # update loss
        self.loss_stats[name] = float(output[name])
        # update ave loss
        self.loss_stats[name + '_ave'] = float(np.sum(np.array(self.loss_stats[name + '_history']))
                                         / len(self.loss_stats[name + '_history']))
        # update var loss
        self.loss_stats[name + '_std'] = np.sqrt(np.var(np.array(self.loss_stats[name + '_history'])))

  def update_time_stats(self):
    # stop timer
    self.toc = time.time()
    # update total run time
    self.time_stats['run_time'] = int(time.time() - self.start_time)
    # update total step time
    self.time_stats['step_time'] = ((self.toc - self.tic) / 
                                    (self.config.batch_size * len(self.config.gpus)))
    # update time history
    if 'step_time_history' not in self.time_stats.keys():
      self.time_stats['step_time_history'] = []
    self.time_stats['step_time_history'].append(self.time_stats['step_time'])
    if len(self.time_stats['step_time_history']) > self.stats_history_length:
      self.time_stats['step_time_history'].pop(0)
    # update time ave
    self.time_stats['step_time_ave'] = float(np.sum(np.array(self.time_stats['step_time_history']))
                   / len(self.time_stats['step_time_history']))
    # start timer
    self.tic = time.time()

  def print_stats(self, loss_stats, time_stats, queue_stats, step):
    loss_string = print_dict('LOSS STATS', loss_stats, 'blue')
    time_string = print_dict('TIME STATS', time_stats, 'magenta')
    queue_string = print_dict('QUEUE STATS', queue_stats, 'yellow')
    print_string = loss_string + time_string + queue_string
    os.system('clear')
    print("TRAIN INFO - step " + str(step))
    print(print_string)

class EvalLatNet(LatNet):

  def __init__(self, config):
    super(LatNet, self).__init__(config)

    # needed configs
    self.train_mode = config.train_mode
    self.train_iters = config.train_iters

    # shape converter from in_tensor to out_tensor
    self.train_shape_converters = {}

    # make optimizer
    self.optimizer = Optimizer(config, name='opt', optimizer_name='adam')

  @classmethod
  def add_options(cls, group):
    pass

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

