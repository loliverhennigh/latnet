

import time
from copy import copy
import os
from termcolor import colored, cprint
import tensorflow as tf
import numpy as np
from tqdm import *
import math
import itertools

import lattice
import nn as nn

from data_queue import DataQueue
from shape_converter import ShapeConverter
from optimizer import Optimizer
from shape_converter import SubDomain
from sim_saver import SimSaver
from config import NONSAVE_CONFIGS
from utils.python_utils import *
from utils.numpy_utils import mobius_extract_2, stack_grid

class LatNet(object):
  def __init__(self, config):
    super(LatNet, self).__init__(config)
    # needed configs
    self.run_mode = config.run_mode
    if config.run_mode == "train":
      self.train_mode = config.train_mode
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.network_dir  = config.network_dir
    #gpus = config.gpus.split(',')
    #self.gpus = map(int, gpus)

    # make save and load values options
    self.checkpoint_path = self._make_checkpoint_path(config)

  @classmethod
  def add_options(cls, group):
    group.add_argument('--network_dir', 
                   help='where to save neural network', 
                   type=str,
                   default='./network_checkpoint')
    group.add_argument('--cstate_depth',
                   help='depth of compressed state',
                   type=int,
                   default=16)
    group.add_argument('--cboundary_depth',
                   help='depth of compressed boundary',
                   type=int,
                   default=32)

  def _make_checkpoint_path(self, config):
    # make checkpoint path with all the flags specifing different directories
 
    # run through all params and add them to the base path
    base_path = self.network_dir + '/' + self.network_name
    dic = config.__dict__
    keys = list(dic.keys())
    keys.sort()
    for k in keys:
      if k not in NONSAVE_CONFIGS:
        base_path += '/' + k + '.' + str(dic[k])

    if self.run_mode == "train":
      if self.train_mode == "full":
        base_path += '/full_train'
    return base_path

  def _make_saver(self):
    variables = tf.global_variables()
    variables_autoencoder = [v for i, v in enumerate(variables) if ("coder" in v.name[:v.name.index(':')]) or ('global' in v.name[:v.name.index(':')])]
    variables_compression = [v for i, v in enumerate(variables) if "compression_mapping" in v.name[:v.name.index(':')]]
    #for i, v in enumerate(variables_autoencoder):
    #  print(v.name)
    #for i, v in enumerate(variables_compression):
    #  print(v.name)
    
    self.saver_all = tf.train.Saver(variables, max_to_keep=1)
    self.saver_autoencoder = tf.train.Saver(variables_autoencoder)
    #self.saver_compression = tf.train.Saver(variables_compression)

  def load_checkpoint(self, maybe_remove_prev=False):

    # make saver
    self._make_saver()

    # load if training everything
    if self.run_mode == "train":
      if self.train_mode == "full":
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt is not None:
          print("looking for checkpoint in " + self.checkpoint_path) 
          try:
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
          except:
            print("there was a problem loading variables, random init will be used instead")
  
      elif self.train_mode == "compression":
        full_train_ckpt = tf.train.get_checkpoint_state(self.checkpoint_path + '/full_train')
        if full_train_ckpt is not None:
          print("trying init full train from " + full_train_ckpt.model_checkpoint_path)
          self.saver_autoencoder.restore(self.sess, full_train_ckpt.model_checkpoint_path)
          try:
            self.saver_autoencoder.restore(self.sess, full_train_ckpt.model_checkpoint_path)
          except:
            print("there was a problem using variables from full train. Try training with --train_mode=full before training compression mapping.")
            exit()
        else:
          print("there was a problem using variables from full train. Try training with --train_mode=full before training compression mapping.")
          exit()
  
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt is not None:
          print("trying init compression train from " + ckpt.model_checkpoint_path)
          try:
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
          except:
            print("there was a problem using variables from compression train. using init from full train instead.")

    if self.run_mode in ["eval", 'decode']:
      print("looking for checkpoint in " + self.checkpoint_path) 
      ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
      if ckpt is None:
        print("now looking for checkpoint in " + self.checkpoint_path + '/full_train')
        ckpt =  tf.train.get_checkpoint_state(self.checkpoint_path + '/full_train')
      if ckpt is not None:
        try:
          self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
        except:
          print("there was a problem loading variables, random init will be used instead")
      else:
        print("no checkpoint found, using rand init") 
 
  def save_checkpoint(self, global_step):
    save_path = os.path.join(self.checkpoint_path, 'model.ckpt')
    self.saver_all.save(self.sess, save_path, global_step=int(global_step))

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
        if type(feed_dict[name]) in (list, tuple):
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
        for i in range(len(out_names)):
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
    self.sess = sess

  def unroll_global_step(self):
    # global step counter
    self.add_step_counter('global_step') 
    global_step = lambda : self.run('global_step', feed_dict={})
    return global_step

  def unroll_save_summary(self, graph_def):
    summary_op = tf.summary.merge_all()
    self.out_tensors['summary_op'] = summary_op
    self.summary_writer = tf.summary.FileWriter(self.checkpoint_path, graph_def=graph_def)
    def save_summary(feed_dict, step):
      summary_str = self.run('summary_op', feed_dict=feed_dict)
      self.summary_writer.add_summary(summary_str, step) 
    return save_summary

  def unroll_state_encoder(self):
    state_name = 'state'
    cstate_name = 'cstate'
    self.add_lattice(state_name, gpu_id=1)
    with tf.variable_scope('unroll') as scope:
      self.encoder_state(in_name=state_name,
                         out_name=cstate_name)
    def encode_state(feed_dict):
      cstate = self.run(cstate_name, feed_dict=feed_dict)
      return cstate

    # make shape converter
    shape_converters = {}
    shape_converters['state_converter'] = self.shape_converters[(state_name, cstate_name)]

    return encode_state, shape_converters

  def unroll_boundary_encoder(self):
    boundary_name = 'boundary'
    cboundary_name = 'cboundary'
    self.add_boundary(boundary_name, gpu_id=1)
    with tf.variable_scope('unroll') as scope:
      self.encoder_boundary(in_name=boundary_name,
                         out_name=cboundary_name)
    def encode_boundary(feed_dict):
      cboundary = self.run(cboundary_name, feed_dict=feed_dict)
      return cboundary

    # make shape converter
    shape_converters = {}
    shape_converters['boundary_converter'] = self.shape_converters[(boundary_name, cboundary_name)]

    return encode_boundary, shape_converters

  def unroll_compression_mapping(self):
    cstate_name = 'cstate_0'
    cboundary_name = 'cboundary_0'
    out_name = 'cstate_1'
    self.add_cstate(cstate_name)
    self.add_cboundary(cboundary_name)
    with tf.variable_scope('unroll') as scope:
      self.compression_mapping(in_cstate_name=cstate_name,
                               in_cboundary_name=cboundary_name,
                               out_name=out_name)

    def compression_mapping(feed_dict):
      feed_dict[cstate_name] = feed_dict.pop('cstate')
      feed_dict[cboundary_name] = feed_dict.pop('cboundary')
      cstate = self.run(out_name, feed_dict=feed_dict)
      return cstate

    # make shape converter
    shape_converters = {}
    shape_converters['cstate_converter'] = self.shape_converters[(cstate_name, out_name)]
    return compression_mapping, shape_converters

  def unroll_decode_state(self):
    cstate_name = 'cstate_2'
    state_name = 'state_2'
    self.add_cstate(cstate_name)
    with tf.variable_scope('unroll') as scope:
      self.decoder_state(in_name=cstate_name,
                         out_name=state_name)

    def decode_state(feed_dict):
      feed_dict[cstate_name] = feed_dict.pop('cstate')
      state = self.run(state_name, feed_dict=feed_dict)
      return state

    # make shape converter
    shape_converters = {}
    shape_converters['state_converter'] = self.shape_converters[(cstate_name, state_name)]
    return decode_state, shape_converters

class TrainLatNet(LatNet):

  def __init__(self, config):
    super(TrainLatNet, self).__init__(config)

    # needed configs
    self.train_mode = config.train_mode
    self.train_iters = config.train_iters
    self.batch_size = config.batch_size
    self.seq_length = config.seq_length
    self.dataset = config.dataset
    gpus = config.gpus.split(',')
    self.gpus = list(map(int, gpus))

    # make optimizer
    self.optimizer = Optimizer(config, name='opt')

    # make data queue
    self.data_queue = self.make_data_queue(config)

    # values for when to print and save data (hard set for now)
    self.save_network_freq = 100
    self.save_tensorboard_freq = 100
    self.stats_print_freq = 20
    self.add_data_freq = 250
    self.nr_data_to_add = (len(self.gpus)*self.batch_size*self.add_data_freq)/50 # hard set to adding a data point for every 10 trained on since last add

    # make stats data
    self.loss_stats = {}
    self.time_stats = {}
    self.start_time = time.time()
    self.tic = time.time()
    self.toc = time.time()
    self.stats_history_length = 300

  @classmethod
  def add_options(cls, group):
    group.add_argument('--train_mode', 
                   help='either train of full states or compressed states', 
                   type=str,
                   choices=['full', 'compression'],
                   default='full')
    group.add_argument('--seq_length', 
                   help='how many step to unroll when training', 
                   type=int,
                   default=5)
    group.add_argument('--train_cshape', 
                   help='size of data to train on', 
                   type=str,
                   default='16x16')
    group.add_argument('--batch_size',
                   help='batch size for training',
                   type=int,
                   default=4)
    group.add_argument('--train_iters',
                   help='num iters to train network',
                   type=int,
                   default=500000)
    group.add_argument('--gpus',
                   help='gpus to train on',
                   type=str,
                   default='0')
    group.add_argument('--gpu_fraction',
                   help='fraction of gpu memory to use',
                   type=float,
                   default=0.9)

  def unroll_train_full(self):
    for i in range(len(self.gpus)):
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
        for j in range(self.seq_length):
          self.add_lattice(true_state_names[j], gpu_id=i)
      
        # unroll network
        self.full_seq_pred(in_state_name=state_name,
                           in_boundary_name=boundary_name,
                           out_cstate_names=cstate_names,
                           out_names=pred_state_names,
                           gpu_id=i)

        # save output in tensorboard
        for n in pred_state_names:
          self.lattice_summary(n, n)
    
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
      loss_names = ['l2_loss_gpu_' + str(x) for x in range(len(self.gpus))]
      loss_name = 'l2_loss'
      self.sum_losses(loss_names=loss_names, out_name=loss_name)
      # round up gradients
      gradient_names = ['gradient_gpu_' + str(x) for x in range(len(self.gpus))]
      gradient_name = 'gradient'
      self.sum_gradients(gradient_names=gradient_names, out_name=gradient_name)

    ###### Train Operation ######
    self.out_tensors['train_op'] = self.optimizer.train_op(all_params, 
                                             self.out_tensors['gradient'], 
                                             self.out_tensors['global_step'])

    # make a train step
    def train_step(feed_dict):
      loss, _ = self.run(['l2_loss', 'train_op'], feed_dict=feed_dict)
      return {'l2_loss': loss}

    # make shape converter
    shape_converters = {}
    shape_converters['state_converter'] = self.shape_converters[(state_name, cstate_names[-1])]
    shape_converters['seq_state_converter'] = self.shape_converters[(state_name, pred_state_names[-1])]

    return train_step, shape_converters

  def unroll_train_comp(self):
    for i in range(len(self.gpus)):
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
        self.add_cstate(cstate_name)
        self.add_cboundary(cboundary_name)
        for j in range(self.seq_length):
          self.add_cstate(true_cstate_names[j])
      
        # unroll network
        self.comp_seq_pred(in_cstate_name=cstate_name,
                           in_cboundary_name=cboundary_name,
                           out_names=pred_cstate_names,
                           gpu_id=i)
    
        # calc loss
        self.l2_loss(true_name=true_cstate_names[1:],
                     pred_name=pred_cstate_names[1:],
                     loss_name=loss_name,
                     normalize=True)
 
        # calc grad
        if i == 0:
          all_params = tf.trainable_variables()
          comp_params = [v for i, v in enumerate(all_params) if "compression_mapping" in v.name[:v.name.index(':')]]
    
        self.gradients(loss_name=loss_name, grad_name=grad_name, params=comp_params)

    ###### Round up losses and Gradients on gpu:0 ######
    with tf.device('/gpu:%d' % self.gpus[0]):
      # round up losses
      loss_names = ['l2_loss_gpu_' + str(x) for x in range(len(self.gpus))]
      loss_name = 'l2_loss'
      self.sum_losses(loss_names=loss_names, out_name=loss_name)
      # round up gradients
      gradient_names = ['gradient_gpu_' + str(x) for x in range(len(self.gpus))]
      gradient_name = 'gradient'
      self.sum_gradients(gradient_names=gradient_names, out_name=gradient_name)

    ###### Train Operation ######
    self.out_tensors['train_op'] = self.optimizer.train_op(comp_params, 
                                             self.out_tensors['gradient'], 
                                             self.out_tensors['global_step'])

    # make a train step
    def train_step(feed_dict):
      loss, _ = self.run(['l2_loss', 'train_op'], feed_dict=feed_dict)
      return {'l2_loss': loss}

    # make shape converter
    shape_converters = {}
    shape_converters['seq_cstate_converter'] = self.shape_converters[(cstate_name, pred_cstate_names[-1])]

    return train_step, shape_converters

  def make_data_queue(self, config):
    # add script name to domains TODO This is a little weird and might be taken out later
    for domain in self.domains:
      if domain is not None:
        domain.script_name = self.script_name

    data_queue = DataQueue(config, 
                           self.domains)
    return data_queue

  def train(self):
    # unroll network
    with tf.Graph().as_default():
      get_step = self.unroll_global_step()
      if self.train_mode == "full": 
        train_step, train_shape_converter = self.unroll_train_full()
      elif self.train_mode == "compression":
        encode_state, state_shape_converter = self.unroll_state_encoder()
        encode_boundary, boundary_shape_converters = self.unroll_boundary_encoder()
        train_step, train_shape_converter = self.unroll_train_comp()
        train_shape_converter.update(state_shape_converter)
      self.start_session()
      graph_def = self.sess.graph.as_graph_def(add_shapes=True)
      save_summary = self.unroll_save_summary(graph_def)
      self.load_checkpoint() 
   
    # get data
    if self.train_mode == "full":
      self.data_queue.load_dp(100, 
                 train_shape_converter['state_converter'], 
                 train_shape_converter['seq_state_converter'])
    elif self.train_mode == "compression":
      self.data_queue.load_cdp(100, 
                 train_shape_converter['state_converter'], 
                 train_shape_converter['seq_cstate_converter'], 
                 encode_state=encode_state, 
                 encode_boundary=encode_boundary)
 
    while True:
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
      if step % self.stats_print_freq == 0:
        if self.train_mode == "full":
          self.print_stats(self.loss_stats, self.time_stats, self.data_queue.queue_dp_stats(), step)
        elif self.train_mode == "compression":
          self.print_stats(self.loss_stats, self.time_stats, self.data_queue.queue_cdp_stats(), step)
  
      # save in tensorboard 
      if step % self.save_tensorboard_freq == 0:
        save_summary(feed_dict, step)
    
      # save in network
      if step % self.save_network_freq == 0:
        self.save_checkpoint(step)
   
      # possibly add more data
      if step % self.add_data_freq == 0:
        if self.train_mode == "full":
          self.active_dp_add(train_shape_converter['state_converter'], 
                              train_shape_converter['seq_state_converter'])
        elif self.train_mode == "compression":
          self.active_cdp_add(train_shape_converter['state_converter'], 
                              train_shape_converter['seq_cstate_converter'], 
                              encode_state=encode_state, 
                              encode_boundary=encode_boundary)

  def active_dp_add(self, state_converter, seq_state_converter):
    if self.dataset == "jhtdb":
      self.data_queue.add_rand_dp(self.nr_data_to_add, 
                                  state_converter, 
                                  seq_state_converter)
    elif self.dataset == "sailfish": #TODO fix this into new code base
      loss_data_pair = []
      ratio_add = 5
      for i in tqdm(range(self.nr_data_to_add*ratio_add)):
        sim_ind, dp, feed_dict = self.data_queue.rand_dp(
                                         state_converter, 
                                         seq_state_converter)
        loss_names = ['l2_loss']
        loss_output = self.run(loss_names, feed_dict=feed_dict, return_dict=True)
        loss_data_pair.append((loss_output['l2_loss'], sim_ind, dp))
      loss_data_pair.sort()
      sim_list = [x[1] for x in loss_data_pair[-self.nr_data_to_add:]] 
      dp_list  = [x[2] for x in loss_data_pair[-self.nr_data_to_add:]] 
      self.data_queue.add_dps(sim_list, dp_list)

  def active_cdp_add(self, state_converter, seq_cstate_converter, encode_state, encode_boundary):
    if self.dataset == "JHTDB":
      self.data_queue.add_rand_cdp(self.nr_data_to_add, 
                                   state_converter, 
                                   seq_cstate_converter, 
                                   encode_state=encode_state, 
                                   encode_boundary=encode_boundary)
    elif self.dataset == "sailfish": #TODO fix this into new code base
      loss_data_pair = []
      ratio_add = 5
      for i in tqdm(range(self.nr_data_to_add*ratio_add)):
        sim_ind, cdp, feed_dict = self.data_queue.rand_cdp(
                                         state_converter, 
                                         seq_cstate_converter,
                                         encode_state,
                                         encode_boundary)
        loss_names = ['l2_loss']
        loss_output = self.run(loss_names, feed_dict=feed_dict, return_dict=True)
        loss_data_pair.append((loss_output['l2_loss'], sim_ind, cdp))
      loss_data_pair.sort()
      sim_list = [x[1] for x in loss_data_pair[-self.nr_data_to_add:]] 
      cdp_list  = [x[2] for x in loss_data_pair[-self.nr_data_to_add:]] 
      self.data_queue.add_cdps(sim_list, cdp_list)
 
  def update_loss_stats(self, output):
    names = list(output.keys())
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
        self.loss_stats[name] = output[name]
        # update ave loss
        self.loss_stats[name + '_ave'] = (np.sum(np.array(self.loss_stats[name + '_history']))
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
                                    (self.batch_size * len(self.gpus)))
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
    super(EvalLatNet, self).__init__(config)
    # needed configs
    self.sim_dir = config.sim_dir
    self.dataset = config.dataset
    self.sim_shape = self.domain.sim_shape
    self.sim_cshape = [x/pow(2,config.nr_downsamples) for x in self.sim_shape]
    self.eval_cshape = str2shape(config.eval_cshape)
    self.num_iters = config.num_iters
    self.sim_save_every = config.sim_save_every

    # initialize sim saver
    self.sim_saver = SimSaver(config)

    # initialize simulations
    self.domain.script_name = self.script_name
    self.sim = self.domain(config, self.sim_dir + '/' + self.domain.wrapper_name)
    if self.domain.wrapper_name == 'sailfish':
      self.sim.generate_data(1)

  @classmethod
  def add_options(cls, group):
    group.add_argument('--num_iters',
                   help='number of iterations to run network generated simulation',
                   type=int,
                   default=15)
    group.add_argument('--sim_restore_iter',
                   help='what iteration to start the simulation (0 if starting simulation from initial conditions)',
                   type=int,
                   default=0)
    group.add_argument('--eval_cshape',
                   help='shape of compresses state for domain decomposision',
                   type=str,
                   default='32x32')
    group.add_argument('--sim_save_every',
                   help=' save cstate every sim_save_every iters',
                   type=int,
                   default=4)

  def state_input_generator(self, subdomain):
    if self.start_state is not None:
      start_state, pad_start_state = mobius_extract_2(self.start_state, 
                                                      subdomain, 
                                                      has_batch=True, 
                                                      padding_type=self.sim.padding_type,
                                                      return_padding=True)
    else:
      vel = self._domain.velocity_initial_conditions(0,0,None)
      feq = self.DxQy.vel_to_feq(vel).reshape([1] + self.DxQy.dims*[1] + [self.DxQy.Q])
      start_state = np.zeros([1] + subdomain.size + [self.DxQy.Q]) + feq
      pad_start_state = np.zeros([1] + subdomain.size + [1])
    return {'state': [start_state, pad_start_state]}

  def boundary_input_generator(self, subdomain):
    if self.start_boundary is not None:
      input_boundary, pad_input_boundary = mobius_extract_2(self.start_boundary, 
                                                            subdomain,
                                                            has_batch=True, 
                                                            padding_type=self.sim.padding_type,
                                                            return_padding=True)
    else:
      input_boundary = self.input_boundary(subdomain)
      pad_input_boundary = np.zeros(subdomain.size + [1])
    return {'boundary': [input_boundary, pad_input_boundary]}

  def cstate_cboundary_input_generator(self, subdomain):
      sub_cstate =    mobius_extract_2(self.cstate,
                                       subdomain, 
                                       has_batch=True, 
                                       padding_type=self.sim.padding_type,
                                       return_padding=True)
      sub_cboundary = mobius_extract_2(self.cboundary,
                                       subdomain, 
                                       has_batch=True, 
                                       padding_type=self.sim.padding_type,
                                       return_padding=True)
      return {'cstate': sub_cstate, 'cboundary': sub_cboundary}

  def eval(self):
    # unroll network
    with tf.Graph().as_default():
      encode_state, state_shape_converter = self.unroll_state_encoder()
      encode_boundary, boundary_shape_converter = self.unroll_boundary_encoder()
      compression_mapping, cstate_shape_converter = self.unroll_compression_mapping()
      state_shape_converter = state_shape_converter['state_converter']
      boundary_shape_converter = boundary_shape_converter['boundary_converter']
      cstate_shape_converter = cstate_shape_converter['cstate_converter']
      self.start_session()
      self.load_checkpoint() 
   
    # get start state and boundary of simulation
    print("Computing compressed state") 
    self.start_state = self.sim.read_state(1, subdomain=None, add_batch=True, return_padding=False)
    self.cstate = self.mapping(mapping=encode_state,
                               shape_converter=state_shape_converter, 
                               input_generator=self.state_input_generator,
                               output_shape=self.sim_cshape,
                               run_output_shape=self.eval_cshape)
    self.start_state = None
    print("Computing compressed boundary") 
    self.start_boundary = self.sim.read_boundary(subdomain=None, add_batch=True, return_padding=False)
    self.cboundary = self.mapping(encode_boundary, boundary_shape_converter, 
                         self.boundary_input_generator, self.sim_cshape,
                         self.eval_cshape)
    self.start_boundary = None

    print("Running compressed simulation") 
    for i in tqdm(range(self.num_iters)):
      self.cstate = self.mapping(compression_mapping, cstate_shape_converter, 
                       self.cstate_cboundary_input_generator, self.sim_cshape,
                       self.eval_cshape)
      if i % self.sim_save_every:
        self.sim_saver.save(i, self.cstate)

  def mapping(self, mapping, shape_converter, input_generator, output_shape, run_output_shape):
    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(output_shape, run_output_shape)]
    output = []
    iter_list = [range(x) for x in nr_subdomains]
    for ijk in itertools.product(*iter_list):
      #print(str(ijk) + " out of " + str(nr_subdomains))
      # make input and output subdomains
      if not(type(shape_converter) is list):
        shape_converter = [shape_converter]
      input_subdomain = []
      output_subdomain = []
      for converter in shape_converter:
        pos = [x * y for x, y in zip(ijk, run_output_shape)]
        subdomain = SubDomain(pos, run_output_shape)
        input_subdomain.append(converter.out_in_subdomain(copy(subdomain)))
        output_subdomain.append(converter.in_out_subdomain(copy(input_subdomain[-1])))
      output_subdomain = output_subdomain[0]
      output_subdomain.pos  = [y - x for x, y in zip(output_subdomain.pos, pos)]
      output_subdomain.size = run_output_shape

      # generate input with input generator
      sub_input = input_generator(*input_subdomain)

      # perform mapping function and extract out if needed
      if not (type(sub_input) is list):
        sub_input = [sub_input]
      for j in range(1):
        sub_output = mapping(*sub_input)
      if not (type(sub_output) is list):
        sub_output = [sub_output]

      for i in range(len(sub_output)):
        sub_output[i] = mobius_extract_2(sub_output[i], 
                                         output_subdomain, 
                                         has_batch=True)

      # append to list of sub outputs
      output.append(sub_output)

    # make total output shape
    total_subdomain = SubDomain(len(output_shape)*[0], output_shape)
    ctotal_subdomain = shape_converter[0].out_in_subdomain(copy(total_subdomain))
    total_subdomain  = shape_converter[0].in_out_subdomain(copy(ctotal_subdomain))
    total_subdomain = SubDomain([-x for x in total_subdomain.pos], output_shape)

    # stack back together to form one output
    output = llist2list(output)
    for i in range(len(output)):
      output[i] = stack_grid(output[i],
                             nr_subdomains,
                             has_batch=True)
      output[i] = mobius_extract_2(output[i],
                                   total_subdomain, 
                                   has_batch=True)
    if len(output) == 1:
      output = output[0]
    return output

class DecodeLatNet(EvalLatNet):

  def __init__(self, config):
    super(DecodeLatNet, self).__init__(config)
    # needed configs
    self.compare = config.compare

    # restart simulations
    """
    if self.compare:
      self.sim = self.domain(config, self.sim_dir + '/' + self.domain.wrapper_name)
      if self.domain.wrapper_name == 'sailfish':
        self.sim.generate_data(1)
    """

  @classmethod
  def add_options(cls, group):
    group.add_argument('--compare',
                   help='generate true data to compare too',
                   type=str2bool,
                   default=True)

  def state_subdomain_to_state(self, subdomain, shape_converter, cstate, decode_mapping):
    cstate_subdomain = shape_converter.out_in_subdomain(copy(subdomain))
    sub_cstate = mobius_extract_2(cstate,
                                  cstate_subdomain, 
                                  has_batch=True, 
                                  padding_type=self.sim.padding_type,
                                  return_padding=True)
    sub_cstate = {'cstate': sub_cstate}
    sub_state = decode_mapping(sub_cstate)
    #print(sub_state.keys())
    return sub_state

  def decode(self):
    # unroll network
    with tf.Graph().as_default():
      decode_mapping, decode_shape_converter = self.unroll_decode_state()
      self.start_session()
      self.load_checkpoint() 
   
    print("Decompressing ") 
    for i in tqdm(range(self.num_iters)):
      if i % self.sim_save_every:
        cstate = self.sim_saver.load(i)
        for subdomain in self.decode_subdomains:
          sub_state = self.state_subdomain_to_state(subdomain,
                                 decode_shape_converter['state_converter'],
                                 cstate, decode_mapping)
          self.sub_state_computation(sub_state, subdomain, i)

  def sub_state_computation(self, sub_state, subdomain, iteration):
    print("wrong")
    pass

















