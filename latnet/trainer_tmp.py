

import time
from copy import copy
import os
from termcolor import colored, cprint
import tensorflow as tf
import numpy as np
from tqdm import *

import lattice
import nn as nn

import matplotlib.pyplot as plt
from shape_converter import ShapeConverter
from optimizer import Optimizer
from shape_converter import SubDomain
from network_saver import NetworkSaver
from data_queue import DataQueue
from utils.python_utils import *

class Trainer(LatNet):
  # default network name
  #network_name = 'advanced_network'

  def __init__(self, config, network_arch):
    super(LatNet, self).__init__(config, network_arch)

    self.train_mode = config.train_mode
    self.dataset = config.dataset

  @classmethod
  def add_options(cls, group, network_name):
    pass

  def make_data_queue(self):
    # add script name to domains TODO This is a little weird and might be taken out later
    for domain in self.domains:
      if domain is not None:
        domain.script_name = self.script_name

    data_queue = DataQueue(self.config, 
                           self.config.train_sim_dir,
                           self.domains, 
                           self.train_shape_converter())
 
  def train(self):
    # unroll network
    with tf.Graph().as_default():
      save_sumary = self.unroll_summary()
      save_sumary = self.unroll_summary()
      if self.train_mode == "full": 
         
        get_step, train_step, save_summary = self.unroll_full()
      elif self.train_mode == "compression":
        compress_state, compress_boundary, get_step, train_step, save_summary = self.unroll_compression()

    # make data queue
    data_queue = self.make_data_queue()

    # train
    while True:
      step = get_step()
      if self.train_mode == "full":
        feed_dict = data_queue.dp_minibatch()
      elif self.train_mode == "compression":
        feed_dict = data_queue.cdp_minibatch()

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
        self.saver.save_checkpoint(self.sess, int(step))
    
      # possibly get more data
      if step % 200 == 0:
        self.active_data_add()
  

  def train_full(self):
    # make train step and saver summary
    data_queue = self.make_data_queue_full()

    while True:
      self.train_step(get_step, feed_dict, train_step, dave_summary, data_queue)
 
  def train_compression(self):
    # make train step and saver summary
    with tf.Graph().as_default():
      get_step, train_step, save_summary = self.unroll_full()
    self.make_data_queue_full()

    while True:

    self.loop_train_step(get_step, train_step, save_summary)

  def train_step(self, step, feed_dict, train_step, data_queue):
    # run train operation 
    output = train_step(feed_dict)
          
    # update loss summary
    self.update_loss_stats(output)
  
    # update time summary
    self.update_time_stats()
  
    # print required data and save
    if step % steps_per_print == 0:
      self.print_stats(self.loss_stats, self.time_stats, data_queue_train.queue_stats(), step)

    # save in tensorboard 
    if step % 100 == 0 and self.train_autoencoder:
      save_summary(feed_dict)
  
    # save in network
    if step % self.config.save_network_freq == 0:
      self.saver.save_checkpoint(self.sess, int(step))
  
    # possibly get more data
    if step % 200 == 0:
      self.active_data_add()

  def active_data_add(self):
    # TODO this should be cleaned up
    if self.dataset == "JHTDB":
      self.data_queue_train.add_rand_dps(50)
    else:
      loss_data_point_pair = []
      for i in tqdm(xrange(200)):
        sim_index, data_point, feed_dict = self.data_queue_train.rand_data_point()
        loss_names = ['loss_gen']
        loss_output = self._network.run(loss_names, feed_dict=feed_dict, return_dict=True)
        loss_data_point_pair.append((loss_output['loss_gen'], sim_index, data_point))
  
      loss_data_point_pair.sort() 
      for i in xrange(40):
        self.data_queue_train.add_data_point(loss_data_point_pair[-i][1], loss_data_point_pair[-i][2])

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

