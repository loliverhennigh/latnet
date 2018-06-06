

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

class Trainer(object):
  # default network name
  #network_name = 'advanced_network'

  def __init__(self, config):
    # in and out tensors
    self.config = config # TODO remove this when config is corrected
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.network_dir  = config.latnet_network_dir
    self.seq_length = config.seq_length
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

  @classmethod
  def add_options(cls, group, network_name):
    pass

  def init_network(self):
    self._network = self.network(self.config)
    self._network.train_unroll()

  def make_data_queue(self):
    # add script name to domains
    for domain in self.domains:
      if domain is not None:
        domain.script_name = self.script_name
    self.data_queue_train = DataQueue(self.config, 
                                      self.config.train_sim_dir + '/train',
                                      self.domains, 
                                      self._network.train_shape_converter(), 
                                      self.config.start_num_data_points_train)
    self.data_queue_test = DataQueue(self.config, 
                                     self.config.train_sim_dir + '/test' , 
                                     self.domains, 
                                     self._network.train_shape_converter(), 
                                     self.config.start_num_data_points_test)

  def train(self):
 
    # steps per print (hard set for now untill done debugging)
    steps_per_print = 20

    while True: 
      # get batch of data
      feed_dict = self.data_queue_train.minibatch()
      feed_dict['phase'] = 1

      # perform optimization step for gen
      gen_names = ['gen_train_op', 'loss_gen']
      if not self.gan:
        gen_names += ['loss_l2']
      if self.gan:
        gen_names += ['loss_l1', 'loss_gen_un_class', 'loss_layer_l2', 'loss_gen_con_class']
      gen_output = self._network.run(gen_names, feed_dict=feed_dict, return_dict=True)
      if self.gan:
        disc_names = ['disc_train_op', 'loss_disc', 'loss_disc_un_class', 'loss_disc_con_class']
        disc_output = self._network.run(disc_names, feed_dict=feed_dict, return_dict=True)
        gen_output.update(disc_output)
         
      # update loss summary
      self.update_loss_stats(gen_output)

      # update time summary
      self.update_time_stats()

      # print required data and save
      step = self._network.run('gen_global_step')
      if step % steps_per_print == 0:
        self.print_stats(self.loss_stats, self.time_stats, self.data_queue_train.queue_stats(), step)
        # TODO integrat this into self.saver
        tf_feed_dict = {}
        for name in feed_dict.keys():
          if type(feed_dict[name]) is tuple:
            tf_feed_dict[self._network.in_tensors[name]] = feed_dict[name][0]
            tf_feed_dict[self._network.in_pad_tensors[name]] = feed_dict[name][1]
          else:
            tf_feed_dict[self._network.in_tensors[name]] = feed_dict[name]
        ###
        self._network.saver.save_summary(self._network.sess, tf_feed_dict, int(self._network.run('gen_global_step')))

      if step % self.config.save_network_freq == 0:
        self._network.saver.save_checkpoint(self._network.sess, int(self._network.run('gen_global_step')))

      #if step % 2000 == 0:
      #  print("getting new data")
      #  self.active_data_add()

      # test data
      if step % 1000 == 0:
        feed_dict = self.data_queue_test.minibatch()
        feed_dict['phase'] = 1
        seq_str = lambda x: '_' + str(x) + '_gpu_0'
        gen_names = []
        for i in xrange(self.seq_length):
          gen_names.append('pred_state' + seq_str(i))
        gen_output = self._network.run(gen_names, feed_dict=feed_dict, return_dict=True)
        for key in feed_dict.keys():
          if 'true' not in key:
            feed_dict.pop(key)
        self.save_test_sample(feed_dict, gen_output)

      # end simulation
      if step > self.train_iters:
        break

  def save_test_sample(self, true_dict, gen_dict):
    save_dir = self.network_dir + '/snapshot/'
    try:
      os.makedirs(save_dir)
    except:
      pass
    for key in true_dict.keys():
      np.save(save_dir + key, true_dict[key])
      #true_dict[key]
      # probably make method for this or remove the image save
      plt.imshow(self.DxQy.lattice_to_norm(true_dict[key])[0,0,...,0])
      plt.savefig(save_dir + key + 'x.png')
      plt.imshow(self.DxQy.lattice_to_norm(true_dict[key])[0,:,0,:,0])
      plt.savefig(save_dir + key + 'y.png')
      plt.imshow(self.DxQy.lattice_to_norm(true_dict[key])[0,...,0,0])
      plt.savefig(save_dir + key + 'z.png')
    for key in gen_dict.keys():
      np.save(save_dir + key, gen_dict[key])
      # probably make method for this or remove the image save
      plt.imshow(self.DxQy.lattice_to_norm(gen_dict[key])[0,0,...,0])
      plt.savefig(save_dir + key + 'x.png')
      plt.imshow(self.DxQy.lattice_to_norm(gen_dict[key])[0,:,0,:,0])
      plt.savefig(save_dir + key + 'y.png')
      plt.imshow(self.DxQy.lattice_to_norm(gen_dict[key])[0,...,0,0])
      plt.savefig(save_dir + key + 'z.png')

  def active_data_add(self):
    # TODO this should be cleaned up
    loss_data_point_pair = []
    for i in tqdm(xrange(1000)):
      sim_index, data_point, feed_dict = self.data_queue_train.rand_data_point()
      loss_names = ['loss_l2']
      loss_output = self._network.run(loss_names, feed_dict=feed_dict, return_dict=True)
      loss_data_point_pair.append((loss_output['loss_l2'], sim_index, data_point))

    loss_data_point_pair.sort() 
    for i in xrange(100):
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

