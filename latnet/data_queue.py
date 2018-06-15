
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
import glob
from tqdm import *
import sys
import os.path
import gc
#import skfmm
import time
import psutil as ps
import shutil
from copy import copy
from sailfish_simulation import TrainSailfishSimulation
from utils.python_utils import *

from Queue import Queue
from shape_converter import SubDomain
import lattice
import threading
from utils.text_histogram import vector_to_text_hist

class DataQueue:
  def __init__(self, config, domains, shape_converters):

    # base dir where all the xml files are
    self.base_dir = config.train_sim_dir
    self.waiting_time = 0.0
    self.needed_to_wait = False

    # configs
    self.batch_size      = config.batch_size
    self.seq_length      = config.seq_length
    self.train_autoencoder = config.train_autoencoder
    self.nr_downsamples  = config.nr_downsamples
    self.free_gpu        = True
    gpus = config.gpus.split(',')
    self.gpus = map(int, gpus)
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.input_cshape = str2shape(config.input_cshape)

    # shape converter
    self.shape_converters = shape_converters
    gpu_str = '_gpu_' + str(self.gpus[0])
    self.state_shape_converter = self.shape_converters['state' + gpu_str,
                            'cstate_' + str(self.seq_length-1) + gpu_str]
    if self.train_autoencoder:
      self.seq_state_shape_converter = self.shape_converters['state' + gpu_str, 
                              'pred_state_' + str(self.seq_length-1) + gpu_str]
    else:
      self.seq_state_shape_converter = self.shape_converters['true_state_' + str(self.seq_length-1) + gpu_str, 
                              'true_cstate_' + str(self.seq_length-1) + gpu_str]
    self.cratio = pow(2, self.nr_downsamples)

    # make queue
    self.max_queue = config.max_queue
    self.queue = Queue() # to stop halting when putting on the queue
    self.queue_batches = []

    # generate base dataset and start queues
    self.sim_runners = []
    for domain in domains:
      print("generating " + domain.name + " dataset")
      start_num_data_points = 1000/(len(domains)*domain.num_simulations)
      for i in tqdm(xrange(domain.num_simulations)):
        sim = TrainSailfishSimulation(config, domain, self.base_dir + '/sim_' + domain.name + '_' + str(i).zfill(4))
        if sim.need_to_generate():
          sim.generate_train_data()
        sim.make_rand_data_points(start_num_data_points, 
                                 seq_length=self.seq_length,
                                 state_shape_converter=self.state_shape_converter, 
                                 seq_state_shape_converter=self.seq_state_shape_converter,
                                 input_cshape=self.input_cshape,
                                 cratio=self.cratio)
        self.sim_runners.append(sim)
        thr = threading.Thread(target= (lambda: self.data_worker(sim)))
        thr.daemon = True
        thr.start()

  def data_worker(self, sim):
    while True:
      self.queue.get()

      # get geometry and lat data
      state, geometry, seq_state = sim.read_train_data()

      # add to que
      self.queue_batches.append((state, geometry, seq_state))
      self.queue.task_done()

  def rand_data_point(self):
    sim_index = np.random.randint(0, len(self.sim_runners))
    data_point = self.sim_runners[sim_index].rand_data_point(
                                 seq_length=self.seq_length,
                                 state_shape_converter=self.state_shape_converter, 
                                 seq_state_shape_converter=self.seq_state_shape_converter,
                                 input_cshape=self.input_cshape,
                                 cratio=self.cratio)
    state, geometry, seq_state = self.sim_runners[sim_index].data_point_to_data(data_point, add_batch=True)

    # make feed dict
    feed_dict = {}
    gpu_str = '_gpu_' + str(self.gpus[0])
    feed_dict['state' + gpu_str] = state
    feed_dict['boundary' + gpu_str] = geometry
    for j in xrange(self.seq_length):
      feed_dict['true_state_' + str(j) + gpu_str] = seq_state[j]

    return sim_index, data_point, feed_dict

  def add_data_point(self, sim_index, data_point):
    self.sim_runners[sim_index].add_data_point(data_point)
 
  def minibatch(self):

    # queue up data if needed
    for i in xrange(self.max_queue - len(self.queue_batches) - self.queue.qsize()):
      self.queue.put(None)
   
    # possibly wait if data needs time to queue up
    while len(self.queue_batches) < 2*self.batch_size*len(self.gpus): # added times two to make sure enough
      self.waiting_time += 1.0
      self.needed_to_wait = True
      time.sleep(1.0)

    # generate batch of data in the form of a feed dict
    batch_state = []
    batch_pad_state = []
    batch_geometry = []
    batch_pad_geometry = []
    batch_seq_state = []
    batch_pad_seq_state = []
    for i in xrange(self.batch_size*len(self.gpus)): 
      batch_state.append(self.queue_batches[0][0][0])
      batch_pad_state.append(self.queue_batches[0][0][1])
      batch_geometry.append(self.queue_batches[0][1][0])
      batch_pad_geometry.append(self.queue_batches[0][1][1])
      batch_seq_state.append([x[0] for x in self.queue_batches[0][2]])
      batch_pad_seq_state.append([x[1] for x in self.queue_batches[0][2]])
      self.queue_batches.pop(0)

    # concate batches together
    batch_state = np.stack(batch_state, axis=0)
    batch_pad_state = np.stack(batch_pad_state, axis=0)
    batch_geometry = np.stack(batch_geometry, axis=0)
    batch_pad_geometry = np.stack(batch_pad_geometry, axis=0)
    new_batch_seq_state = []
    new_batch_pad_seq_state = []
    for i in xrange(self.seq_length):
      new_batch_seq_state.append(np.stack([x[i] for x in batch_seq_state], axis=0))
      new_batch_pad_seq_state.append(np.stack([x[i] for x in batch_pad_seq_state], axis=0))
    batch_seq_state = new_batch_seq_state
    batch_pad_seq_state = new_batch_pad_seq_state

    # make feed dict
    feed_dict = {}
    for i in xrange(len(self.gpus)):
      gpu_str = '_gpu_' + str(self.gpus[i])
      feed_dict['state' + gpu_str] = (batch_state[i*self.batch_size:(i+1)*self.batch_size],
                                      batch_pad_state[i*self.batch_size:(i+1)*self.batch_size])
      feed_dict['boundary' + gpu_str] = (batch_geometry[i*self.batch_size:(i+1)*self.batch_size],
                                         batch_pad_geometry[i*self.batch_size:(i+1)*self.batch_size])
      for j in xrange(self.seq_length):
        feed_dict['true_state_' + str(j) + gpu_str] = (batch_seq_state[j][i*self.batch_size:(i+1)*self.batch_size],
                            batch_pad_seq_state[j][i*self.batch_size:(i+1)*self.batch_size])

    return feed_dict

  def num_data_points(self):
    num_points = 0
    for i in xrange(len(self.sim_runners)):
      num_points += len(self.sim_runners[i].data_points)
    return num_points

  def ind_histogram(self):
    inds = []
    for i in xrange(len(self.sim_runners)):
      for j in xrange(len(self.sim_runners[i].data_points)):
        inds.append(self.sim_runners[i].data_points[j].ind)
    inds = np.array(inds)
    return vector_to_text_hist(inds, bins=10)

  def queue_stats(self):
    stats = {}
    stats['percent full'] = int(100*float(len(self.queue_batches))/float(self.max_queue))
    stats['total_wait_time'] = int(self.waiting_time)
    stats['queue_waited'] = self.needed_to_wait
    stats['num_data_points'] = self.num_data_points()
    stats['ind_histogram'] = self.ind_histogram()
    if len(self.queue_batches) > 1:
      stats['input_shape'] = self.queue_batches[0][0][0].shape
      stats['output_shape'] = self.queue_batches[0][2][0][0].shape
    self.needed_to_wait = False
    return stats

