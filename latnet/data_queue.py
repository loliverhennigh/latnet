
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
from sailfish_runner import TrainSailfishRunner

from Queue import Queue
from shape_converter import SubDomain
import lattice
import threading

class DataQueue:
  def __init__(self, config, train_sim, shape_converters):

    # base dir where all the xml files are
    self.base_dir = config.train_sim_dir
    self.script_name = train_sim.script_name
    self.waiting_time = 0.0
    self.needed_to_wait = False

    # configs
    self.batch_size      = config.batch_size
    self.num_simulations = config.num_simulations
    self.seq_length      = config.seq_length
    self.nr_downsamples  = config.nr_downsamples
    self.free_gpu        = True
    gpus = config.gpus.split(',')
    self.gpus = map(int, gpus)
    self.DxQy = lattice.TYPES[config.DxQy]()

    # shape
    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape

    input_cshape = config.input_cshape.split('x')
    input_cshape = map(int, input_cshape)
    self.input_cshape = input_cshape
    self.shape_converters = shape_converters

    # make queue
    self.max_queue = config.max_queue
    self.queue = Queue() # to stop halting when putting on the queue
    self.queue_batches = []

    # generate base dataset and start queues
    self.sim_runners = []
    print("generating dataset")
    for i in tqdm(xrange(self.num_simulations)):
      sim = TrainSailfishRunner(config, self.base_dir + '/sim_' + str(i), self.script_name) 
      if sim.need_to_generate():
        sim.generate_train_data()
      thr = threading.Thread(target= (lambda: self.data_worker(sim)))
      thr.daemon = True
      thr.start()

  def data_worker(self, sim):
    while True:
      self.queue.get()

      # select random piece to grab from data
      cratio = pow(2, self.nr_downsamples)
      rand_pos = [np.random.randint(-self.input_cshape[0], self.sim_shape[0]/cratio),
                  np.random.randint(-self.input_cshape[1], self.sim_shape[1]/cratio)]
      #rand_pos = [-self.input_cshape[0],-self.input_cshape[1]]
      cstate_subdomain = SubDomain(rand_pos, self.input_cshape)

      # get state subdomain and geometry_subdomain
      gpu_str = '_gpu_' + str(self.gpus[0])
      state_shape_converter = self.shape_converters['state' + gpu_str,
                                                   'cstate_' + str(self.seq_length-1) + gpu_str]
      geometry_subdomain = state_shape_converter.out_in_subdomain(copy(cstate_subdomain))
      state_subdomain = copy(geometry_subdomain)

      # get seq state subdomain
      seq_state_shape_converter = self.shape_converters['state' + gpu_str, 
                                                   'pred_state_' + str(self.seq_length-1) + gpu_str]
      seq_state_subdomain = seq_state_shape_converter.in_out_subdomain(copy(state_subdomain))
      geometry_small_subdomain = copy(seq_state_subdomain)
      geometry_small_subdomain.add_edges(2)

      # get geometry and lat data
      state, geometry, geometry_small, seq_state = sim.read_train_data(state_subdomain,
                                                             geometry_subdomain,
                                                             geometry_small_subdomain,
                                                             seq_state_subdomain,
                                                             self.seq_length)

      # add to que
      self.queue_batches.append((state, geometry, geometry_small, seq_state))
      self.queue.task_done()
 
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
    batch_geometry = []
    batch_geometry_small = []
    batch_seq_state = []
    for i in xrange(self.batch_size*len(self.gpus)): 
      batch_state.append(self.queue_batches[0][0])
      batch_geometry.append(self.queue_batches[0][1])
      batch_geometry_small.append(self.queue_batches[0][2])
      batch_seq_state.append(self.queue_batches[0][3])
      self.queue_batches.pop(0)

    # concate batches together
    batch_state = np.stack(batch_state, axis=0)
    batch_geometry = np.stack(batch_geometry, axis=0)
    batch_geometry_small = np.stack(batch_geometry_small, axis=0)
    new_batch_seq_state = []
    for i in xrange(self.seq_length):
      new_batch_seq_state.append(np.stack([x[i] for x in batch_seq_state], axis=0))
    batch_seq_state = new_batch_seq_state

    # make feed dict
    feed_dict = {}
    for i in xrange(len(self.gpus)):
      gpu_str = '_gpu_' + str(self.gpus[i])
      feed_dict['state' + gpu_str] = batch_state[i*self.batch_size:(i+1)*self.batch_size]
      feed_dict['boundary' + gpu_str] = batch_geometry[i*self.batch_size:(i+1)*self.batch_size]
      feed_dict['boundary_small' + gpu_str] = batch_geometry_small[i*self.batch_size:(i+1)*self.batch_size]
      for j in xrange(self.seq_length):
        feed_dict['true_state_' + str(j) + gpu_str] = batch_seq_state[j][i*self.batch_size:(i+1)*self.batch_size]
      
    return feed_dict

  def queue_stats(self):
    stats = {}
    stats['percent full'] = int(100*float(len(self.queue_batches))/float(self.max_queue))
    stats['total_wait_time'] = int(self.waiting_time)
    stats['queue_waited'] = self.needed_to_wait
    if len(self.queue_batches) > 1:
      stats['input_shape'] = self.queue_batches[0][0].shape
      stats['output_shape'] = self.queue_batches[0][3][0].shape
    self.needed_to_wait = False
    return stats

