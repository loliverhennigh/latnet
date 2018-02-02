
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
import threading

class DataQueue:
  def __init__(self, config, train_sim, shape_converters):

    # base dir where all the xml files are
    self.base_dir = config.train_sim_dir
    self.script_name = train_sim.script_name

    # configs
    self.batch_size      = config.batch_size
    self.num_simulations = config.num_simulations
    self.seq_length      = config.seq_length
    self.nr_downsamples  = config.nr_downsamples
    self.free_gpu        = True
    gpus = config.gpus.split(',')
    self.gpus = map(int, gpus)

    # shape
    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape
    input_shape = config.input_shape.split('x')
    input_shape = map(int, input_shape)
    self.input_shape = input_shape
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

      # edge padding to line up
      cstate_subdomain = SubDomain([0,0], [1,1])
      state_subdomain = self.shape_converters['state' + '_gpu_' + str(self.gpus[0]),
                                              'cstate_0_gpu_' + str(self.gpus[0])].out_in_subdomain(copy(cstate_subdomain)) 

      # select random piece to grab from data
      cratio = pow(2, self.nr_downsamples)
      rand_pos = [cratio * np.random.randint(0, self.sim_shape[0]/cratio), 
                  cratio * np.random.randint(0, self.sim_shape[1]/cratio)]
      rand_pos[0] = rand_pos[0] + state_subdomain.pos[0] # pad edges
      rand_pos[1] = rand_pos[1] + state_subdomain.pos[1]
      

      state_subdomain = SubDomain(rand_pos, self.input_shape)
      geometry_subdomain = SubDomain(rand_pos, self.input_shape)
      seq_state_subdomain = []
      for i in xrange(self.seq_length):
        seq_state_subdomain.append(self.shape_converters['state' + '_gpu_' + str(self.gpus[0]), 'pred_state_' + str(i) + '_gpu_' + str(self.gpus[0])].in_out_subdomain(copy(state_subdomain)))

      # get geometry and lat data
      state, geometry, seq_state = sim.read_train_data(state_subdomain,
                                                       geometry_subdomain,
                                                       seq_state_subdomain)

      # add to que
      self.queue_batches.append((state, geometry, seq_state))
      self.queue.task_done()
 
  def minibatch(self):

    # queue up data if needed
    for i in xrange(self.max_queue - len(self.queue_batches) - self.queue.qsize()):
      self.queue.put(None)
   
    # possibly wait if data needs time to queue up
    while len(self.queue_batches) < 2*self.batch_size*len(self.gpus): # added times two to make sure enough
      print("spending time waiting for queue")
      time.sleep(1.01)

    # generate batch of data in the form of a feed dict
    batch_state = []
    batch_geometry = []
    batch_seq_state = []
    for i in xrange(self.batch_size*len(self.gpus)): 
      batch_state.append(self.queue_batches[0][0])
      batch_geometry.append(self.queue_batches[0][1])
      batch_seq_state.append(self.queue_batches[0][2])
      self.queue_batches.pop(0)

    # concate batches together
    batch_state = np.stack(batch_state, axis=0)
    batch_geometry = np.stack(batch_geometry, axis=0)
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
      for j in xrange(self.seq_length):
        feed_dict['true_state_' + str(j) + gpu_str] = batch_seq_state[j][i*self.batch_size:(i+1)*self.batch_size]
      
    return feed_dict

  def queue_stats(self):
    stats = {}
    stats['percent_full'] = int(100*float(len(self.queue_batches))/float(self.max_queue))
    return stats

