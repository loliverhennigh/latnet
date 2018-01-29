
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
    self.free_gpu        = True

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
      sim = TrainSailfishRunner(config, self.base_dir + 'sim_' + str(i), self.script_name) 
      sim.generate_train_data()
      thr = threading.Thread(target= (lambda: self.data_worker(sim)))
      thr.daemon = True
      thr.start()

  def data_worker(self, sim):
    while True:
      self.queue.get()

      # select random piece to grab from data
      rand_pos = [np.random.randint(0, self.sim_shape[0]), np.random.randint(0, self.sim_shape[1])]
      state_subdomain = SubDomain(rand_pos, self.input_shape)
      geometry_subdomain = SubDomain(rand_pos, self.input_shape)
      seq_state_subdomain = []
      for i in xrange(self.seq_length):
        seq_state_subdomain.append(self.shape_converters['state', 'pred_state_' + str(i)].in_out_subdomain(copy(state_subdomain)))

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
    while len(self.queue_batches) < 2*self.batch_size: # added times two to make sure enough
      print("spending time waiting for queue")
      time.sleep(1.01)

    # generate batch of data in the form of a feed dict
    batch_state = []
    batch_geometry = []
    batch_seq_state = []
    for i in xrange(self.batch_size): 
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
    #feed_dict['state'] = np.zeros_like(batch_state)
    #feed_dict['state'] = np.zeros_like(batch_state)
    feed_dict['state'] = batch_state
    #feed_dict['boundary'] = np.zeros_like(batch_geometry)
    feed_dict['boundary'] = batch_geometry
    for i in xrange(self.seq_length):
      #feed_dict['true_state_' + str(i)] = np.zeros_like(batch_seq_state[i])
      #feed_dict['true_state_' + str(i)] = np.zeros_like(batch_seq_state[i]) + 1.0
      feed_dict['true_state_' + str(i)] = batch_seq_state[i]
      
    return feed_dict

