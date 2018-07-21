
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
from john_hopkins_simulation import TrainJohnHopkinsSimulation
from utils.python_utils import *
from utils.numpy_utils import *

from Queue import Queue
from shape_converter import SubDomain
import lattice
import threading
from utils.text_histogram import vector_to_text_hist

class DataQueue(object):
  def __init__(self, config, base_dir, domains, shape_converters, mapping=None):

    # base dir where all the xml files are
    self.base_dir = base_dir

    # values for queue stats
    self.waiting_time = 0.0
    self.needed_to_wait = False

    # configs
    self.max_queue       = self.max_queue
    self.dataset         = config.dataset
    self.batch_size      = config.batch_size
    self.seq_length      = config.seq_length
    self.nr_downsamples  = config.nr_downsamples
    self.gpus            = map(int, config.gpus.split(','))
    self.DxQy            = lattice.TYPES[config.DxQy]()
    self.input_cshape    = str2shape(config.input_cshape)

    # shape converter
    self.shape_converters = shape_converters
    gpu_str = '_gpu_' + str(self.gpus[0])
    self.state_shape_converter = self.shape_converters['state' + gpu_str,
                            'cstate_' + str(self.seq_length-1) + gpu_str]
    self.seq_state_shape_converter = self.shape_converters['true_state_' + str(self.seq_length-1) + gpu_str, 
                            'true_cstate_' + str(self.seq_length-1) + gpu_str]
    self.cratio = pow(2, self.nr_downsamples)
    self.mapping = mapping

  def init_sims(self, domains):
    sims = []

    # generate base dataset and start queues
    for domain in self.domains:
      for i in tqdm(xrange(domain.num_simulations)):
        save_dir = self.base_dir + '/sim_' + domain.name + '_' + str(i).zfill(4)
        sim = domain(config, save_dir)
        sims.append(sim)
    return sims

  def add_rand_dp(self, sims, num_dps):
    dps_per_sim = int(num_dps/len(sims))
    for sim in sims:
      sim.add_rand_dps(num_dps=dps_per_sim,
                       seq_length=self.seq_length,
                       state_shape_converter=self.state_shape_converter, 
                       seq_state_shape_converter=self.seq_state_shape_converter,
                       input_cshape=self.input_cshape,
                       cratio=self.cratio)

  def add_rand_cdp(self, sims, num_cdps, mapping):
    cdps_per_sim = int(num_dps/len(sims))
    for sim in sims:
      sim.add_rand_dps(num_cdps=cdps_per_sim,
                       seq_length=self.seq_length,
                       state_shape_converter=self.state_shape_converter, 
                       seq_state_shape_converter=self.seq_state_shape_converter,
                       input_cshape=self.input_cshape,
                       cratio=self.cratio,
                       mapping=mapping)

  def start_dp_queue(self, doms):
    # queue
    queue_dp = Queue()
    queue_dp_batches = []
    for dom in doms:
      thr = threading.Thread(target= (lambda: self.dp_worker(dom)))
      thr.daemon = True
      thr.start()
    return queue_dp, queue_dp_batches

  def start_cdp_queue(self, doms):
    # queue
    queue_cdp = Queue()
    queue_cdp_batches = []
    for dom in doms:
      thr = threading.Thread(target= (lambda: self.cdp_worker(dom)))
      thr.daemon = True
      thr.start()
    return queue_cdp, queue_cdp_batches

  def rand_dp(self):
    index = np.random.randint(0, len(self.sims))
    dp = self.sims[index].rand_dp(
                 seq_length=self.seq_length,
                 state_shape_converter=self.state_shape_converter, 
                 seq_state_shape_converter=self.seq_state_shape_converter,
                 input_cshape=self.input_cshape,
                 cratio=self.cratio)
    state, boundary, seq_state = self.sims[index].read_dp(dp, add_batch=True)

    # make feed dict
    feed_dict = {}
    gpu_str = '_gpu_' + str(self.gpus[0])
    feed_dict['state' + gpu_str] = state
    feed_dict['boundary' + gpu_str] = boundary
    for j in xrange(self.seq_length):
      feed_dict['true_state_' + str(j) + gpu_str] = seq_state[j]

    return index, data_point, feed_dict

  def rand_cdp(self, mapping):
    index = np.random.randint(0, len(self.sims))
    cdp = self.sims[index].rand_cdp(
                 seq_length=self.seq_length,
                 state_shape_converter=self.state_shape_converter, 
                 seq_state_shape_converter=self.seq_state_shape_converter,
                 input_cshape=self.input_cshape,
                 cratio=self.cratio,
                 mapping=mapping)
    cstate, cboundary, seq_cstate = self.sims[index].read_cdp(cdp, add_batch=True)

    # make feed dict
    feed_dict = {}
    gpu_str = '_gpu_' + str(self.gpus[0])
    feed_dict['cstate' + gpu_str] = cstate
    feed_dict['cboundary' + gpu_str] = cboundary 
    for j in xrange(self.seq_length):
      feed_dict['true_cstate_' + str(j) + gpu_str] = seq_cstate[j]

    return index, data_point, feed_dict

  def add_dp(self, sim_index, dp):
    self.sims[sim_index].add_dp(dp)
 
  def add_cdp(self, sim_index, cdp):
    self.sims[sim_index].add_cdp(cdp)

  def dps_minibatch(self):
    # queue up data if needed
    for i in xrange((self.max_queue/(len(self.gpus) * self.batch_size)) - (len(self.queue_dp_batches) + self.queue_dp.qsize())):
      self.queue_dp.put(None)
    while len(self.queue_dp_batches) <= 2: # added times two to make sure enough
      self.waiting_time += 1.0
      self.needed_to_wait = True
      time.sleep(1.0)
    feed_dict = self.queue_dp_batches.pop(0)
    return feed_dict

  def cdps_minibatch(self):
    # queue up data if needed
    for i in xrange((self.max_queue/(len(self.gpus) * self.batch_size)) - (len(self.queue_cdp_batches) + self.queue_cdp.qsize())):
      self.queue_cdp.put(None)
    while len(self.queue_cdp_batches) <= 2: # added times two to make sure enough
      self.waiting_time += 1.0
      self.needed_to_wait = True
      time.sleep(1.0)
    feed_dict = self.queue_cdp_batches.pop(0)
    return feed_dict

  def num_dps(self):
    num_points = 0
    for i in xrange(len(self.sim_runners)):
      num_points += len(self.sim_runners[i].data_points)
    return num_points

  def num_cdps(self):
    num_points = 0
    for i in xrange(len(self.sim_runners)):
      num_points += len(self.sim_runners[i].cdata_points)
    return num_points

  def dp_ind_histogram(self):
    inds = []
    for i in xrange(len(self.sim_runners)):
      for j in xrange(len(self.sim_runners[i].data_points)):
        inds.append(self.sim_runners[i].data_points[j].ind)
    inds = np.array(inds)
    return vector_to_text_hist(inds, bins=10)

  def dp_ind_histogram(self):
    inds = []
    for i in xrange(len(self.sim_runners)):
      for j in xrange(len(self.sim_runners[i].cdata_points)):
        inds.append(self.sim_runners[i].cdata_points[j].ind)
    inds = np.array(inds)
    return vector_to_text_hist(inds, bins=10)

  def queue_stats(self):
    stats = {}
    stats['percent full'] = int(100*float(len(self.gpus)*self.batch_size*len(self.queue_batches))/float(self.max_queue))
    stats['samples in queue'] = int(len(self.gpus) * self.batch_size * self.queue.qsize())
    stats['total_wait_time'] = int(self.waiting_time)
    stats['queue_waited'] = self.needed_to_wait
    stats['num_data_points'] = self.num_data_points()
    stats['ind_histogram'] = self.dp_ind_histogram()
    stats['ind_histogram'] = self.cdp_ind_histogram()
    if len(self.queue_batches) > 1:
      gpu_str = '_gpu_' + str(self.gpus[0])
      if self.train_autoencoder:
        stats['input_shape'] = self.queue_batches[0]['state' + gpu_str][0].shape
        stats['output_shape'] = self.queue_batches[0]['true_state_0' + gpu_str][0].shape
      else:
        stats['input_shape'] = self.queue_batches[0]['cstate' + gpu_str][0].shape
        stats['output_shape'] = self.queue_batches[0]['true_comp_cstate_0' + gpu_str][0].shape
    self.needed_to_wait = False
    return stats

  def dp_worker(self, sim):
    while True:
      self.queue.get()

      batch_state = []
      batch_boundary = []
      batch_seq_state = []
      for i in xrange(self.batch_size*len(self.gpus)):
        # get data
        state, boundary, seq_state = sim.read_state_data()

        # place into batches
        batch_state.append(state)
        batch_boundary.append(boundary)
        batch_seq_state.append(seq_state)

      # make feed dict
      feed_dict = {}
      for i in xrange(len(self.gpus)):
        gpu_str = '_gpu_' + str(self.gpus[i])
        feed_dict['state' + gpu_str]    = np_stack_list(batch_state[i*self.batch_size:(i+1)*self.batch_size])
        feed_dict['boundary' + gpu_str] = np_stack_list(batch_boundary[i*self.batch_size:(i+1)*self.batch_size])
        for j in xrange(self.seq_length):
          feed_dict['true_state_' + str(j) + gpu_str] = np_stack_llist(batch_seq_state[i*self.batch_size:(i+1)*self.batch_size][j])

      # add to que
      self.queue_batches.append(feed_dict)
      self.queue.task_done()

  def cdp_worker(self, sim):
    while True:
      self.queue.get()

      batch_cstate = []
      batch_cboundary = []
      batch_seq_cstate = []
      for i in xrange(self.batch_size*len(self.gpus)):
        # get data
        cstate, cboundary, seq_cstate = sim.read_cstate_data()

        # place into batches
        batch_cstate.append(cstate)
        batch_cboundary.append(cboundary)
        batch_seq_cstate.append(seq_cstate)

      # make feed dict
      feed_dict = {}
      for i in xrange(len(self.gpus)):
        gpu_str = '_gpu_' + str(self.gpus[i])
        feed_dict['cstate' + gpu_str]    = np_stack_list(batch_cstate[i*self.batch_size:(i+1)*self.batch_size])
        feed_dict['cboundary' + gpu_str] = np_stack_list(batch_cboundary[i*self.batch_size:(i+1)*self.batch_size])
        for j in xrange(self.seq_length):
          feed_dict['true_cstate_' + str(j) + gpu_str] = np_stack_llist(batch_seq_cstate[i*self.batch_size:(i+1)*self.batch_size][j])

      # add to que
      self.queue_cdp_batches.append(feed_dict)
      self.queue.task_done()


