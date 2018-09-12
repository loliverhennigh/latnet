
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import time
from queue import Queue
import threading

import lattice
from utils.text_histogram import vector_to_text_hist
from utils.python_utils import *
from utils.numpy_utils import *

class DataQueue(object):
  def __init__(self, config, domains):

    # base dir where all the xml files are
    self.base_dir = config.train_data_dir

    # values for queue stats
    self.waiting_time = 0.0
    self.needed_to_wait = False

    # configs
    self.max_queue       = config.max_queue
    self.dataset         = config.dataset
    self.batch_size      = config.batch_size
    self.seq_length      = config.seq_length
    self.nr_downsamples  = config.nr_downsamples
    self.gpus            = map(int, config.gpus.split(','))
    self.DxQy            = lattice.TYPES[config.DxQy]()
    self.train_cshape    = str2shape(config.train_cshape)
    self.cratio = pow(2, self.nr_downsamples)

    # initalize domains
    self.init_sims(config, domains)

    # start queues
    self.start_dp_queue()
    self.start_cdp_queue()

  @classmethod
  def add_options(cls, group):
    group.add_argument('--train_data_dir', 
                   help='where to save train data', 
                   type=str,
                   default='./train_data')
    group.add_argument('--start_num_dps', 
                   help='num datapoint before starting training', 
                   type=int,
                   default=200)
    group.add_argument('--max_queue', 
                   help='max datapoints in queue', 
                   type=int,
                   default=200)

  def init_sims(self, config, domains):
    sims = []

    # generate base dataset and start queues
    for domain in domains:
      for i in tqdm(range(domain.num_simulations)):
        save_dir = self.base_dir + '/sim_' + domain.name + '_' + str(i).zfill(4)
        sim = domain(config, save_dir)
        sims.append(sim)
    self.sims = sims

  def load_dp(self, num_dps, state_converter, seq_state_converter):
    dps_per_sim = int(num_dps/len(self.sims))
    for sim in self.sims:
      try:
        print("loaded")
        sim.load_dp()
      except:
        sim.add_rand_dps(num_dps=dps_per_sim,
                        seq_length=self.seq_length,
                        state_shape_converter=state_converter, 
                        seq_state_shape_converter=seq_state_converter,
                        train_cshape=self.train_cshape,
                        cratio=self.cratio)
 
  def load_cdp(self, num_cdps, state_converter, seq_cstate_converter, encode_state, encode_boundary):
    cdps_per_sim = int(num_cdps/len(self.sims))
    for sim in self.sims:
      try:
        sim.load_cdp()
      except:
        sim.add_rand_cdps(num_cdps=cdps_per_sim,
                        seq_length=self.seq_length,
                        state_shape_converter=state_converter, 
                        seq_cstate_shape_converter=seq_cstate_converter,
                        train_cshape=self.train_cshape,
                        cratio=self.cratio,
                        encode_state=encode_state,
                        encode_boundary=encode_boundary)

  def add_rand_dp(self, num_dps, state_converter, seq_state_converter):
    dps_per_sim = int(num_dps/len(self.sims))
    for sim in self.sims:
      sim.add_rand_dps(num_dps=dps_per_sim,
                       seq_length=self.seq_length,
                       state_shape_converter=state_converter, 
                       seq_state_shape_converter=seq_state_converter,
                       train_cshape=self.train_cshape,
                       cratio=self.cratio)

  def add_rand_cdp(self, num_cdps, state_converter, seq_cstate_converter, encode_state, encode_boundary):
    cdps_per_sim = int(num_cdps/len(self.sims))
    for sim in self.sims:
      sim.add_rand_cdps(num_cdps=cdps_per_sim,
                        seq_length=self.seq_length,
                        state_shape_converter=state_converter, 
                        seq_cstate_shape_converter=seq_cstate_converter,
                        train_cshape=self.train_cshape,
                        cratio=self.cratio,
                        encode_state=encode_state,
                        encode_boundary=encode_boundary)

  def start_dp_queue(self):
    # queue
    self.queue_dp = Queue()
    self.queue_dp_batches = []
    for sim in self.sims:
      thr = threading.Thread(target= (lambda: self.dp_worker(sim)))
      thr.daemon = True
      thr.start()

  def start_cdp_queue(self):
    # queue
    self.queue_cdp = Queue()
    self.queue_cdp_batches = []
    for sim in self.sims:
      thr = threading.Thread(target= (lambda: self.cdp_worker(sim)))
      thr.daemon = True
      thr.start()

  def rand_dp(self, state_shape_converter, seq_state_shape_converter):
    index = np.random.randint(0, len(self.sims))
    dp = self.sims[index].generate_rand_dp(
                 seq_length=self.seq_length,
                 state_shape_converter=state_shape_converter, 
                 seq_state_shape_converter=seq_state_shape_converter,
                 train_cshape=self.train_cshape,
                 cratio=self.cratio)
    state, boundary, seq_state = self.sims[index].read_dp(dp, add_batch=True)

    # make feed dict
    feed_dict = {}
    gpu_str = '_gpu_' + str(self.gpus[0])
    feed_dict['state' + gpu_str] = state
    feed_dict['boundary' + gpu_str] = boundary
    for j in range(self.seq_length):
      feed_dict['true_state' + gpu_str + '_' + str(j)] = seq_state[j]

    return index, dp, feed_dict

  def rand_cdp(self, state_shape_converter, seq_cstate_shape_converter, encode_state, encode_boundary):
    index = np.random.randint(0, len(self.sims))
    cdp = self.sims[index].generate_rand_cdp(
                 seq_length=self.seq_length,
                 state_shape_converter=state_shape_converter, 
                 seq_state_shape_converter=seq_cstate_shape_converter,
                 train_cshape=self.train_cshape,
                 cratio=self.cratio,
                 encode_state=encode_state,
                 encode_boundary=encode_boundary)
    cstate, cboundary, seq_cstate = self.sims[index].read_cdp(cdp, add_batch=True)

    # make feed dict
    feed_dict = {}
    gpu_str = '_gpu_' + str(self.gpus[0])
    feed_dict['cstate' + gpu_str] = cstate
    feed_dict['cboundary' + gpu_str] = cboundary 
    for j in range(self.seq_length):
      feed_dict['true_cstate' + gpu_str + '_' + str(j)] = seq_cstate[j]

    return index, cdp, feed_dict

  def add_dps(self, sim_indexes, dps):
    for ind, dp in zip(sim_indexes, dps):
      self.sims[ind].add_dp(dp)
    for sim in self.sims:
      sim.save_dps()
 
  def add_cdps(self, sim_indexes, cdps):
    for ind, cdp in zip(sim_indexes, cdps):
      self.sims[ind].add_cdp(cdp)
    for sim in self.sims:
      sim.save_cdps()

  def dp_minibatch(self):
    # queue up data if needed
    for i in range((self.max_queue/(len(self.gpus) * self.batch_size)) - (len(self.queue_dp_batches) + self.queue_dp.qsize())):
      self.queue_dp.put(None)
    while len(self.queue_dp_batches) <= 2: # added times two to make sure enough
      self.waiting_time += 1.0
      self.needed_to_wait = True
      time.sleep(1.0)
    feed_dict = self.queue_dp_batches.pop(0)
    return feed_dict

  def cdp_minibatch(self):
    # queue up data if needed
    for i in range((self.max_queue/(len(self.gpus) * self.batch_size)) - (len(self.queue_cdp_batches) + self.queue_cdp.qsize())):
      self.queue_cdp.put(None)
    while len(self.queue_cdp_batches) <= 2: # added times two to make sure enough
      self.waiting_time += 1.0
      self.needed_to_wait = True
      time.sleep(1.0)
    feed_dict = self.queue_cdp_batches.pop(0)
    return feed_dict

  def num_dps(self):
    num_points = 0
    for i in range(len(self.sims)):
      num_points += len(self.sims[i].data_points)
    return num_points

  def num_cdps(self):
    num_points = 0
    for i in range(len(self.sims)):
      num_points += len(self.sims[i].cdata_points)
    return num_points

  def dp_ind_histogram(self):
    inds = []
    for i in range(len(self.sims)):
      for j in range(len(self.sims[i].data_points)):
        inds.append(self.sims[i].data_points[j].ind)
    inds = np.array(inds)
    return vector_to_text_hist(inds, bins=10)

  def cdp_ind_histogram(self):
    inds = []
    for i in range(len(self.sims)):
      for j in range(len(self.sims[i].cdata_points)):
        inds.append(self.sims[i].cdata_points[j].ind)
    inds = np.array(inds)
    return vector_to_text_hist(inds, bins=10)
 
  def queue_dp_stats(self):
    stats = {}
    stats['percent full'] = int(100*float(len(self.gpus)*self.batch_size*len(self.queue_dp_batches))/float(self.max_queue))
    stats['samples in queue'] = int(len(self.gpus) * self.batch_size * self.queue_dp.qsize())
    stats['total_wait_time'] = int(self.waiting_time)
    stats['queue_waited'] = self.needed_to_wait
    stats['num_data_points'] = self.num_dps()
    stats['ind_histogram'] = self.dp_ind_histogram()
    if len(self.queue_dp_batches) > 1:
      gpu_str = '_gpu_' + str(self.gpus[0])
      stats['input_shape'] = self.queue_dp_batches[0]['state' + gpu_str][0].shape
      stats['output_shape'] = self.queue_dp_batches[0]['true_state' + gpu_str + '_0'][0].shape
    self.needed_to_wait = False
    return stats
 
  def queue_cdp_stats(self):
    stats = {}
    stats['percent full'] = int(100*float(len(self.gpus)*self.batch_size*len(self.queue_cdp_batches))/float(self.max_queue))
    stats['samples in queue'] = int(len(self.gpus) * self.batch_size * self.queue_cdp.qsize())
    stats['total_wait_time'] = int(self.waiting_time)
    stats['queue_waited'] = self.needed_to_wait
    stats['num_data_points'] = self.num_cdps()
    stats['ind_histogram'] = self.cdp_ind_histogram()
    if len(self.queue_cdp_batches) > 1:
      gpu_str = '_gpu_' + str(self.gpus[0])
      stats['input_shape'] = self.queue_cdp_batches[0]['cstate' + gpu_str][0].shape
      stats['output_shape'] = self.queue_cdp_batches[0]['true_cstate' + gpu_str + '_0'][0].shape
    self.needed_to_wait = False
    return stats


  def dp_worker(self, sim):
    while True:
      self.queue_dp.get()

      batch_state = []
      batch_boundary = []
      batch_seq_state = []
      for i in range(self.batch_size*len(self.gpus)):
        # get data
        state, boundary, seq_state = sim.read_rand_dp()

        # place into batches
        batch_state.append(state)
        batch_boundary.append(boundary)
        batch_seq_state.append(seq_state)

      # make feed dict
      feed_dict = {}
      for i in range(len(self.gpus)):
        gpu_str = '_gpu_' + str(self.gpus[i])
        feed_dict['state' + gpu_str]    = np_stack_list(batch_state[i*self.batch_size:(i+1)*self.batch_size])
        if batch_boundary[0] is not None:
          feed_dict['boundary' + gpu_str] = np_stack_list(batch_boundary[i*self.batch_size:(i+1)*self.batch_size])
        for j in range(self.seq_length):
          batch_seq_state_tmp = [x[j] for x in batch_seq_state]
          feed_dict['true_state' + gpu_str + '_' + str(j)] = np_stack_list(batch_seq_state_tmp[i*self.batch_size:(i+1)*self.batch_size])

      # add to que
      self.queue_dp_batches.append(feed_dict)
      self.queue_dp.task_done()

  def cdp_worker(self, sim):
    while True:
      self.queue_cdp.get()

      batch_cstate = []
      batch_cboundary = []
      batch_seq_cstate = []
      for i in range(self.batch_size*len(self.gpus)):
        # get data
        cstate, cboundary, seq_cstate = sim.read_rand_cdp()

        # place into batches
        batch_cstate.append(cstate)
        batch_cboundary.append(cboundary)
        batch_seq_cstate.append(seq_cstate)

      # make feed dict
      feed_dict = {}
      for i in range(len(self.gpus)):
        gpu_str = '_gpu_' + str(self.gpus[i])
        feed_dict['cstate' + gpu_str]    = np_stack_list(batch_cstate[i*self.batch_size:(i+1)*self.batch_size])
        if batch_cboundary[0] is not None:
          feed_dict['cboundary' + gpu_str] = np_stack_list(batch_cboundary[i*self.batch_size:(i+1)*self.batch_size])
        for j in range(self.seq_length):
          batch_seq_cstate_tmp = [x[j] for x in batch_seq_cstate]
          feed_dict['true_cstate' + gpu_str + '_' + str(j)] = np_stack_list(batch_seq_cstate_tmp[i*self.batch_size:(i+1)*self.batch_size])

      # add to que
      self.queue_cdp_batches.append(feed_dict)
      self.queue_cdp.task_done()


