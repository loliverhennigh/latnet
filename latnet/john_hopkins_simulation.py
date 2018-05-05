
import os
import psutil as ps
import glob
import sys
from copy import copy
from tqdm import *
import requests
import time

from Queue import Queue
from threading import Thread
import lattice
import utils.numpy_utils as numpy_utils
from utils.python_utils import *

from shape_converter import SubDomain

# import sailfish
sys.path.append('../sailfish')
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.lb_base import LBForcedSim

import numpy as np
import h5py

class JohnHopkinsSimulation:

  def __init__(self, config, save_dir=None):

    self.save_dir = save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    self.step_ratio = 2
    self.config=config
 
    self.sim_shape = [1024, 1024, 1024]

    self.padding_type = []
    for i in xrange(3):
      self.padding_type.append('periodic')

  def make_url(self, subdomain, iteration):
    url_begining = "http://dsp033.pha.jhu.edu/jhtdb/getcutout/com.gmail.loliverhennigh101-4da6ce46/isotropic1024coarse/p,u/" 
    url_end =  str(iteration * self.step_ratio) + ",1/"
    url_end += str(subdomain.pos[0]) + ","
    url_end += str(subdomain.size[0]) + "/"
    url_end += str(subdomain.pos[1]) + ","
    url_end += str(subdomain.size[1]) + "/"
    url_end += str(subdomain.pos[2]) + ","
    url_end += str(subdomain.size[2]) + "/hdf5/"
    #print(url_begining + url_end)
    return url_begining + url_end

  def download_datapoint(self, subdomain, iteration):
    filename = self.make_filename(subdomain, iteration)
    path_filename = self.save_dir + filename
    if not os.path.isfile(path_filename):
      r = ''
      while (r == '') or (r == '<h1>Server Error (500)</h1>'):
        try:
          r = requests.get(self.make_url(subdomain, iteration)) 
          #print(r)
        except:
          print("having trouble getting data, will sleep and try again")
          time.sleep(2)
      with open(path_filename, 'wb') as f:
        f.write(r.content)

  def make_filename(self, subdomain, iteration):
    filename =  "/iteration_" + str(iteration) 
    filename += "pos_" + str(subdomain.pos[0]) + '_' + str(subdomain.pos[1]) +  '_' + str(subdomain.pos[2]) + '_'
    filename += "size_" + str(subdomain.size[0]) + '_' + str(subdomain.size[1]) + '_' + str(subdomain.size[2])
    filename += ".h5"
    return filename

  def read_boundary(self, subdomain=None, add_batch=False, return_padding=True):
    if subdomain is None:
      boundary = np.zeros(self.sim_shape + [1])
    else:
      boundary = np.zeros(subdomain.size + [1])
    if add_batch:
      boundary = np.expand_dims(boundary, axis=0)
    boundary_pad = np.zeros(list(boundary.shape[:-1]) + [1])
    if return_padding:
      return boundary, boundary_pad
    else:
      return boundary

  def read_state(self, iteration, subdomain=None, add_batch=False, return_padding=True):
    # load flow file
    state_file = self.save_dir + self.make_filename(subdomain, iteration)
    state_stream = h5py.File(state_file)
    key_vel      = state_stream.keys()[5]
    key_pressure = state_stream.keys()[4]
    vel = np.array(state_stream[key_vel])
    pressure = np.array(state_stream[key_pressure])
    state = np.concatenate([vel, pressure], axis=-1)
    state = state.astype(np.float32)/10.0
    if return_padding:
      state_pad = np.zeros(list(state.shape[:-1]) + [1])
      return state, state_pad
    else:
      return state

class TrainJohnHopkinsSimulation(JohnHopkinsSimulation):

  def __init__(self, config, save_dir):
    JohnHopkinsSimulation.__init__(self, config, save_dir)
    # more configs will probably be added later
    self.data_points = []

  def read_data(self, augment=False):

    # select datapoint
    point_ind = np.random.randint(0, len(self.data_points))
    data_point = self.data_points[point_ind]

    # read state
    state = self.read_state(data_point.ind, data_point.state_subdomain)

    # read boundary
    boundary = self.read_boundary(data_point.state_subdomain)

    # read seq states
    seq_state = []
    for i in xrange(data_point.seq_length):
      seq_state.append(self.read_state(data_point.ind + i, data_point.seq_state_subdomain))

    return state, boundary, seq_state

  def data_point_to_data(self, data_point, add_batch=False):
    # read state
    state = self.read_state(data_point.ind, data_point.state_subdomain, add_batch)

    # read boundary
    boundary = self.read_boundary(data_point.state_subdomain, add_batch)

    # read seq states
    seq_state = []
    for i in xrange(data_point.seq_length):
      seq_state.append(self.read_state(data_point.ind + i, data_point.seq_state_subdomain, add_batch))

    return state, boundary, seq_state

  def add_data_point(self, data_point):
    # either add to train or test set
    
    self.data_points.append(data_point)

  def make_rand_data_points(self, num_points, seq_length, state_shape_converter, seq_state_shape_converter, input_cshape, cratio):
    generate_data = True
    if os.path.isfile(self.save_dir + '/data_points.txt'):
      generate_data = False
      self.read_data_points() 
      if len(self.data_points) != num_points:
        self.data_points = [] 
        generate_data = True
    if generate_data:

      # make queue 
      my_queue = Queue(50)
 
      def worker():
        while True:
          i = my_queue.get()
          # make datapoint and add to list
          self.data_points.append(self.rand_data_point(seq_length, state_shape_converter, seq_state_shape_converter, input_cshape, cratio))
          my_queue.task_done()

      for i in tqdm(xrange(20)):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

      for i in tqdm(xrange(num_points)):
        my_queue.put(i)
      my_queue.join()

    self.save_data_points() 

  def save_data_points(self):
    datapoint_filename = self.save_dir + '/data_points.txt'
    with open(datapoint_filename, "w") as f:
      for point in self.data_points:
        save_string = point.to_string()
        f.write(save_string)

  def read_data_points(self):
    datapoint_filename = self.save_dir + '/data_points.txt'
    with open(datapoint_filename, "r") as f:
      string_points = f.readlines()
      for p in string_points: 
        d = DataPoint(0,0,0,0,0)
        d.load_string(p)
        self.data_points.append(d)

  def rand_data_point(self, seq_length, state_shape_converter, seq_state_shape_converter, input_cshape, cratio):
    # select random index
    ind = np.random.randint(0, (5024/self.step_ratio)-seq_length)

    # select random pos to grab from data
    rand_pos = [np.random.randint(input_cshape[0], self.sim_shape[0]/cratio - input_cshape[0]),
                np.random.randint(input_cshape[1], self.sim_shape[1]/cratio - input_cshape[1]),
                np.random.randint(input_cshape[2], self.sim_shape[2]/cratio - input_cshape[2])]
    cstate_subdomain = SubDomain(rand_pos, input_cshape)

    # get state subdomain and geometry_subdomain
    state_subdomain = state_shape_converter.out_in_subdomain(copy(cstate_subdomain))

    # get seq state subdomain
    seq_state_subdomain = seq_state_shape_converter.in_out_subdomain(copy(state_subdomain))

    # download data
    self.download_datapoint(state_subdomain, ind)
    for i in xrange(seq_length):
      self.download_datapoint(seq_state_subdomain, ind+i)

    # data point and return it
    return DataPoint(ind, seq_length, state_subdomain, seq_state_subdomain)

  def need_to_generate(self):
    return False

class DataPoint:

  def __init__(self, ind, train, seq_length, state_subdomain, seq_state_subdomain):
    self.ind = ind
    self.train = train
    self.seq_length = seq_length
    self.state_subdomain = state_subdomain
    self.seq_state_subdomain = seq_state_subdomain

  def to_string(self):
    string  = ''
    string += str(self.ind) + ','
    string += str(self.train) + ','
    string += str(self.seq_length) + ','
    for p in self.state_subdomain.pos:
      string += str(p) + ','
    for s in self.state_subdomain.size:
      string += str(s) + ','
    for p in self.seq_state_subdomain.pos:
      string += str(p) + ','
    for s in self.seq_state_subdomain.size:
      string += str(s) + ','
    string += '\n'
    return string

  def load_string(self, string):
    values = string.split(',')[:-1]
    values = [int(x) for x in values]
    self.ind = values.pop(0)
    self.train = values.pop(0)
    self.seq_length = values.pop(0)
    dims = len(values)/4
    self.state_subdomain = SubDomain([values[0], values[1], values[2]], 
                                     [values[3], values[4], values[5]])
    self.seq_state_subdomain = SubDomain([values[6], values[7], values[8]], 
                                         [values[9], values[10], values[11]])

def flip_boundary_vel(boundary):
  boundary[...,1] = -boundary[...,1]
  return boundary

def rotate_boundary_vel(boundary, k):
  for i in xrange(k):
    store_boundary = boundary[...,0]
    boundary[...,0] = -boundary[...,1]
    boundary[...,1] = boundary[...,0]
  return boundary

