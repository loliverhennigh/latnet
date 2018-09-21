
import numpy as np
import os
import sys
from copy import copy
from queue import Queue
from tqdm import *
from threading import Thread
import h5py
import time

from wrapper import SailfishWrapper, JHTDBWrapper, SpectralDNSWrapper
import lattice
from shape_converter import SubDomain
import utils.numpy_utils as numpy_utils
from utils.python_utils import *

class Domain(object):

  def __init__(self, config, save_dir):
    super(Domain, self).__init__(config, save_dir)
    self.save_dir = save_dir
    self.lb_to_ln = config.lb_to_ln
    self.debug_sailfish = config.debug_sailfish
 
    self.DxQy = lattice.TYPES[config.DxQy]()

    # get padding TODO code not clean here 
    if self.DxQy.dims == 2:
      self.padding_type = ['zero', 'zero']
      if self.periodic_x:
        self.padding_type[0] = 'periodic'
      if self.periodic_y:
        self.padding_type[1] = 'periodic'
    if self.DxQy.dims == 3:
      self.padding_type = ['zero', 'zero', 'zero']
      if self.periodic_x:
        self.padding_type[0] = 'periodic'
      if self.periodic_y:
        self.padding_type[1] = 'periodic'
      if self.periodic_z:
        self.padding_type[2] = 'periodic'

  @classmethod
  def add_options(cls, group):
    group.add_argument('--lb_to_ln', 
                   help='ratio of flow simulation steps to neural network steps', 
                   type=int,
                   default=32)
    group.add_argument('--DxQy', 
                   help='type of flow data', 
                   type=str,
                   default='D2Q9')
    group.add_argument('--visc', 
                   help='viscocity of fluid', 
                   type=float,
                   default=0.1)
    group.add_argument('--domain_name', 
                   help='domain name to run (non use config for sailfish)', 
                   type=str,
                   default='channel')

 
class TrainDomain(Domain):
  def __init__(self, config, save_dir):
    super(TrainDomain, self).__init__(config, save_dir)

    # data point for training (both uncompressed and compressed)
    self.data_points = []
    self.cdata_points = []

    # datapoint filenames
    self.dp_filename = self.save_dir + '/data_points_seq_length_' + str(config.seq_length) + '.txt'
    self.cdp_filename = self.save_dir + '/cdata_points_seq_length_' + str(config.seq_length) + '.txt'

  @classmethod
  def add_options(cls, group):
    pass

  def select_rand_dp(self):
    point_ind = np.random.randint(0, len(self.data_points))
    return self.data_points[point_ind]

  def select_rand_cdp(self, augment=False):
    point_ind = np.random.randint(0, len(self.cdata_points))
    return self.cdata_points[point_ind]

  def read_dp(self, dp, add_batch=False):
    # read state
    state = self.read_state(dp.ind, dp.state_subdomain, add_batch)

    # read boundary
    boundary = self.read_boundary(dp.state_subdomain, add_batch)

    # read seq states
    seq_state = []
    for i in range(dp.seq_length):
      seq_state.append(self.read_state(dp.ind + i, dp.seq_state_subdomain, add_batch))

    return state, boundary, seq_state

  def read_cdp(self, cdp, add_batch=False):
    # read cstate
    cstate = self.read_cstate(cdp.ind, cdp.state_subdomain, add_batch)

    # read cboundary
    cboundary = self.read_cboundary(cdp.state_subdomain, add_batch)

    # read seq cstates
    seq_cstate = []
    for i in range(cdp.seq_length):
      seq_cstate.append(self.read_cstate(cdp.ind + i, cdp.seq_state_subdomain, add_batch))

    return cstate, cboundary, seq_cstate

  def read_rand_dp(self, add_batch=False):
    dp = self.select_rand_dp()
    return self.read_dp(dp, add_batch)

  def read_rand_cdp(self, add_batch=False):
    cdp = self.select_rand_cdp()
    return self.read_cdp(cdp, add_batch)

  def add_dp(self, dp):
    self.data_points.append(dp)

  def add_cdp(self, cdp):
    self.cdata_points.append(cdp)

  def save_dps(self):
    with open(self.dp_filename, "w") as f:
      for point in self.data_points:
        save_string = point.to_string()
        f.write(save_string)

  def save_cdps(self):
    with open(self.cdp_filename, "w") as f:
      for point in self.cdata_points:
        save_string = point.to_string()
        f.write(save_string)

  def load_dp(self):
    with open(self.dp_filename, "r") as f:
      string_points = f.readlines()
      for p in string_points: 
        d = DataPoint(0,0,0,0)
        d.load_string(p)
        self.data_points.append(d)

  def load_cdp(self):
    with open(self.cdp_filename, "r") as f:
      string_points = f.readlines()
      for p in string_points: 
        d = DataPoint(0,0,0,0)
        d.load_string(p)
        self.cdata_points.append(d)

class SailfishDomain(Domain, SailfishWrapper):
  wrapper_name = 'sailfish'
  def __init__(self, config, save_dir):
    super(SailfishDomain, self).__init__(config, save_dir)

  @classmethod
  def add_options(cls, group, dim):
    pass

  @classmethod
  def update_defaults(cls, defaults):
    pass

  def read_state(self, iteration, subdomain=None, add_batch=False, return_padding=True):
    # load flow file
    state_file = self.iter_to_state_filename(iteration)
    state = np.load(state_file)
    state = state.f.dist0a[:,1:-1,1:self.sim_shape[1]+1]
    state = state.astype(np.float32)
    state = np.swapaxes(state, 0, 1)
    state = np.swapaxes(state, 1, 2)
    state = self.DxQy.subtract_lattice(state)
    
    if subdomain is not None:
      state, pad_state = numpy_utils.mobius_extract(state, subdomain,
                                            padding_type=self.padding_type,
                                            return_padding=True)
    elif return_padding:
      pad_state = np.zeros_like(state[...,0:1])

    if add_batch:
      state = np.expand_dims(state, axis=0)
      if return_padding:
        pad_state = np.expand_dims(pad_state, axis=0)

    if return_padding:
      return (state, pad_state)
    else:
      return state

  def read_boundary(self, subdomain=None, add_batch=False, return_padding=True):
    boundary_file = self.boundary_filename()
    boundary = None
    boundary = np.load(boundary_file)
    boundary = boundary.astype(np.float32)
    boundary = boundary[1:-1,1:-1]
    if subdomain is not None:
      boundary, pad_boundary = numpy_utils.mobius_extract(boundary, subdomain,
                                            padding_type=self.padding_type,
                                            return_padding=True)
    elif return_padding:
      pad_boundary = np.zeros_like(boundary[...,0:1])

    if add_batch:
      boundary     = np.expand_dims(boundary, axis=0)
      if return_padding:
        pad_boundary = np.expand_dims(pad_boundary, axis=0)

    if return_padding:
      return (boundary, pad_boundary)
    else:
      return boundary

  def read_cstate(self, iteration, subdomain=None, add_batch=False, return_padding=True):
    # load flow file
    cstate_file = self.iter_to_cstate_filename(iteration)
    cstate = np.load(cstate_file)

    if subdomain is not None:
      cstate, pad_cstate = numpy_utils.mobius_extract(cstate, subdomain,
                                                padding_type=self.padding_type,
                                                return_padding=True)
    elif return_padding:
      pad_cstate = np.zeros_like(cstate[...,0:1])

    if add_batch:
      cstate = np.expand_dims(cstate, axis=0)
      if return_padding:
        pad_cstate = np.expand_dims(pad_cstate, axis=0)

    if return_padding:
      return (cstate, pad_cstate)
    else:
      return cstate

  def read_cboundary(self, subdomain=None, add_batch=False, return_padding=True):
    cboundary_file = self.cboundary_filename()
    cboundary = np.load(cboundary_file)

    if subdomain is not None:
      cboundary, pad_cboundary = numpy_utils.mobius_extract(cboundary, subdomain,
                                            padding_type=self.padding_type,
                                            return_padding=True)
    elif return_padding:
      pad_cboundary = np.zeros_like(cboundary[...,0:1])

    if add_batch:
      cboundary = np.expand_dims(cboundary, axis=0)
      if return_padding:
        pad_cboundary = np.expand_dims(pad_cboundary, axis=0)

    if return_padding:
      return (cboundary, pad_cboundary)
    else:
      return cboundary

  def read_vel_rho(self, iteration, subdomain=None, add_batch=False):
    state = self.read_state(iteration, subdomain, add_batch=add_batch)
    vel = self.DxQy.lattice_to_vel(state)
    rho = self.DxQy.lattice_to_rho(state)
    return vel, rho

  def generate_data(self, num_iters):
    self.new_sim(num_iters)

class SpectralDNSDomain(Domain, SpectralDNSWrapper):
  wrapper_name = 'spectralDNS'
  def __init__(self, config, save_dir):
    super(SpectralDNSDomain, self).__init__(config, save_dir)

  @classmethod
  def add_options(cls, group, dim):
    pass

  @classmethod
  def update_defaults(cls, defaults):
    pass

  def load_stream(self):
    if not hasattr(self, 'state_stream'):
      print("loading dataset...")
      self.state_stream = h5py.File(self.h5filename, driver='core')
      print("finished...")
      self.states = {}

  def read_state(self, iteration, subdomain=None, add_batch=False, return_padding=True):
    # load flow file
    self.load_stream()

    t = time.time()
    spectral_iteration = self.latnet_iter_to_spectral_iter(iteration)
    if str(spectral_iteration) not in list(self.states.keys()):
      #print("Storing data in memory to save time \n iteration: " + str(spectral_iteration))
      key_vel_u    = self.state_stream['3D']['U'][str(spectral_iteration)].value.astype(np.float32)
      key_vel_v    = self.state_stream['3D']['V'][str(spectral_iteration)].value.astype(np.float32)
      key_vel_w    = self.state_stream['3D']['W'][str(spectral_iteration)].value.astype(np.float32)
      key_pressure = self.state_stream['3D']['P'][str(spectral_iteration)].value.astype(np.float32)
      state = np.stack([key_vel_u, key_vel_v, key_vel_w, key_pressure], axis=-1)
      self.states[str(spectral_iteration)] = state
    else:
      state = self.states[str(spectral_iteration)]
    
    if subdomain is not None:
      state, pad_state = numpy_utils.mobius_extract(state, subdomain,
                                            padding_type=self.padding_type,
                                            return_padding=True)
    elif return_padding:
      pad_state = np.zeros_like(state[...,0:1])

    if add_batch:
      state = np.expand_dims(state, axis=0)
      if return_padding:
        pad_state = np.expand_dims(pad_state, axis=0)

    #print("needed time was: " + str(time.time() - t))

    if return_padding:
      return (state, pad_state)
    else:
      return state

  def read_boundary(self, subdomain=None, add_batch=False, return_padding=True):
    return None

  def read_cstate(self, iteration, subdomain=None, add_batch=False, return_padding=True):
    # load flow file
    cstate_file = self.iter_to_cstate_filename(iteration)
    cstate = np.load(cstate_file)

    if subdomain is not None:
      cstate, pad_cstate = numpy_utils.mobius_extract(cstate, subdomain,
                                                padding_type=self.padding_type,
                                                return_padding=True)
    elif return_padding:
      pad_cstate = np.zeros_like(cstate[...,0:1])

    if add_batch:
      cstate = np.expand_dims(cstate, axis=0)
      if return_padding:
        pad_cstate = np.expand_dims(pad_cstate, axis=0)

    if return_padding:
      return (cstate, pad_cstate)
    else:
      return cstate

  def read_cboundary(self, subdomain=None, add_batch=False, return_padding=True):
    return None

  def read_vel_rho(self, iteration, subdomain=None, add_batch=False):
    state = self.read_state(iteration, subdomain, add_batch=add_batch)
    vel = self.DxQy.lattice_to_vel(state)
    rho = self.DxQy.lattice_to_rho(state)
    return vel, rho

  def need_to_generate(self):
    # check if need to generate train data or not
    need = True
    if os.path.isfile(self.h5filename):
      need = False
    return need 

  def generate_data(self):
    self.new_sim()

class JHTDBDomain(Domain, JHTDBWrapper):
  wrapper_name = 'JHTDB'
  def __init__(self, config, save_dir):
    super(JHTDBDomain, self).__init__(config, save_dir)

  def add_options(cls, group, dim):
    pass

  def read_state(self, iteration, subdomain=None, add_batch=False, return_padding=True):
    state_file = self.save_dir + self.iter_to_state_filename(iteration, subdomain)
    state_stream = h5py.File(state_file)
    key_vel      = state_stream.keys()[5]
    key_pressure = state_stream.keys()[4]
    vel = np.array(state_stream[key_vel])
    pressure = np.array(state_stream[key_pressure])
    state_stream = None
    state = np.concatenate([vel/10.0, pressure/3.0], axis=-1)
    state = state.astype(np.float32)
    if add_batch:
      state = np.expand_dims(state, axis=0)
    if return_padding:
      state_pad = np.zeros(list(state.shape[:-1]) + [1])
      return state, state_pad
    else:
      return state

  def read_boundary(self, subdomain=None, add_batch=False, return_padding=True):
    return None

  def read_cstate(self, iteration, subdomain=None, add_batch=False, return_padding=True):
    # load flow file
    state_file = self.save_dir + self.iter_to_cstate_filename(iteration, subdomain)

    cstate = np.load(state_file)

    if add_batch:
      cstate = np.expand_dims(cstate, axis=0)
    if return_padding:
      cstate_pad = np.zeros(list(cstate.shape[:-1]) + [1])
      return cstate, cstate_pad
    else:
      return cstate

  def read_cboundary(self, subdomain=None, add_batch=False, return_padding=True):
    return None

  def read_vel_rho(self, iteration, subdomain=None, add_batch=False):
    state = self.read_state(iteration, subdomain, add_batch, return_padding=False)
    return state[...,0:3], state[...,3:4]

class TrainSailfishDomain(SailfishDomain, TrainDomain):
  def __init__(self, config, save_dir):
    super(TrainSailfishDomain, self).__init__(config, save_dir)
    self.num_states = self.max_sim_iters

    # check if need to generate data and do so
    if self.need_to_generate() and (not (config.run_mode == 'generate_data')):
      self.generate_train_data()

  def need_to_generate(self):
    # check if need to generate train data or not
    need = False
    state_filenames = self.list_state_filenames()
    if len(state_filenames) != self.num_states:
      need = True 
    boundary_filename = self.boundary_filename()
    if not os.path.isfile(boundary_filename):
      need = True 
    return need 

  def generate_train_data(self):
    self.new_sim(self.num_states)

  def generate_rand_dp(self, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio):
    # select random index
    state_files = self.list_state_filenames()
    ind = np.random.randint(1, len(state_files) - seq_length)

    # select random pos to grab from data
    rand_pos = [np.random.randint(-train_cshape[0], self.sim_shape[0]/cratio+1),
                np.random.randint(-train_cshape[1], self.sim_shape[1]/cratio+1)]
    cstate_subdomain = SubDomain(rand_pos, train_cshape)

    # get state subdomain and geometry_subdomain
    state_subdomain = state_shape_converter.out_in_subdomain(copy(cstate_subdomain))

    # get seq state subdomain
    seq_state_subdomain = seq_state_shape_converter.in_out_subdomain(copy(state_subdomain))

    # data point and return it
    return DataPoint(ind, seq_length, state_subdomain, seq_state_subdomain)

  def generate_rand_cdp(self, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio, encode_state, encode_boundary):
    # select random index
    state_files = self.list_state_filenames()
    ind = np.random.randint(1, len(state_files) - seq_length)

    # select random pos to grab from data
    rand_pos = [np.random.randint(-train_cshape[0], int(self.sim_shape[0]/cratio+1)),
                np.random.randint(-train_cshape[1], int(self.sim_shape[1]/cratio+1))]
    cstate_subdomain = SubDomain(rand_pos, train_cshape)

    # get state subdomain and geometry_subdomain
    in_cstate_subdomain = seq_state_shape_converter.out_in_subdomain(copy(cstate_subdomain))

    # generate cstate files if needed
    for i in range(seq_length):
      if not os.path.isfile(self.iter_to_cstate_filename(ind + i)):
        encode_cstate_subdomain = SubDomain(self.DxQy.dims*[0], [int(x/cratio) for x in self.sim_shape])
        encode_state_subdomain = state_shape_converter.out_in_subdomain(copy(encode_cstate_subdomain))
        state = self.read_state(ind+i, subdomain=encode_state_subdomain, add_batch=True, return_padding=True)
        cstate = encode_state({'state':state})[0]
        np.save(self.iter_to_cstate_filename(ind + i), cstate)
      
    # generate cboundary file if needed
    if not os.path.isfile(self.cboundary_filename()):
      boundary = self.read_boundary(subdomain=encode_state_subdomain, add_batch=True, return_padding=True)
      cboundary = encode_boundary({'boundary':boundary})[0]
      np.save(self.cboundary_filename(), cboundary)

    # data point and return it
    return DataPoint(ind, seq_length, in_cstate_subdomain, cstate_subdomain)

  def add_rand_dps(self, num_dps, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio):
    for i in range(num_dps):
      # make datapoint and add to list
      self.data_points.append(self.generate_rand_dp(seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio))
    self.save_dps()

  def add_rand_cdps(self, num_cdps, seq_length, state_shape_converter, seq_cstate_shape_converter, train_cshape, cratio, encode_state, encode_boundary):
    for i in range(num_cdps):
      # make datapoint and add to list
      self.cdata_points.append(self.generate_rand_cdp(seq_length, state_shape_converter, seq_cstate_shape_converter, train_cshape, cratio, encode_state, encode_boundary))
    self.save_cdps()

class TrainSpectralDNSDomain(SpectralDNSDomain, TrainDomain):
  def __init__(self, config, save_dir):
    super(SpectralDNSDomain, self).__init__(config, save_dir)

    # check if need to generate data and do so
    if self.need_to_generate() and (not (config.run_mode == 'generate_data')):
      self.generate_train_data()
    self.load_stream()

  def need_to_generate(self):
    # check if need to generate train data or not
    need = True
    if os.path.isfile(self.h5filename):
      need = False
    return need 

  def generate_train_data(self):
    self.new_sim()

  def generate_rand_dp(self, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio):
    # select random index
    state_iters = self.list_state_iters()
    ind = np.random.randint(self.cutoff_latnet_iteration, len(state_iters) - seq_length)

    # select random pos to grab from data
    rand_pos = [np.random.randint(-train_cshape[0], self.sim_shape[0]/cratio+1),
                np.random.randint(-train_cshape[1], self.sim_shape[1]/cratio+1),
                np.random.randint(-train_cshape[2], self.sim_shape[2]/cratio+1)]
    cstate_subdomain = SubDomain(rand_pos, train_cshape)

    # get state subdomain and geometry_subdomain
    state_subdomain = state_shape_converter.out_in_subdomain(copy(cstate_subdomain))

    # get seq state subdomain
    seq_state_subdomain = seq_state_shape_converter.in_out_subdomain(copy(state_subdomain))

    # data point and return it
    return DataPoint(ind, seq_length, state_subdomain, seq_state_subdomain)

  def generate_rand_cdp(self, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio, encode_state, encode_boundary):
    # select random index
    state_iters = self.list_state_iters()
    ind = np.random.randint(self.cutoff_latnet_iteration, len(state_iters) - seq_length)

    # select random pos to grab from data
    rand_pos = [np.random.randint(-train_cshape[0], self.sim_shape[0]/cratio+1),
                np.random.randint(-train_cshape[1], self.sim_shape[1]/cratio+1),
                np.random.randint(-train_cshape[2], self.sim_shape[2]/cratio+1)]
    cstate_subdomain = SubDomain(rand_pos, train_cshape)

    # get state subdomain and geometry_subdomain
    in_cstate_subdomain = seq_state_shape_converter.out_in_subdomain(copy(cstate_subdomain))

    # generate cstate files if needed
    for i in range(seq_length):
      if not os.path.isfile(self.iter_to_cstate_filename(ind + i)):
        encode_cstate_subdomain = SubDomain(self.DxQy.dims*[0], [int(x/cratio) for x in self.sim_shape])
        encode_state_subdomain = state_shape_converter.out_in_subdomain(copy(encode_cstate_subdomain))
        state = self.read_state(ind+i, subdomain=encode_state_subdomain, add_batch=True, return_padding=True)
        cstate = encode_state({'state':state})[0]
        np.save(self.iter_to_cstate_filename(ind + i), cstate)
      
    # data point and return it
    return DataPoint(ind, seq_length, in_cstate_subdomain, cstate_subdomain)

  def add_rand_dps(self, num_dps, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio):
    for i in range(num_dps):
      # make datapoint and add to list
      self.data_points.append(self.generate_rand_dp(seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio))
    self.save_dps()

  def add_rand_cdps(self, num_cdps, seq_length, state_shape_converter, seq_cstate_shape_converter, train_cshape, cratio, encode_state, encode_boundary):
    for i in range(num_cdps):
      # make datapoint and add to list
      self.cdata_points.append(self.generate_rand_cdp(seq_length, state_shape_converter, seq_cstate_shape_converter, train_cshape, cratio, encode_state, encode_boundary))
    self.save_cdps()

class TrainJHTDBDomain(JHTDBDomain, TrainDomain):
  def __init__(self, config, save_dir):
    super(TrainJHTDBDomain, self).__init__(config, save_dir)

    self.dp_queue = Queue(20)
    self.cdp_queue = Queue(20)
    for i in range(10):
      t = Thread(target=self.download_dp_worker)
      t.daemon = True
      t.start()
      t = Thread(target=self.download_cdp_worker)
      t.daemon = True
      t.start()

  def download_dp_worker(self):
    while True:
      (seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio) = self.dp_queue.get()
      # make datapoint and add to list
      self.data_points.append(self.generate_rand_dp(seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio))
      self.dp_queue.task_done()
 
  def download_cdp_worker(self):
    while True:
      (seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio, encode_state, encode_boundary) = self.cdp_queue.get()
      # make datapoint and add to list
      self.data_points.append(self.generate_rand_cdp(seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio, encode_state, encode_boundary))
      self.cdp_queue.task_done()

  def add_rand_dps(self, num_dps, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio):
    self.save_dps()
    self.dp_queue.join()
    for i in tqdm(range(num_dps)):
      self.dp_queue.put((seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio))
    if len(self.data_points) == 0:
      self.dp_queue.join()
    self.save_dps()

  def add_rand_cdps(self, num_cdps, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio, encode_state, encode_boundary):
    self.save_cdps()
    self.cdp_queue.join()
    for i in tqdm(range(num_cdps)):
      self.cdp_queue.put((seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio, encode_state, encode_boundary))
    if len(self.cdata_points) == 0:
      self.cdp_queue.join()
    self.save_cdps()

  def generate_rand_dp(self, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio):
    # select random index
    ind = np.random.randint(0, (5024/self.lb_to_ln)-seq_length)

    # select random pos to grab from data
    rand_pos = [np.random.randint(train_cshape[0]+16, self.sim_shape[0]/cratio - train_cshape[0] - 16),
                np.random.randint(train_cshape[1]+16, self.sim_shape[1]/cratio - train_cshape[1] - 16),
                np.random.randint(train_cshape[2]+16, self.sim_shape[2]/cratio - train_cshape[2] - 16)]
    cstate_subdomain = SubDomain(rand_pos, train_cshape)

    # get state subdomain and geometry_subdomain
    state_subdomain = state_shape_converter.out_in_subdomain(copy(cstate_subdomain))

    # get seq state subdomain
    seq_state_subdomain = seq_state_shape_converter.in_out_subdomain(copy(state_subdomain))

    # download data
    self.download_state(ind, state_subdomain)
    for i in range(seq_length):
      self.download_state(ind+i, seq_state_subdomain)

    # data point and return it
    return DataPoint(ind, seq_length, state_subdomain, seq_state_subdomain)

  def generate_rand_cdp(self, seq_length, state_shape_converter, seq_state_shape_converter, train_cshape, cratio, encode_state, encode_boundary):
    # select random index
    ind = np.random.randint(0, (5024/self.lb_to_ln)-seq_length)

    # select random pos to grab from data
    rand_pos = [np.random.randint(-train_cshape[0], self.sim_shape[0]/cratio+1),
                np.random.randint(-train_cshape[1], self.sim_shape[1]/cratio+1)]
    cstate_subdomain = SubDomain(rand_pos, train_cshape)

    # get state subdomain and geometry_subdomain
    in_cstate_subdomain = seq_state_shape_converter.out_in_subdomain(copy(cstate_subdomain))

    # download data and compress
    self.download_state(state_subdomain, ind)
    state = self.read_state(ind, state_subdomain, add_batch=True, return_padding=True)
    cstate = encode_state({'state':state})[0]
    np.save(self.iter_to_cstate_filename(ind + i, state_subdomain), cstate)
    os.remove(self.iter_to_state_filename(ind + i, state_subdomain))
    for i in range(seq_length):
      self.download_state(seq_state_subdomain, ind + i)
      state = self.read_state(ind + i, seq_state_subdomain, add_batch=True, return_padding=True)
      cstate = encode_state({'state':state})[0]
      np.save(self.iter_to_cstate_filename(ind + i, seq_state_subdomain), cstate)
      os.remove(self.iter_to_state_filename(ind + i, seq_state_subdomain))

    # data point and return it
    return DataPoint(ind, seq_length, in_cstate_subdomain, cstate_subdomain)

class DataPoint:

  def __init__(self, ind, seq_length, state_subdomain, seq_state_subdomain):
    self.ind = ind
    self.seq_length = seq_length
    self.state_subdomain = state_subdomain
    self.seq_state_subdomain = seq_state_subdomain

  def to_string(self):
    string  = ''
    string += str(self.ind) + ','
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
    self.seq_length = values.pop(0)
    dims = len(values)/4
    if dims == 2:
      self.state_subdomain = SubDomain([values[0], values[1]], 
                                       [values[2], values[3]])
      self.seq_state_subdomain = SubDomain([values[4], values[5]], 
                                           [values[6], values[7]])
    elif dims == 3:
      self.state_subdomain = SubDomain([values[0], values[1], values[2]], 
                                       [values[3], values[4], values[5]])
      self.seq_state_subdomain = SubDomain([values[6], values[7], values[8]], 
                                           [values[9], values[10], values[11]])



