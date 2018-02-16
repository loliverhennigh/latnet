
import os
import psutil as ps
import glob

import lattice
import utils.numpy_utils as numpy_utils

import numpy as np

class SailfishRunner:

  def __init__(self, config, save_dir, script_name):
    self.save_dir = save_dir
    self.lb_to_ln = config.lb_to_ln
    self.max_sim_iters = config.max_sim_iters
    self.script_name = script_name
    self.debug_sailfish = config.debug_sailfish
    self.boundary_mask = config.boundary_mask
 
    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape

    self.DxQy = lattice.TYPES[config.DxQy]()

    # set padding config
    self.padding_type = []
    if config.periodic_x:
      self.padding_type.append('periodic')
    else:
      self.padding_type.append('zero')
    if config.periodic_y:
      self.padding_type.append('periodic')
    else:
      self.padding_type.append('zero')

  def list_cpoints(self):
    cpoints = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    cpoints.sort()
    return cpoints

  def boundary_file(self):
    return self.save_dir + "/flow_geometry.npy"

  def first_cpoint(self):
    cpoints = self.list_cpoints()
    return cpoints[0], self.cpoint_to_iter(cpoints[0])

  def last_cpoint(self):
    cpoints = self.list_cpoints()
    return cpoints[-1], self.cpoint_to_iter(cpoints[-1])
 
  def is_restorable(self):
    cpoints = self.list_cpoints()
    boundary_file = self.boundary_file()
    return ((len(cpoints) > 0) and os.path.isfile(boundary_file))

  def cpoint_to_iter(self, cpoint_name):
    sailfish_iter = int(cpoint_name.split('.')[-4])
    return self.sailfish_iter_to_latnet_iter(sailfish_iter)

  def iter_to_cpoint(self, iteration):
    sailfish_iter = self.latnet_iter_to_sailfish_iter(iteration)
    zpadding = len(self.last_cpoint()[0].split('.')[-4])
    cpoint = (self.save_dir + '/flow.' 
             + str(sailfish_iter).zfill(zpadding)
             + '.0.cpoint.npz')
    return cpoint

  def sailfish_iter_to_latnet_iter(self, iteration):
    return int(iteration/self.lb_to_ln)

  def latnet_iter_to_sailfish_iter(self, iteration):
    return iteration*self.lb_to_ln

  def make_sim_dir(self):
    with open(os.devnull, 'w') as devnull:
      p = ps.subprocess.Popen(('mkdir -p ' + self.save_dir + "/store").split(' '), 
                               stdout=devnull, stderr=devnull)
      p.communicate()

  def clean_dir(self):
    store_files = glob.glob(self.save_dir + "/*")
    self.rm_files(store_files)

  def clean_store_dir(self):
    store_files = glob.glob(self.save_dir + "/store/*")
    self.rm_files(store_files)
 
  def mv_store_dir(self):
    store_files = glob.glob(self.save_dir + "/store/*")
    for f in store_files:
      p = ps.subprocess.Popen(["mv", f, self.save_dir + "/"])
      p.communicate()

  def rm_files(self, file_list):
    for f in file_list:
      with open(os.devnull, 'w') as devnull:
        p = ps.subprocess.Popen(["rm", f], 
                                 stdout=devnull, stderr=devnull)
        p.communicate()
 
  def new_sim(self, num_iters):

    self.make_sim_dir()
    self.clean_dir()
    self.clean_store_dir()

    cmd = ('./' + self.script_name 
         + ' --run_mode=generate_data'
         + ' --max_sim_iters=' + str(self.lb_to_ln*num_iters + 1)
         + ' --checkpoint_from=0')
    if self.debug_sailfish:
      cmd += ' --mode=visualization'
      cmd += ' --scr_scale=.5'
    else:
      cmd += ' --train_sim_dir=' + self.save_dir + '/store/flow'
    p = ps.subprocess.Popen(cmd.split(' '), 
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
    p.communicate()
   
    self.mv_store_dir()
 
  def restart_sim(self, num_iters, keep_old=False):

    assert self.is_restorable(), "trying to restart sim without finding proper save"
    self.clean_store_dir()

    last_cpoint, last_iter = self.last_cpoint()

    cmd = ('./' + self.script_name 
         + ' --run_mode=generate_data'
         + ' --max_sim_iters=' + str(self.latnet_iter_to_sailfish_iter(num_iters
                                                                    + last_iter) + 1)
         + ' --checkpoint_from=0'
         + ' --restore_geometry=True'
         + ' --restore_from=' + last_cpoint[:-13])
    if self.debug_sailfish:
      cmd += ' --mode=visualization'
      cmd += ' --scr_scale=.5'
    else:
      cmd += ' --train_sim_dir=' + self.save_dir + '/store/flow'
    p = ps.subprocess.Popen(cmd.split(' '), 
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
    p.communicate()
  
    if not keep_old:
      self.clean_dir()
    self.mv_store_dir()

  def read_boundary(self, subdomain=None):
    boundary_file = self.boundary_file()
    boundary = None
    if os.path.isfile(boundary_file):
      boundary = np.load(boundary_file)
      boundary = boundary.astype(np.float32)
      boundary = boundary[1:-1,1:-1]
      if subdomain is not None:
        boundary = numpy_utils.mobius_extract(boundary, subdomain)
                                              #padding_type=self.padding_type)
    # get mask for loss
    """
    mask_in = np.ones(self.sim_shape + [1])
    if subdomain is not None:
      mask_in = numpy_utils.mobius_extract(mask_in, subdomain, 
                                        padding_type=self.padding_type)
    mask_out = (1.0 - mask_in)
    boundary = np.concatenate([boundary, mask_in, mask_out], axis=-1)
    """

    return boundary

  def read_state(self, iteration, subdomain=None):
    # load flow file
    state_file = self.iter_to_cpoint(iteration)
    state = np.load(state_file)
    state = state.f.dist0a[:,1:-1,1:self.sim_shape[0]+1]
    state = state.astype(np.float32)
    state = np.swapaxes(state, 0, 1)
    state = np.swapaxes(state, 1, 2)
    state = self.DxQy.subtract_lattice(state)
    if subdomain is not None:
      state = numpy_utils.mobius_extract(state, subdomain)
                                         #padding_type=self.padding_type)
    return state

  def read_vel_rho(self, iteration, subdomain=None):
    state = self.read_state(iteration, subdomain)
    vel = self.DxQy.lattice_to_vel(state)
    rho = self.DxQy.lattice_to_rho(state)
    return vel, rho

class TrainSailfishRunner(SailfishRunner):

  def __init__(self, config, save_dir, script_name):
    SailfishRunner.__init__(self, config, save_dir, script_name)
    self.num_cpoints = config.max_sim_iters
    # more configs will probably be added later

  def read_train_data(self, state_subdomain, boundary_subdomain, boundary_small_subdomain, seq_state_subdomain, seq_length, augment=True):

    # read state
    state_files = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    ind = np.random.randint(1, len(state_files) - seq_length)
    state = self.read_state(ind, state_subdomain)

    # read boundary
    boundary = self.read_boundary(boundary_subdomain)
    boundary_small = self.read_boundary(boundary_small_subdomain)

    # read seq states
    seq_state = []
    for i in xrange(seq_length):
      seq_state.append(self.read_state(ind + i, seq_state_subdomain))

    # rotate data possibly
    if augment:
      flip = np.random.randint(0,2)
      """
      if flip == 1:
        state = self.DxQy.flip_lattice(state)
        seq_state = [self.DxQy.flip_lattice(lat) for lat in seq_state]
        boundary = np.flip(boundary, 0)
        boundary_small = np.flip(boundary_small, 0)
      """
      rotate=np.random.randint(0,4)
      if rotate > 0:
        state = self.DxQy.rotate_lattice(state, rotate)
        seq_state = [self.DxQy.rotate_lattice(lat, rotate) for lat in seq_state]
        boundary = np.rot90(boundary, k=rotate, axes=(0,1))
        boundary_small = np.rot90(boundary_small, k=rotate, axes=(0,1))

    return state, boundary, boundary_small, seq_state

  def generate_train_data(self):
    self.new_sim(self.num_cpoints)

  def need_to_generate(self):
    # check if need to generate train data or not
    need = False
    cpoints = self.list_cpoints()
    if len(cpoints) != self.num_cpoints:
      need = True 
    boundary_file = self.boundary_file()
    if not os.path.isfile(boundary_file):
      need = True 
    return need 

