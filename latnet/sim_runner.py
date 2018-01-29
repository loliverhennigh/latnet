
import glob
import psutil as ps
import shutil
import os

import lattice
import utils.numpy_utils as numpy_utils


import matplotlib.pyplot as plt

import numpy as np



class SimRunner:
  # generates simulation data for training 

  def __init__(self, config, save_dir, script_name):
    self.save_dir = save_dir
    self.num_cpoints = 30
    self.lb_to_ln = config.lb_to_ln
    self.max_sim_iters = config.max_sim_iters
    self.script_name = script_name
    self.seq_length = config.seq_length

    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape
    self.DxQy = lattice.TYPES[config.DxQy]

    # hard set for now
    self.max_times_called = np.random.randint(200,800)*(self.num_cpoints/config.seq_length)
    self.times_called = 0

  def last_cpoint(self):
    cpoints = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    cpoints.sort()
    if len(cpoints):
      return cpoints[-1]
    else:
      return None

  def first_cpoint(self):
    cpoints = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    cpoints.sort()
    if len(cpoints):
      return cpoints[-1]
    else:
      return None

  def run_sailfish_sim(script_name, save_dir, num_iters, lb_to_ln):
    cmd = ('./' + script_name 
         + ' --run_mode=generate_data'
         + ' --train_sim_dir=' + save_dir
         + ' --max_sim_iters=' + str(lb_to_ln*num_iters)
         + ' --checkpoint_from=0')
    p = ps.subprocess.Popen(cmd.split(' '), 
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
    p.communicate()

  def generate_cpoint(self):

    # possibly make sim dir
    self.make_sim_dir()

    # clean store dir
    self.clean_store_dir()

    # base cmd
    cmd = ('./' + self.script_name 
         + ' --run_mode=generate_data'
         + ' --train_sim_dir=' + self.save_dir + '/store/flow')

    # determine last checkpoint
    last_cpoint = self.last_cpoint()
    last_step = None
    if last_cpoint is not None:
      last_step = int(last_cpoint.split('/')[-1][5:-13])

    # add iteration till next cpoint and possible restor cpoint
    if last_step is not None:
      num_run_iters = last_step + (self.num_cpoints * self.lb_to_ln)
      if num_run_iters < self.max_sim_iters:
        cmd += ' --checkpoint_from=' + str(last_step)
        cmd += ' --restore_from=' + last_cpoint[:-13]
        cmd += ' --restore_geometry=True'
        cmd += ' --max_sim_iters=' + str(num_run_iters)
      else:
        self.clean_save_dir()
        self.make_sim_dir()
        cmd += ' --max_sim_iters=' + str(self.num_cpoints * self.lb_to_ln)
        cmd += ' --checkpoint_from=' + str(0)
    else:
      cmd += ' --max_sim_iters=' + str(self.num_cpoints * self.lb_to_ln)
      cmd += ' --checkpoint_from=' + str(0)
 
    # run cmd
    """
    with open(os.devnull, 'w') as devnull:
      p = ps.subprocess.Popen(cmd.split(' '), 
                              stdout=devnull, stderr=devnull, 
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
      p.communicate()
    """
    p = ps.subprocess.Popen(cmd.split(' '), 
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
    p.communicate()

    # mv cpoints over
    self.clean_prev_cpoints()

    # if no cpoints in dir then need to restart simulation
    print(len(glob.glob(self.save_dir + "/*.0.cpoint.npz")))
    if len(glob.glob(self.save_dir + "/*.0.cpoint.npz")) != self.num_cpoints:
      self.generate_cpoint()

  def read_data(self, state_subdomain, geometry_subdomain, seq_state_subdomain):

    # if read geometry too many times generate new data
    self.times_called += 1
    if self.times_called > self.max_times_called:
      self.generate_cpoint()
      self.times_called = 0


    # read state
    state_files = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    ind = np.random.randint(0, len(state_files) - self.seq_length)
    state = self.read_state(ind, state_subdomain)

    # read geometry
    geometry = self.read_geometry(geometry_subdomain)

    # read seq states
    seq_state = []
    for i in xrange(self.seq_length):
      seq_state.append(self.read_state(ind + i, seq_state_subdomain[i]))

    return state, geometry, seq_state

  def read_geometry(self, subdomain):
    geometry_file = self.save_dir + "/flow_geometry.npy"
    geometry = None
    if os.path.isfile(geometry_file):
      geometry = np.load(geometry_file)
      geometry = geometry.astype(np.float32)
      geometry = geometry[1:-1,1:-1]
      geometry = numpy_utils.mobius_extract(geometry, subdomain)
    return geometry

  def read_state(self, ind, subdomain):
    # load flow file
    state_files = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    state_files.sort()
    state = np.load(state_files[ind])
    state = state.f.dist0a[:,1:-1,1:self.sim_shape[0]+1]
    state = state.astype(np.float32)
    state = np.swapaxes(state, 0, 1)
    state = np.swapaxes(state, 1, 2)
    state = self.DxQy.subtract_lattice(state)
    state = numpy_utils.mobius_extract(state, subdomain)
    return state

  def clean_store_dir(self):
    files = glob.glob(self.save_dir + "/store/*")
    for f in files:
      with open(os.devnull, 'w') as devnull:
        p = ps.subprocess.Popen(["rm", f], 
                                 stdout=devnull, stderr=devnull)
        p.communicate()

  def clean_prev_cpoints(self):
    old_cpoints = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    for c in old_cpoints:
      with open(os.devnull, 'w') as devnull:
        p = ps.subprocess.Popen(["rm", c], 
                                 stdout=devnull, stderr=devnull)
        p.communicate()
    new_cpoints = glob.glob(self.save_dir + "/store/*.0.cpoint.npz")
    if len(new_cpoints) == self.num_cpoints:
      for c in new_cpoints:
        #with open(os.devnull, 'w') as devnull:
        #  p = ps.subprocess.Popen(["mv", c, self.save_dir + "/"], 
        #                           stdout=devnull, stderr=devnull)
        #  p.communicate()
        p = ps.subprocess.Popen(["mv", c, self.save_dir + "/"])
        p.communicate()
      # move geometry
      with open(os.devnull, 'w') as devnull:
        #p = ps.subprocess.Popen(["mv", self.save_dir + "/store/flow_geometry.npy", self.save_dir + "/"], 
        #                         stdout=devnull, stderr=devnull)
        #p.communicate()
        p = ps.subprocess.Popen(["mv", self.save_dir + "/store/flow_geometry.npy", self.save_dir + "/"])
        p.communicate()

  def clean_save_dir(self):
    shutil.rmtree(self.save_dir)

  def make_sim_dir(self):
    with open(os.devnull, 'w') as devnull:
      p = ps.subprocess.Popen(('mkdir -p ' + self.save_dir + "/store").split(' '), 
                               stdout=devnull, stderr=devnull)
      p.communicate()



