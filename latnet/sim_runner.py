
import glob
import psutil as ps
import shutil
import os

import lattice
import utils.padding_utils as padding_utils

import numpy as np

class SimRunner:
  # generates simulation data for training 

  def __init__(self, config, save_dir, script_name):
    self.save_dir = save_dir
    self.num_cpoints = config.seq_length * 2
    self.lb_to_ln = config.lb_to_ln
    self.max_iters = config.max_sim_iters
    self.max_iters_till_next_cpoint = 5000 # hard set for now
    self.script_name = script_name

    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape

    # hard set for now
    self.max_times_called = 1000*self.num_cpoints
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

  def generate_cpoint(self):

    # possibly make sim dir
    self.make_sim_dir()

    # clean store dir
    self.clean_store_dir()

    # iters till next cpoint save
    iters_till_next_cpoint = np.random.randint(0, self.max_iters_till_next_cpoint)

    # base cmd
    cmd = ('./' + self.script_name 
         + ' --run_mode=generate_data'
         + ' --sailfish_sim_dir=' + self.save_dir + '/store/flow')

    # determine last checkpoint
    last_cpoint = self.last_cpoint()
    last_step = None
    if last_cpoint is not None:
      last_step = int(last_cpoint.split('/')[-1][5:-13])

    # add iteration till next cpoint and possible restor cpoint
    if last_step is not None:
      num_run_iters = last_step + iters_till_next_cpoint + (self.num_cpoints * self.lb_to_ln)
      if num_run_iters < self.max_iters:
        cmd += ' --checkpoint_from=' + str(last_step + iters_till_next_cpoint)
        cmd += ' --restore_from=' + last_cpoint[:-13]
        cmd += ' --restore_geometry=True'
        cmd += ' --max_sim_iters=' + str(num_run_iters)
      else:
        self.clean_save_dir()
        self.make_sim_dir()
        cmd += ' --max_sim_iters=' + str(iters_till_next_cpoint + (self.num_cpoints * self.lb_to_ln))
        cmd += ' --checkpoint_from=' + str(iters_till_next_cpoint)
    else:
      cmd += ' --max_sim_iters=' + str(iters_till_next_cpoint + (self.num_cpoints * self.lb_to_ln))
      cmd += ' --checkpoint_from=' + str(iters_till_next_cpoint)
 
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
    if len(glob.glob(self.save_dir + "/*.0.cpoint.npz")) != self.num_cpoints:
      self.generate_cpoint()

  def read_geometry(self, pos, radius):

    # if read geometry too many times generate new data
    self.times_called += 1
    if self.times_called > self.max_times_called:
      self.generate_cpoint()
      self.times_called = 0

    geometry_file = self.save_dir + "/flow_geometry.npy"
    geometry_array = None
    if os.path.isfile(geometry_file):
      geometry_array = np.load(geometry_file)
      geometry_array = geometry_array.astype(np.float32)
      geometry_array = geometry_array[1:-1,1:-1]
      geometry_array = padding_utils.mobius_extract_pad_2(geometry_array, pos, size=[2*radius,2*radius], padding=0)
    return geometry_array

  def read_seq_states(self, seq_length, pos, radius, padding_decrease_seq):
    # load flow file
    state_out = []
    state_in = None
    state_files = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    state_files.sort()
    subtract_weights = lattice.get_weights_numpy(9).reshape(1,1,9)
    if len(state_files) >= seq_length:
      start_pos = np.random.randint(0, len(state_files) - seq_length)
      for i in xrange(seq_length):
        state = np.load(state_files[i])
        state = state.f.dist0a[:,1:-1,1:self.sim_shape[0]+1]
        state = state.astype(np.float32)
        state = np.swapaxes(state, 0, 1)
        state = np.swapaxes(state, 1, 2)
        state = state - subtract_weights
        if i == 0:
          state_in = padding_utils.mobius_extract_pad_2(state, pos, size=[2*radius, 2*radius], padding=0)
        state = padding_utils.mobius_extract_pad(state, pos, radius - padding_decrease_seq[i])
        state = padding_utils.mobius_extract_pad_2(state, pos, size=[2*(radius - padding_decrease_seq[i)), 2*(radius - padding_decrease_seq[i])], padding=0)
        state_out.append(state)
    return state_in, state_out  

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



