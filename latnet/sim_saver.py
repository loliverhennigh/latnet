
import lattice

import numpy as np
import matplotlib.pyplot as plt
import psutil as ps
import os
import glob

class SimSaver:

  def __init__(self, config, script_name):
    self.save_dir = config.sim_dir
    self.save_format = config.save_format
    self.save_cstate = config.save_cstate
    self.script_name = script_name
    self.DxQy = lattice.TYPES[config.DxQy]
    self.sim_restore_iter = config.sim_restore_iter
    self.lb_to_ln = config.lb_to_ln
    self.sim_shape 

    self.latnet_files = []

    if self.sim_restore_iter >= 1:
      self.start_state, self.start_boundary = self.generate_start_data()

  def generate_start_data(self):
    self.run_sailfish_sim(self.script_name, self.save_dir, 
                          self.sim_restore_iter, self.lb_to_ln)
    start_state = self.read_state(self.sim_restore_iter-1)
    start_boundary = self.read_boundary()
    return start_state, start_boundary

  def generate_comparison_data(self):
    self.run_sailfish_sim(self.script_name, self.save_dir, 
                          self.num_iters, self.sim_save_every*self.self.lb_to_ln)
    
  def read_boundary(self):
    boundary_file = self.save_dir + "/flow_geometry.npy"
    boundary = None
    if os.path.isfile(boundary_file):
      boundary = np.load(boundary_file)
      boundary = boundary.astype(np.float32)
      boundary = boundary[1:-1,1:-1]
    return boundary 

  def read_state(self, ind):
    # load flow file
    state_files = glob.glob(self.save_dir + "*.0.cpoint.npz")
    state_files.sort()
    state = np.load(state_files[ind])
    state = state.f.dist0a[:,1:-1,1:self.sim_shape[0]+1]
    state = state.astype(np.float32)
    state = np.swapaxes(state, 0, 1)
    state = np.swapaxes(state, 1, 2)
    state = self.DxQy.subtract_lattice(state)
    return state

  def save_numpy(self, iteration, vel, rho, cstate):
    file_name = self.save_dir + str(iteration).zfill(6) + ".cpoint"
    plt.imshow(state[0,:,:,0])
    plt.savefig('figs/out_state_' + str(i) + '.png')
    if self.save_cstate:
      np.savez(file_name, vel=vel, rho=rho, cstate=cstate)
    else:
      np.savez(file_name, vel=vel, rho=rho)
    self.latnet_files.append(file_name)

  def run_sailfish_sim(self, script_name, save_dir, num_iters, lb_to_ln):
    cmd = ('./' + script_name 
         + ' --run_mode=generate_data'
         + ' --train_sim_dir=' + save_dir + 'store/flow'
         + ' --max_sim_iters=' + str(lb_to_ln*num_iters)
         + ' --checkpoint_from=0')
    p = ps.subprocess.Popen(cmd.split(' '), 
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
    p.communicate()

  def visualizer(iteration, state):
    pass

  def compare_true_generated(iteration, sailfish_state, latnet_state):
    pass
