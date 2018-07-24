
import numpy as np
import math
import itertools
from tqdm import *
from copy import copy
import os
import sys
import glob
import psutil as ps
import time
import matplotlib.pyplot as plt

import lattice
import utils.numpy_utils as numpy_utils
from shape_converter import SubDomain

sys.path.append('../../sailfish')
from sailfish.subdomain import Subdomain2D, Subdomain3D
from sailfish.node_type import NTEquilibriumVelocity, NTFullBBWall, NTDoNothing, NTZouHeVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.lb_base import LBForcedSim

class SailfishWrapper(object):

  def __init__(self, config, save_dir):
    self.config = config
    self.save_dir = save_dir
    self.DxQy = lattice.TYPES[config.DxQy]()

  @classmethod
  def update_defaults(cls, defaults):
      pass

  def geometry_boundary_conditions(self, hx, hy, shape):
    pass

  def velocity_boundary_conditions(self, hx, hy, shape):
    pass

  def density_boundary_conditions(self, hx, hy, shape):
    pass

  def velocity_initial_conditions(self, hx, hy, shape):
    pass

  def density_initial_conditions(self, hx, hy, shape):
    pass

  def make_geometry_input(self, where_boundary, velocity, where_velocity, density, where_density):
    # TODO Clean this
    boundary_array = np.expand_dims(where_boundary, axis=-1).astype(np.float32)
    velocity_array = np.array(velocity).reshape(len(where_velocity.shape) * [1] + [self.DxQy.dims])
    velocity_array = velocity_array * np.expand_dims(where_velocity, axis=-1).astype(np.float32)
    density_array = density * np.expand_dims(where_density, axis=-1).astype(np.float32)
    force_array = 1e5 * np.array(self.force) * np.ones_like(velocity_array).astype(np.float32) # 1e5 to scale force to same range as vel

    input_geometry = np.concatenate([boundary_array,
                                     velocity_array,
                                     density_array,
                                     force_array], axis=-1)
    return input_geometry

  def make_sailfish_subdomain(self, config):

    velocity_initial_conditions = self.velocity_initial_conditions
    density_initial_conditions = self.density_initial_conditions

    geometry_boundary_conditions = self.geometry_boundary_conditions
    velocity_boundary_conditions = self.velocity_boundary_conditions
    density_boundary_conditions = self.density_boundary_conditions

    make_geometry_input = self.make_geometry_input    
    train_sim_dir = config.train_sim_dir

    if hasattr(self, 'force'):
      dom_force = self.force
    else:
      dom_force = None

    bc = NTFullBBWall

    if self.DxQy.dims == 2:
      class SailfishSubdomain(Subdomain2D):

        if dom_force is not None:
          force = dom_force
          
        def boundary_conditions(self, hx, hy):
  
          # restore from old dir or make new geometry
          if config.restore_geometry:
            restore_boundary_conditions = np.load(train_sim_dir[:-10] + "boundary.npy")
            where_boundary = restore_boundary_conditions[...,0].astype(np.bool)
            where_velocity = np.logical_or(restore_boundary_conditions[...,1].astype(np.bool), restore_boundary_conditions[...,2].astype(np.bool))
            if len(np.where(where_velocity)[0]) == 0:
              velocity = (0.0,0.0)
            else:
              velocity = (restore_boundary_conditions[np.where(where_velocity)[0][0], np.where(where_velocity)[1][0], 1],
                          restore_boundary_conditions[np.where(where_velocity)[0][0], np.where(where_velocity)[1][0], 2])
            where_density  = restore_boundary_conditions[...,3].astype(np.bool)
            density = 1.0
            #self.force = (restore_boundary_conditions[0,0,4], restore_boundary_conditions[0,0,5])
          else:
            where_boundary = geometry_boundary_conditions(hx, hy, [self.gx, self.gy])
            where_velocity, velocity = velocity_boundary_conditions(hx, hy, [self.gx, self.gy])
            where_density, density = density_boundary_conditions(hx, hy, [self.gx, self.gy])
            #self.force = force
  
          # set boundarys
          self.set_node(where_boundary, bc)
  
          # set velocities
          self.set_node(where_velocity, NTEquilibriumVelocity(velocity))
          #self.set_node(where_velocity, NTZouHeVelocity(velocity))
  
          # set densitys
          self.set_node(where_density, NTDoNothing)
  
          # save geometry
          save_geometry = make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
          np.save(train_sim_dir[:-10] + "boundary.npy", save_geometry)
  
        def initial_conditions(self, sim, hx, hy):
          # set start density
          rho = density_initial_conditions(hx, hy,  [self.gx, self.gy])
          sim.rho[:] = rho
  
          # set start velocity
          vel = velocity_initial_conditions(hx, hy,  [self.gx, self.gy])
          sim.vx[:] = vel[0]
          sim.vy[:] = vel[1]
   


    elif self.DxQy.dims == 3:
      class SailfishSubdomain(Subdomain3D):
  
        def boundary_conditions(self, hx, hy, hz):
  
          # restore from old dir or make new geometry
          if config.restore_geometry:
            restore_boundary_conditions = np.load(train_sim_dir[:-10] + "boundary.npy")
            where_boundary = restore_boundary_conditions[...,0].astype(np.bool)
            where_velocity = np.logical_or(restore_boundary_conditions[...,1].astype(np.bool), restore_boundary_conditions[...,2].astype(np.bool), restore_boundary_conditions[...,3].astype(np.bool))
            vel_ind = np.where(where_velocity)
            velocity = restore_boundary_conditions[vel_ind[0][0], vel_ind[1][0], vel_ind[2][0], :]
            velocity = (velocity[1], velocity[2], velocity[3])
            where_density  = restore_boundary_conditions[...,4].astype(np.bool)
            density = 1.0
          else:
            where_boundary = geometry_boundary_conditions(hx, hy, hz, [self.gx, self.gy, self.gz])
            where_velocity, velocity = velocity_boundary_conditions(hx, hy, hz, [self.gx, self.gy, self.gz])
            where_density, density = density_boundary_conditions(hx, hy, hz, [self.gx, self.gy, self.gz])
  
          # set boundarys
          self.set_node(where_boundary, bc)
  
          # set velocities
          self.set_node(where_velocity, NTEquilibriumVelocity(velocity))
          #self.set_node(where_velocity, NTZouHeVelocity(velocity))
  
          # set densitys
          self.set_node(where_density, NTDoNothing)
  
          # save geometry
          save_geometry = make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
          np.save(train_sim_dir[:-10] + "boundary.npy", save_geometry)
  
        def initial_conditions(self, sim, hx, hy):
          # set start density
          rho = density_initial_conditions(hx, hy,  [self.gx, self.gy])
          sim.rho[:] = rho
  
          # set start velocity
          vel = velocity_initial_conditions(hx, hy,  [self.gx, self.gy])
          sim.vx[:] = vel[0]
          sim.vy[:] = vel[1]
          sim.vy[:] = vel[1]

    return SailfishSubdomain

  def create_sailfish_simulation(self, config):

    # update defaults
    shape = self.sim_shape
    train_sim_dir = self.train_sim_dir
    max_iters = self.max_sim_iters
    lb_to_ln = self.lb_to_ln
    visc = config.visc
    periodic_x = self.periodic_x
    periodic_y = self.periodic_y
    if len(shape) == 3:
      periodic_z = self.periodic_z
    restore_geometry = config.restore_geometry
    mode = config.mode
    subgrid = config.subgrid

    class SailfishSimulation(LBFluidSim, LBForcedSim): 
      subdomain = self.make_sailfish_subdomain(config)
      
      @classmethod
      def add_options(cls, group, dim):
        group.add_argument('--domain_name', help='all modes', type=str,
                              default='')
        group.add_argument('--train_sim_dir', help='all modes', type=str,
                              default='')
        group.add_argument('--sim_dir', help='all modes', type=str,
                              default='')
        group.add_argument('--run_mode', help='all modes', type=str,
                              default='')
        group.add_argument('--max_sim_iters', help='all modes', type=int,
                              default=1000)
        group.add_argument('--restore_geometry', help='all modes', type=bool,
                              default=False)
        group.add_argument('--lb_to_ln', help='all modes', type=int,
                              default=60)

      @classmethod
      def update_defaults(cls, defaults):
        defaults.update({
          'mode': mode,
          'precision': 'half',
          'subgrid': config.subgrid,
          'periodic_x': periodic_x,
          'periodic_y': periodic_y,
          'lat_nx': shape[1],
          'lat_ny': shape[0],
          'checkpoint_from': 0,
          'visc': visc
          })
        if len(shape) == 3:
          defaults.update({
            'grid': 'D3Q15',
            'periodic_z': periodic_z,
            'lat_nz': shape[2]
          })
        if mode is not 'visualization':
          defaults.update({
            'output_format': 'npy',
            'max_iters': max_iters,
            'checkpoint_file': train_sim_dir,
            'checkpoint_every': lb_to_ln
          })

      @classmethod
      def modify_config(cls, config):
        config.visc   = visc

      def __init__(self, *args, **kwargs):
        super(SailfishSimulation, self).__init__(*args, **kwargs)
        if hasattr(self.subdomain, 'force'):
          self.add_body_force(self.subdomain.force)

    ctrl = LBSimulationController(SailfishSimulation)

    return ctrl

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

    if not self.debug_sailfish:
      cmd = ('./' + self.script_name 
           + ' --run_mode=generate_data'
           + ' --domain_name=' + self.name
           + ' --max_sim_iters=' + str(self.lb_to_ln*num_iters + 1)
           + ' --train_sim_dir=' + self.save_dir + '/store/flow')
    else:
      cmd = ('./' + self.script_name 
           + ' --domain_name=' + self.name
           + ' --mode=visualization'
           + ' --run_mode=generate_data')
    print(cmd)
    p = ps.subprocess.Popen(cmd.split(' '), 
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
    p.communicate()
   
    self.mv_store_dir()
  
  def restart_sim(self, num_iters, keep_old=False):

    assert self.is_restorable(), "trying to restart sim without finding proper save"
    self.clean_store_dir()

    last_state, last_iter = self.last_state_filename()

    cmd = ('./' + self.script_name 
         + ' --run_mode=generate_data'
         + ' --domain_name=' + self.name
         + ' --max_sim_iters=' + str(self.latnet_iter_to_sailfish_iter(num_iters
                                                                    + last_iter) + 1)
         + ' --restore_geometry=True'
         + ' --restore_from=' + last_state[:-13])
    if self.debug_sailfish:
      cmd += ' --mode=visualization'
      cmd += ' --scr_scale=.5'
    else:
      cmd += ' --train_sim_dir=' + self.save_dir + '/store/flow'
    print(cmd)
    p = ps.subprocess.Popen(cmd.split(' '), 
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
    p.communicate()
  
    if not keep_old:
      self.clean_dir()
    self.mv_store_dir()
 
  def is_restorable(self):
    state_filenames = self.list_states_filenames()
    boundary_filename = self.boundary_filename()
    return ((len(state_filenames) > 0) and os.path.isfile(boundary_filename))

  def list_state_filenames(self):
    states = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    states.sort()
    return states

  def list_cstate_filenames(self):
    cstates = glob.glob(self.save_dir + "/*.0.comp.npy")
    cstates.sort()
    return cstates

  def boundary_filename(self):
    return self.save_dir + "/boundary.npy"

  def cboundary_filename(self):
    return self.save_dir + "/cboundary.npy"

  def first_state_filename(self):
    state_filenames = self.list_state_filenames()
    return state_filenames[0], self.state_filename_to_iter(state_filenames[0])

  def first_cstate_filename(self):
    cstate_filenames = self.list_cstate_filenames()
    return cstate_filenames[0], self.cstate_filename_to_iter(cstate_filenames[0])

  def last_state_filename(self):
    state_filenames = self.list_state_filenames()
    return state_filenames[-1], self.state_filename_to_iter(state_filenames[-1])

  def last_cstate_filename(self):
    cstate_filenames = self.list_cstate_filenames()
    return cstate_filenames[-1], self.cstate_filename_to_iter(cstate_filenames[-1])

  def state_filename_to_iter(self, state_filename):
    sailfish_iter = int(state_filename.split('.')[-4])
    return self.sailfish_iter_to_latnet_iter(sailfish_iter)

  def cstate_filename_to_iter(self, state_filename):
    sailfish_iter = int(cstate_filename.split('.')[-4])
    return self.sailfish_iter_to_latnet_iter(sailfish_iter)

  def iter_to_state_filename(self, iteration):
    sailfish_iter = self.latnet_iter_to_sailfish_iter(iteration)
    zpadding = len(self.last_state_filename()[0].split('.')[-4])
    state_filename = (self.save_dir + '/flow.' 
                    + str(sailfish_iter).zfill(zpadding)
                    + '.0.cpoint.npz')
    return state_filename

  def iter_to_cstate_filename(self, iteration):
    sailfish_iter = self.latnet_iter_to_sailfish_iter(iteration)
    zpadding = len(self.last_state_filename()[0].split('.')[-4])
    cstate_filename = (self.save_dir + '/flow.' 
                    + str(sailfish_iter).zfill(zpadding)
                    + '.0.comp.npy')
    return cstate_filename

class JHTDBWrapper(object):
  def __init__(self, config, save_dir):
    self.config = config
    self.save_dir = save_dir
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.wrapper_name = 'JHTDB'

  def add_options(cls, group, dim):
    pass

  def make_url(self, subdomain, iteration):
    url_begining = "http://dsp033.pha.jhu.edu/jhtdb/getcutout/com.gmail.loliverhennigh101-4da6ce46/isotropic1024coarse/p,u/" 
    url_end =  str(iteration * self.step_ratio) + ",1/"
    url_end += str(subdomain.pos[0]) + ","
    url_end += str(subdomain.size[0]) + "/"
    url_end += str(subdomain.pos[1]) + ","
    url_end += str(subdomain.size[1]) + "/"
    url_end += str(subdomain.pos[2]) + ","
    url_end += str(subdomain.size[2]) + "/hdf5/"
    print(url_end)
    return url_begining + url_end

  def download_state(self, iteration, subdomain):
    filename = self.iter_to_state_filename(iteration, subdomain)
    path_filename = self.save_dir + filename
    if not os.path.isfile(path_filename):
      r = ''
      while (r == ''):
        try:
          r = requests.get(self.make_url(subdomain, iteration)) 
          #print(r)
        except:
          print("having trouble getting data, will sleep and try again")
          time.sleep(2)
        if type(r) is not str:
          if r.status_code == 500:
            r = ''
      with open(path_filename, 'wb') as f:
        f.write(r.content)

  def download_boundary(self, subdomain, iteration):
    pass

  def iter_to_state_filename(self, iteration, subdomain):
    filename =  "/state_iteration_" + str(iteration) 
    filename += "pos_" + str(subdomain.pos[0]) + '_' + str(subdomain.pos[1]) +  '_' + str(subdomain.pos[2]) + '_'
    filename += "size_" + str(subdomain.size[0]) + '_' + str(subdomain.size[1]) + '_' + str(subdomain.size[2])
    filename += ".h5"
    return filename

  def iter_to_cstate_filename(self, iteration, subdomain):
    filename =  "/cstate_iteration_" + str(iteration) 
    filename += "pos_" + str(subdomain.pos[0]) + '_' + str(subdomain.pos[1]) +  '_' + str(subdomain.pos[2]) + '_'
    filename += "size_" + str(subdomain.size[0]) + '_' + str(subdomain.size[1]) + '_' + str(subdomain.size[2])
    filename += ".npy"
    return filename

