
import sys

# import latnet files
import utils.padding_utils as padding_utils

# import sailfish
sys.path.append('../sailfish')
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTEquilibriumVelocity, NTEquilibriumDensity, DynamicValue, NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_base import ForceObject
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S

# import external librarys
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import *


class Domain(object):

  def __init__(self, config, nr_downsamples=0): # once network config is in correctly will just need config

    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape

    self.compressed_sim_shape = [sim_shape[0]/np.power(2, nr_downsamples), 
                                 sim_shape[1]/np.power(2, nr_downsamples)]

    input_shape = config.input_shape.split('x')
    input_shape = map(int, input_shape)
    self.input_shape = input_shape

    compressed_shape = config.compressed_shape.split('x')
    compressed_shape = map(int, compressed_shape)
    self.compressed_shape = compressed_shape

    self.sailfish_sim_dir = config.sailfish_sim_dir
    self.max_sim_iters = config.max_sim_iters
    self.lb_to_ln = config.lb_to_ln
    self.visc = config.visc
    self.restore_geometry = config.restore_geometry

  def boundary_conditions(self, hx, hy):
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
    input_geometry = np.concatenate([np.expand_dims(where_boundary, axis=-1).astype(np.float32),
                                     np.array(velocity).reshape(1,1,2) * np.expand_dims(where_velocity, axis=-1).astype(np.float32),
                                     density *  np.expand_dims(where_density, axis=-1).astype(np.float32)], axis=-1)
    return input_geometry

  def create_sailfish_simulation(self):

    # I think I can fix these problems with inheritence but screw it for now
    # boundary conditions
    geometry_boundary_conditions = self.geometry_boundary_conditions
    velocity_boundary_conditions = self.velocity_boundary_conditions
    density_boundary_conditions = self.density_boundary_conditions

    # init conditions
    velocity_initial_conditions = self.velocity_initial_conditions
    density_initial_conditions = self.density_initial_conditions

    # update defaults
    shape = self.sim_shape
    sailfish_sim_dir = self.sailfish_sim_dir
    max_iters = self.max_sim_iters
    lb_to_ln = self.lb_to_ln
    visc = self.visc
    restore_geometry = self.restore_geometry

    # inportant func
    make_geometry_input = self.make_geometry_input

    class SailfishSubdomain(Subdomain2D):
      
      bc = NTFullBBWall

      def boundary_conditions(self, hx, hy):

        # restore from old dir or make new geometry
        if restore_geometry:
          restore_boundary_conditions = np.load(sailfish_sim_dir[:-10] + "flow_geometry.npy")
          where_boundary = restore_boundary_conditions[:,:,0].astype(np.bool)
          where_velocity = restore_boundary_conditions[:,:,1].astype(np.bool)
          velocity = (restore_boundary_conditions[-1,-1,1], restore_boundary_conditions[-1,-1,2])
          where_density  = restore_boundary_conditions[:,:,3].astype(np.bool)
          density = 1.0
        else:
          where_boundary = geometry_boundary_conditions(hx, hy, [self.gx, self.gy])
          where_velocity, velocity = velocity_boundary_conditions(hx, hy, [self.gx, self.gy])
          where_density, density = density_boundary_conditions(hx, hy, [self.gx, self.gy])

        # set boundarys
        self.set_node(where_boundary, self.bc)

        # set velocities
        self.set_node(where_velocity, NTEquilibriumVelocity(velocity))

        # set densitys
        self.set_node(where_density, NTEquilibriumDensity(density))

        # save geometry
        save_geometry = make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
        np.save(sailfish_sim_dir + "_geometry.npy", save_geometry)

      def initial_conditions(self, sim, hx, hy):
        # set start density
        rho = density_initial_conditions(hx, hy,  [self.gx, self.gy])
        sim.rho[:] = rho

        # set start velocity
        vel = velocity_initial_conditions(hx, hy,  [self.gx, self.gy])
        sim.vx[:] = vel[0]
        sim.vy[:] = vel[1]
   
    class SailfishSimulation(LBFluidSim): 
      subdomain = SailfishSubdomain

      
      @classmethod
      def add_options(cls, group, dim):
        group.add_argument('--sailfish_sim_dir', help='all modes', type=str,
                              default='')
        group.add_argument('--run_mode', help='all modes', type=str,
                              default='')
        group.add_argument('--max_sim_iters', help='all modes', type=int,
                              default=1000)
        group.add_argument('--restore_geometry', help='all modes', type=bool,
                              default=False)

      @classmethod
      def update_defaults(cls, defaults):
        defaults.update({
          'max_iters': max_iters,
          'output_format': 'npy',
          'periodic_y': True,
          'periodic_x': True,
          'checkpoint_file': sailfish_sim_dir,
          'checkpoint_every': lb_to_ln,
          'lat_nx': shape[0],
          'lat_ny': shape[1]
          })

      @classmethod
      def modify_config(cls, config):
        config.visc   = visc
        config.mode   = "batch"

      def __init__(self, *args, **kwargs):
        super(SailfishSimulation, self).__init__(*args, **kwargs)

    ctrl = LBSimulationController(SailfishSimulation)

    return ctrl

  def compute_compressed_state(self, sess, encoder, state, padding):

    self.input_lattice_subdomains = (int(math.ceil(self.sim_shape[0]/float(self.input_shape[0]))), 
                                     int(math.ceil(self.sim_shape[1]/float(self.input_shape[1]))))
    compressed_state = []
    for i in tqdm(xrange(self.input_lattice_subdomains[0])):
      compressed_state_store = []
      for j in xrange(self.input_lattice_subdomains[1]):
        # hard set to zero for now
        zero_state = np.zeros([1] + [self.input_shape[0]+2*padding] +  [self.input_shape[1]+2*padding] + [9])
        compressed_state_store.append(sess.run(encoder, feed_dict={state:zero_state}))
      compressed_state.append(np.concatenate(compressed_state_store, axis=2))
    compressed_state = np.concatenate(compressed_state, axis=1)

    # trim edges
    compressed_state = compressed_state[:,:self.compressed_sim_shape[0],:self.compressed_sim_shape[1]]

    return compressed_state

  def compute_compressed_boundary(self, sess, encoder, boundary, padding):

    self.input_lattice_subdomains = (int(math.ceil(self.sim_shape[0]/float(self.input_shape[0]))), 
                                     int(math.ceil(self.sim_shape[1]/float(self.input_shape[1]))))

    compressed_boundary = []
    for i in tqdm(xrange(self.input_lattice_subdomains[0])):
      compressed_boundary_store = []
      for j in xrange(self.input_lattice_subdomains[1]):
        h = np.mgrid[i*self.input_shape[0] - padding:(i+1)*self.input_shape[0] + padding,
                     j*self.input_shape[1] - padding:(j+1)*self.input_shape[1] + padding]
        hx = np.mod(h[1], self.sim_shape[0])
        hy = np.mod(h[0], self.sim_shape[1])
        where_boundary = self.geometry_boundary_conditions(hx, hy, self.sim_shape)
        where_velocity, velocity = self.velocity_boundary_conditions(hx, hy, self.sim_shape)
        where_density, density = self.density_boundary_conditions(hx, hy, self.sim_shape)
        input_geometry = self.make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
        compressed_boundary_store.append(sess.run(encoder, feed_dict={boundary:np.expand_dims(input_geometry, axis=0)}))

      compressed_boundary.append(np.concatenate(compressed_boundary_store, axis=2))

    compressed_boundary = np.concatenate(compressed_boundary, axis=1)
    
    # trim edges
    compressed_boundary = compressed_boundary[:,:self.compressed_sim_shape[0],:self.compressed_sim_shape[1]]

    return compressed_boundary

  def compute_compressed_mapping(self, sess, compression_mapping, compressed_state, compressed_boundary, np_compressed_state, np_compressed_boundary, padding):

    compressed_shape = np_compressed_state.shape[1:3]
    self.input_lattice_subdomains = (int(math.ceil(compressed_shape[0]/float(self.compressed_shape[0]))), 
                                     int(math.ceil(compressed_shape[1]/float(self.compressed_shape[1]))))

    np_compressed_state_out = []
    for i in tqdm(xrange(self.input_lattice_subdomains[0])):
      compressed_state_store = []
      for j in xrange(self.input_lattice_subdomains[1]):
        pos = (i*self.compressed_shape[0], j*self.compressed_shape[1])
        #plt.imshow(np_compressed_state[0,:,:,0])
        #plt.show()
        compressed_state_ij    = padding_utils.mobius_extract_pad_2(np_compressed_state,    pos, self.compressed_shape, padding, has_batch=True)
        #plt.imshow(compressed_state_ij[0,:,:,0])
        #plt.show()
        compressed_boundary_ij = padding_utils.mobius_extract_pad_2(np_compressed_boundary, pos, self.compressed_shape, padding, has_batch=True)
        compressed_state_store.append(sess.run(compression_mapping, feed_dict={compressed_state:compressed_state_ij, compressed_boundary:compressed_boundary_ij}))

      np_compressed_state_out.append(np.concatenate(compressed_state_store, axis=2))

    np_compressed_state = np.concatenate(np_compressed_state_out, axis=1)

    # trim edges
    np_compressed_state = np_compressed_state[:,:self.compressed_sim_shape[0],:self.compressed_sim_shape[1]]

    return np_compressed_state

  def compute_compressed_boundary_mapping(self, sess, compression_mapping, compressed_state, compressed_boundary, np_compressed_state, np_compressed_boundary, padding):

    compressed_shape = np_compressed_state.shape[1:3]
    self.input_lattice_subdomains = (int(math.ceil(compressed_shape[0]/float(self.compressed_shape[0]))), 
                                     int(math.ceil(compressed_shape[1]/float(self.compressed_shape[1]))))

    np_compressed_state_out = []
    for i in tqdm(xrange(self.input_lattice_subdomains[0])):
      compressed_state_store = []
      for j in xrange(self.input_lattice_subdomains[1]):
        pos = (i*self.compressed_shape[0], j*self.compressed_shape[1])
        #plt.imshow(np_compressed_state[0,:,:,0])
        #plt.show()
        compressed_state_ij    = padding_utils.mobius_extract_pad_2(np_compressed_state,    pos, self.compressed_shape, padding, has_batch=True)
        #plt.imshow(compressed_state_ij[0,:,:,0])
        #plt.show()
        compressed_boundary_ij = padding_utils.mobius_extract_pad_2(np_compressed_boundary, pos, self.compressed_shape, padding, has_batch=True)
        compressed_state_store.append(sess.run(compression_mapping, feed_dict={compressed_state:compressed_state_ij, compressed_boundary:compressed_boundary_ij}))

      np_compressed_state_out.append(np.concatenate(compressed_state_store, axis=2))

    np_compressed_state = np.concatenate(np_compressed_state_out, axis=1)

    # trim edges
    np_compressed_state = np_compressed_state[:,:self.compressed_sim_shape[0],:self.compressed_sim_shape[1]]

    return np_compressed_state

  def compute_decompressed_state(self, sess, decoder, compressed_state, compressed_boundary, np_compressed_state, np_compressed_boundary, padding):

    compressed_shape = np_compressed_state.shape[1:3]
    self.input_lattice_subdomains = (int(math.ceil(compressed_shape[0]/float(self.compressed_shape[0]))), 
                                     int(math.ceil(compressed_shape[1]/float(self.compressed_shape[1]))))


    np_state_out = []
    for i in tqdm(xrange(self.input_lattice_subdomains[0])):
      state_store = []
      for j in xrange(self.input_lattice_subdomains[1]):
        print("CALLED")
        pos = (i*self.compressed_shape[0], j*self.compressed_shape[1])
        compressed_state_ij    = padding_utils.mobius_extract_pad_2(np_compressed_state,    pos, self.compressed_shape, padding, has_batch=True)
        compressed_boundary_ij = padding_utils.mobius_extract_pad_2(np_compressed_boundary, pos, self.compressed_shape, padding, has_batch=True)
        state_store.append(sess.run(decoder, feed_dict={compressed_state:compressed_state_ij, compressed_boundary:compressed_boundary_ij}))

      np_state_out.append(np.concatenate(state_store, axis=2))

    np_state = np.concatenate(np_state_out, axis=1)

    # trim edges
    #np_state = np_state[:,:self.sim_shape[0],:self.sim_shape[1]]

    return np_state

  def get_state_input(self, pos, radius):
    pass

  def get_boundary_input(self, pos, radius):
    pass

  def generate_compressed_state(self, sess, blaa):
    pass

  def update_compressed_state(self, sess, blaa):
    pass

  def extract_state(self, pos, radius, sess, blaa):
    pass

class SubDomain(object):
  # probably add more to this class later :/
  def __init__(self, pos):
    self.pos = pos

