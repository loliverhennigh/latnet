
import sys

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

class Domain(object):

  def __init__(self, config):

    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape

    self.sailfish_sim_dir = config.sailfish_sim_dir
    self.max_sim_iters = config.max_sim_iters
    self.lb_to_ln = config.lb_to_ln
    self.visc = config.visc

    # hard set
    self.max_lat_shape = [1024, 1024]
    self.max_compressed_shape = [512, 512]

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

    class SailfishSubdomain(Subdomain2D):
      
      bc = NTFullBBWall

      def boundary_conditions(self, hx, hy):
        # set boundarys
        where_boundary = geometry_boundary_conditions(hx, hy, [self.gx, self.gy])
        self.set_node(where_boundary, self.bc)

        # set velocities
        where_velocity, velocity = velocity_boundary_conditions(hx, hy, [self.gx, self.gy])
        self.set_node(where_velocity, NTEquilibriumVelocity(velocity))

        # set densitys
        where_density, density = density_boundary_conditions(hx, hy, [self.gx, self.gy])
        self.set_node(where_density, NTEquilibriumDensity(density))

        # restore from old dir
        if restore_geometry:
          restore_geometry = np.load(sailfish_sim_dir[-10] + "flow_geometry.npy")
          where_boundary = restore_geometry[:,:,0].astype(np.bool)
          

        # save geometry
        save_geometry = np.concatenate([np.array(np.expand_dims(where_boundary, axis=-1), dtype=np.float32),
             np.array(velocity).reshape(1,1,2) * np.array(np.expand_dims(where_velocity, axis=-1), dtype=np.float32),
                             density *  np.array(np.expand_dims(where_density, axis=-1), dtype=np.float32)], axis=-1)
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

      @classmethod
      def update_defaults(cls, defaults):
        defaults.update({
          'max_iters': max_iters,
          'output_format': 'npy',
          'periodic_y': True,
          'periodic_x': True,
          'checkpoint_file': sailfish_sim_dir,
          'checkpoint_every': lb_to_ln,
          #'cuda-sched-yield': True,
          #'cuda-minimize-cpu-usage': True,
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

  def compute_compressed_state(self, sess, encoder, state):
     
    pass

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


