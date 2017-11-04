
import sys

# import sailfish
sys.path.append('../sailfish')
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTEquilibriumVelocity, NTEquilibriumDensity, DynamicValue, NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_base import ForceObject
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S

class Domain:

  def __init__(self, config):

    shape = config.shape.split('x')
    shape = map(int, shape)
    self.shape = shape

    self.sailfish_sim_dir = config.sailfish_sim_dir
    self.max_iters = config.max_iters
    self.lb_to_ln = config.lb_to_ln
    self.visc = config.visc

  def initial_conditions(self, sim, hx, hy):
    pass

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
    max_iters = self.max_sim_iters
    lb_to_ln = self.lb_to_ln
    visc = self.visc

    class SaifishSubdomain(Subdomain2D):
      
      bc = NTFullBBWall

      def boundary_conditions(self, hx, hy):
        # set boundarys
        where_boundary = geometry_boundary_conditions(hx, hy, [self.gx, self.gy])
        self.set_node(boundary, self.bc)

        # set velocities
        where_velocity, velocity = velocity_boundary_conditions(hx, hy, [self.gx, self.gy])
        self.set_node(where_velocity, NTEquilibriumVelocity(velocity))

        # set densitys
        where_density, density = density_boundary_conditions(hx, hy, [self.gx, self.gy])
        self.set_node(where_density, NTEquilibriumDensity(density))

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
      def update_defaults(cls, defaults):
        defaults.update({
          'max_iters': max_iters,
          'output_format': 'npy',
          'periodic_y': True,
          'periodic_x': True,
          'checkpoint_file': sailfish_sim_dir,
          'checkpoint_every': lb_to_ln,
          })

      @classmethod
      def modify_config(cls, config):
        config.visc   = visc

      def __init__(self, *args, **kwargs):
        super(SailfishSimulation, self).__init__(*args, **kwargs)

    return SailfishSimulation 

  def get_compressed_state(self, pos, radius):
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


