
import sys

# import sailfish
sys.path.append('../sailfish')
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTEquilibriumVelocity, NTEquilibriumDensity, DynamicValue, NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_base import ForceObject
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S

def Domain():

  def __init__(self, config):

    self.nx = config.nx
    self


  def initial_conditions(self, sim, hx, hy):
    pass

  def boundary_conditions(self, hx, hy):
    pass

  def create_sailfish_simulation(self):

    boundary = self.boundary
    boundary_where_velocity = self.boundary_where_velocity
    boundary_velocity = self.boundary_velocity
    boundary_where_pressure = self.boundary_where_pressure 
    boundary_pressure = self.boundary_pressure 

    class SaifishSubdomain(Subdomain2D):
      
      bc = NTFullBBWall

      def boundary_conditions(self, hx, hy):
        self.set_node(boundary, self.bc)
        self.set_node(boundary_where_velocity, NTEquilibriumVelocity(boundary_velocity))
        self.set_node(boundary_where_pressure, NTEquilibriumDensity(boundary_pressure))

      def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[:] = boundary_velocity[0]
        sim.vy[:] = boundary_velocity[1]
   
    class SailfishSimulation(LBFluidSim): 
      subdomain = SailfishSubdomain

      @classmethod
      def update_defaults(cls, defaults):
        defaults.update({
          'max_iters': 50000,
          'output_format': 'npy',
          'periodic_y': True,
          'periodic_x': True,
          'checkpoint_file': './lkdsfj',
          'checkpoint_every': 120,
          })

      @classmethod
      def modify_config(cls, config):
        config.visc   = 0.1

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


