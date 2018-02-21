
import time

import numpy as np


def vel_to_lattice_state(vel):
  C = np.array([ [0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1] ])
  W = np.array(  [4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.])

  vel_dot_vel = np.sum(vel * vel)
  vel_dot_c = np.sum(np.expand_dims(vel, axis=0) * C, axis=-1)
  feq = W * (1.0 + 
             3.0*vel_dot_c + 
             4.5*vel_dot_c*vel_dot_c - 
             1.5*vel_dot_vel)
  return feq

print(vel_to_lattice_state(np.array([0.05,0.05])))

