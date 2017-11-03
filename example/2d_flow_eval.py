#!/usr/bin/env python

import sys

# import latnet librarys
sys.path.append('../latnet')
from controller import LatNetController

# import important librarys
import numpy as np
import glob
import matplotlib.pyplot as plt

class sim:
  def boundary_conditions(h):

    hx = h.x
    hy = h.x

    # make wall
    wall = ((hx - 128)**2 + (hy - 128)**2 < 20**2)
    plt.imshow(wall)
    plt.show()
  
    # make velocity boundary
    velocity = np.concatenate(2*[np.zeros_like(np.expand_dims(wall, axis=-1))], axis=-1)
    velocity[:,0,0] = 0.1
  
    # make pressure boundary
    pressure = np.zeros_like(np.expand_dims(wall, axis=-1))
    pressure[:,-1] = 1.0
  
    boundary = np.concatenate([wall, velocity, pressure], axis=-1)
  
    return boundary
  
  def init_conditions(h):
  
    start = np.array([(4/9.0), (1/9.0), (1/9.0), (1/9.0), (1/9.0), (1/36.0), (1/36.0), (1/36.0), (1/36.0)])
    start = np.expand_dims(start)
    start = np.expand_dims(start)
  
    start = np.zeros_like(h) + start
    return start


if __name__ == '__main__':
  network_ctrl = LatNetController(latnet_sim=sim())
  network_ctrl.train()
    

