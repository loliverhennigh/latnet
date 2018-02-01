
import lattice

import matplotlib.pyplot as plt
#import pygame

# this class will handle visualizaitions with pygame
# for right now it will just use matplotlib
class Visualizations:

  def __init__(self, config):
    self.DxQy = lattice.TYPES[config.DxQy]

  def update_vel_rho(self, iteration, vel, rho):
    #plt.imshow(vel[0,:,:,0])
    #plt.savefig('./figs/out_vel_' + str(iteration) + '.png')
    pass

