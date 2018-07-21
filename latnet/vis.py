
import lattice

import matplotlib.pyplot as plt
import cv2
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
import numpy as np
import os
import psutil as ps
from utils.python_utils import *
from utils.numpy_utils import *

# this class will handle visualizaitions with pygame
# for right now it will just use cv2 to make a video
class Visualizations:

  def __init__(self, config, sim_shape):
    self.sim_shape = sim_shape
    self.DxQy = lattice.TYPES[config.DxQy]
    self.simulation_video = cv2.VideoWriter()
    self.compare_video = cv2.VideoWriter()
    with open(os.devnull, 'w') as devnull:
      p = ps.subprocess.Popen(('mkdir -p figs').split(' '), 
                               stdout=devnull, stderr=devnull)
      p.communicate()
    self.simulation_video.open('figs/simulation_video.mov', fourcc, 5, (self.sim_shape[1], self.sim_shape[0]*2), True)
    self.compare_video.open('figs/compare_video.mov', fourcc, 5, (self.sim_shape[1], self.sim_shape[0]*4), True)

  def update_vel_rho(self, iteration, vel, rho):
    frame = self.vel_rho_to_frame(vel, rho)
    #frame = self.feild_to_colormap(frame)
    self.simulation_video.write(frame)

  def update_compare_vel_rho(self, iteration, true_vel, true_rho, generated_vel, generated_rho):
    #true_frame = self.vel_rho_to_frame(true_vel, true_rho)
    #generated_frame = self.vel_rho_to_frame(generated_vel, generated_rho)
    #frame = np.concatenate([generated_frame, true_frame], axis=0)
    #frame = np.concatenate([true_frame, true_frame], axis=0)
    #frame = self.feild_to_colormap(frame)
    #self.compare_video.write(frame)

    # plot energy spectrum comparison
    true_energy = energy_spectrum(true_vel[0]) 
    generated_energy = energy_spectrum(generated_vel[0]) 
    true_energy = true_energy/true_energy[1]
    generated_energy = generated_energy/generated_energy[1]
    x = np.arange(1, true_energy.shape[0])
    

    plt.loglog(x, true_energy[1:], label='true energy spectrum')
    plt.loglog(x, generated_energy[1:], label='generated energy spectrum')
    plt.loglog(x, np.power(x, -(5/3.)), label='(-5/3) Power Rule')
    plt.title("Comparison of True and Generated Energy Spectrum")
    plt.ylabel("Energy")
    plt.legend(loc=0)
    plt.savefig('figs/energy_spectrum_iter_' + str(iteration).zfill(4) + '.png')
    plt.close()

  def feild_to_colormap(self, feild):
    feild = feild - np.min(feild)
    feild = np.uint8(255 * feild/np.max(feild))
    feild = cv2.applyColorMap(feild, 2)
    #feild = cv2.resize(rotate_image_cut, (rotate_slice_size*resize_factor, rotate_slice_size*resize_factor))
    #feild = cv2.
    return feild

  def vel_rho_to_frame(self, vel, rho):
    if self.DxQy.dims == 2:
      vel = np.sqrt(np.square(vel[0,:,:,0]) + np.square(vel[0,:,:,1]))
      rho = rho[0,:,:,0]
    elif self.DxQy.dims == 3:
      vel = np.sqrt(np.square(vel[0,...,vel.shape[-2]/2,0]) + np.square(vel[0,...,vel.shape[-2]/2,1]))
      rho = rho[0,...,rho.shape[-2]/2,0]
    vel = self.feild_to_colormap(vel)
    rho = self.feild_to_colormap(rho)
    return np.concatenate([vel, rho], axis=0)



