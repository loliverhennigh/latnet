
import lattice

import matplotlib.pyplot as plt
import cv2
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
import numpy as np
from utils.python_utils import *

# this class will handle visualizaitions with pygame
# for right now it will just use cv2 to make a video
class Visualizations:

  def __init__(self, config, sim_shape):
    self.sim_shape = sim_shape
    self.DxQy = lattice.TYPES[config.DxQy]
    self.simulation_video = cv2.VideoWriter()
    self.compare_video = cv2.VideoWriter()
    self.simulation_video.open('figs/simulation_video.mov', fourcc, 5, (self.sim_shape[1], self.sim_shape[0]*2), True)
    self.compare_video.open('figs/compare_video.mov', fourcc, 5, (self.sim_shape[1], self.sim_shape[0]*4), True)

  def update_vel_rho(self, iteration, vel, rho):
    frame = self.vel_rho_to_frame(vel, rho)
    frame = self.feild_to_colormap(frame)
    self.simulation_video.write(frame)

  def update_compare_vel_rho(self, iteration, true_vel, true_rho, generated_vel, generated_rho):
    true_frame = self.vel_rho_to_frame(true_vel, true_rho)
    generated_frame = self.vel_rho_to_frame(generated_vel, generated_rho)
    frame = np.concatenate([generated_frame, true_frame], axis=0)
    frame = self.feild_to_colormap(frame)
    self.compare_video.write(frame)

  def feild_to_colormap(self, feild):
    feild = feild - np.min(feild)
    feild = np.uint8(255 * feild/np.max(feild))
    feild = cv2.applyColorMap(feild, 2)
    return feild

  def vel_rho_to_frame(self, vel, rho):
    vel = np.sqrt(np.square(vel[0,:,:,0]) + np.square(vel[0,:,:,1]))
    rho = rho[0,:,:,0]
    return np.concatenate([vel, rho], axis=0)
