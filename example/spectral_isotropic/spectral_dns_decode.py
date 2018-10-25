#!/usr/bin/env python

import sys
import os
import time

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from turb_funcs import diagnostics_np


# import latnet
sys.path.append('../../latnet')
from shape_converter import SubDomain
from latnetwork import DecodeLatNet
from domain import SpectralDNSDomain
from archs.standard_jhtdb_arch import StandardJHTDBArch
from controller import LatNetController

import numpy as np
import cv2
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 

class FakeDomain(SpectralDNSDomain):
  name = "SpectralDNS"
  sim_shape = [128, 128, 128]
  periodic_x = True
  periodic_y = True
  periodic_z = True

  def __init__(self, config, save_dir):
    super(FakeDomain, self).__init__(config, save_dir)

def field_to_colormap(field):
  field = field - np.min(field)
  field = np.uint8(255 * field/np.max(field))
  field = cv2.applyColorMap(field, 2)
  return field

class JHTDBSimulation(DecodeLatNet, StandardJHTDBArch):
  script_name = __file__
  domain = FakeDomain
  decode_subdomains = []

  # decode full state
  decode_subdomains.append(SubDomain(pos=[  0,  0,  0], size=[128,128,128]))
  #decode_subdomains.append(SubDomain(pos=[  0,  0,  0], size=[64, 64, 64]))
  #decode_subdomains.append(SubDomain(pos=[  0,  0, 64], size=[64, 64, 64]))
  #decode_subdomains.append(SubDomain(pos=[  0, 64,  0], size=[64, 64, 64]))
  #decode_subdomains.append(SubDomain(pos=[  0, 64, 64], size=[64, 64, 64]))
  #decode_subdomains.append(SubDomain(pos=[ 64,  0,  0], size=[64, 64, 64]))
  #decode_subdomains.append(SubDomain(pos=[ 64,  0, 64], size=[64, 64, 64]))
  #decode_subdomains.append(SubDomain(pos=[ 64, 64,  0], size=[64, 64, 64]))
  #decode_subdomains.append(SubDomain(pos=[ 64, 64, 64], size=[64, 64, 64]))

  # decode slice to save to movie
  #image_subdomain = SubDomain(pos=[32,32,32], size=[64,64,1])
  image_subdomain = SubDomain(pos=[0,0,0], size=[128,128,1])
  decode_subdomains.append(image_subdomain)

  # make video saver
  video = cv2.VideoWriter()
  video.open('./figs/video.mov', fourcc, 5, (image_subdomain.size[0]*2, image_subdomain.size[1]*2), True)

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'run_mode': 'decode',
        'lb_to_ln': 30,
        #'lb_to_ln': 32,
        'dataset': 'spectraldns',
        'eval_cshape': '32x32x32',
        'sim_save_every': 1,
        'num_iters': 32,
        'compare': True,
        'DxQy': 'D3Q4'})

  def sub_state_computation(self, sub_state, subdomain, iteration, compare_state=None):
    save_iters = [0, 15, 30]
    print(compare_state.shape)
    if sub_state.shape[-2] == 1:
      network_vel = field_to_colormap(self.DxQy.lattice_to_norm(sub_state)[0,:,:,0,0])
      network_rho = field_to_colormap(self.DxQy.lattice_to_rho(sub_state)[0,:,:,0,0])
      compare_vel = field_to_colormap(self.DxQy.lattice_to_norm(compare_state)[0,:,:,0,0])
      compare_rho = field_to_colormap(self.DxQy.lattice_to_rho(compare_state)[0,:,:,0,0])
      network_frame = np.concatenate([network_vel, network_rho], axis=0)
      compare_frame = np.concatenate([compare_vel, compare_rho], axis=0)
      frame = np.concatenate([network_frame, compare_frame], axis=1)
      self.video.write(frame)
    elif (sub_state.shape[-2] != 1) and (iteration in save_iters):
      net_vel = self.DxQy.lattice_to_vel(sub_state)[0]
      true_vel = self.DxQy.lattice_to_vel(compare_state)[0]
      diagnostics_np(net_vel, true_vel, iteration=iteration, pos=subdomain.pos, dx=[0.1, 0.1, 0.1], diagnostics=['structure_functions'])



if __name__ == '__main__':
  sim = LatNetController(simulation=JHTDBSimulation)
  sim.run()

