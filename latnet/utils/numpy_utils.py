

import numpy as np
import matplotlib.pyplot as plt
import time

def mobius_extract(dat, subdomain, padding_type=['periodic', 'periodic'], has_batch=False, return_padding=False):
  # extracts a chunk at pos and with size from a dat tensor 
  shape = dat.shape
  if has_batch:
    shape = shape[1:]

  # padding tensor
  padding_tensor = np.ones_like(dat[...,0:1])

  # pad x
  pad_bottom_x = abs(min(subdomain.pos[0], 0))
  pad_top_x = max((subdomain.pos[0] + subdomain.size[0]) - shape[0], 0)
  padding_x = [[pad_bottom_x, pad_top_x]] + len(subdomain.pos)*[[0,0]]
  if has_batch:
    padding_x = [[0,0]] + padding_x
  if padding_type[0] == 'periodic':
    dat = np.pad(dat, padding_x, 'wrap')
    padding_tensor = np.pad(padding_tensor, padding_x, 'wrap')
  elif padding_type[0] == 'zero':
    dat = np.pad(dat, padding_x, 'constant')
    padding_tensor = np.pad(padding_tensor, padding_x, 'constant')
  new_pos_x = subdomain.pos[0] + pad_bottom_x
  if len(subdomain.pos) == 2:
    dat = dat[..., new_pos_x:new_pos_x + subdomain.size[0], :, :]
    padding_tensor = padding_tensor[..., new_pos_x:new_pos_x + subdomain.size[0], :, :]
  elif len(subdomain.pos) == 3:
    dat = dat[..., new_pos_x:new_pos_x + subdomain.size[0], :, :, :]
    padding_tensor = padding_tensor[..., new_pos_x:new_pos_x + subdomain.size[0], :, :, :]
  
  # pad y
  pad_bottom_y = abs(min(subdomain.pos[1], 0))
  pad_top_y = max((subdomain.pos[1] + subdomain.size[1]) - shape[1], 0)
  padding_y = [[0,0], [pad_bottom_y, pad_top_y]] + (len(subdomain.pos)-1)*[[0,0]]
  if has_batch:
    padding_y = [[0,0]] + padding_y
  if padding_type[1] == 'periodic':
    dat = np.pad(dat, padding_y, 'wrap')
    padding_tensor = np.pad(padding_tensor, padding_y, 'wrap')
  elif padding_type[1] == 'zero':
    dat = np.pad(dat, padding_y, 'constant')
    padding_tensor = np.pad(padding_tensor, padding_y, 'constant')
  new_pos_y = subdomain.pos[1] + pad_bottom_y
  if len(subdomain.pos) == 2:
    dat = dat[..., new_pos_y:new_pos_y + subdomain.size[1], :]
    padding_tensor = padding_tensor[..., new_pos_y:new_pos_y + subdomain.size[1], :]
  if len(subdomain.pos) == 3:
    dat = dat[..., new_pos_y:new_pos_y + subdomain.size[1], :, :]
    padding_tensor = padding_tensor[..., new_pos_y:new_pos_y + subdomain.size[1], :, :]
 
  # pad z
  if len(subdomain.pos) == 3:
    pad_bottom_z = abs(min(subdomain.pos[2], 0))
    pad_top_z = max((subdomain.pos[2] + subdomain.size[2]) - shape[2], 0)
    padding_z = [[0,0], [0,0], [pad_bottom_z, pad_top_z], [0,0]]
    if has_batch:
      padding_z = [[0,0]] + padding_z
    if padding_type[2] == 'periodic':
      dat = np.pad(dat, padding_z, 'wrap')
      padding_tensor = np.pad(padding_tensor, padding_z, 'wrap')
    elif padding_type[2] == 'zero':
      dat = np.pad(dat, padding_z, 'constant')
      padding_tensor = np.pad(padding_tensor, padding_z, 'constant')
    new_pos_z = subdomain.pos[2] + pad_bottom_z
    dat = dat[..., new_pos_z:new_pos_z + subdomain.size[2], :]
    padding_tensor = padding_tensor[..., new_pos_z:new_pos_z + subdomain.size[2], :]

  padding_tensor = (1.0 - padding_tensor)

  if return_padding:
    return dat, padding_tensor
  else:
    return dat

def mobius_extract_2(dat, subdomain, padding_type=['periodic', 'periodic', 'periodic'], has_batch=False, return_padding=False):

  # extracts a chunk at pos and with size from a dat tensor 
  shape = dat.shape
  if has_batch:
    shape = shape[1:]

  # padding tensor
  #padding_tensor = np.zeros(list(dat.shape)[:-1] + [1]) + 1 # zeros like this takes no memory so isnt a issue
  padding_tensor = np.zeros(list(dat.shape)[:-1] + [1]) # zeros like this takes no memory so isnt a issue

  # pad x
  axis_x = 0
  if has_batch:
    axis_x = 1
  dat = pad(dat, subdomain.pos[0], subdomain.size[0], axis_x)
  padding_tensor = pad(padding_tensor, subdomain.pos[0], subdomain.size[0], axis_x)

  # pad y
  axis_y = 1
  if has_batch:
    axis_y = 2
  for i in range(len(dat)):
    dat[i] = pad(dat[i], subdomain.pos[1], subdomain.size[1], axis_y)
    padding_tensor[i] = pad(padding_tensor[i], subdomain.pos[1], subdomain.size[1], axis_y)

  # pad z
  if len(subdomain.pos) == 3:
    axis_z = 2
    if has_batch:
      axis_z = 3
    for i in range(len(dat)):
      for j in range(len(dat[i])):
        dat[i][j] = pad(dat[i][j], subdomain.pos[2], subdomain.size[2], axis_z)
        padding_tensor[i][j] = pad(padding_tensor[i][j], subdomain.pos[2], subdomain.size[2], axis_z)

  # concat all elements up
  if len(subdomain.pos) == 3:
    for i in range(len(dat)):
      for j in range(len(dat[i])):
        dat[i][j] = concatenate_bot_mid_top(dat[i][j][0], 
                                            dat[i][j][1], 
                                            dat[i][j][2], 
                                            axis=axis_z,
                                            pad_type=padding_type[2])
        padding_tensor[i][j] = concatenate_bot_mid_top(padding_tensor[i][j][0], 
                                            padding_tensor[i][j][1], 
                                            padding_tensor[i][j][2], 
                                            axis=axis_z,
                                            pad_type=padding_type[2],
                                            zero_value=1.0)
  for i in range(len(dat)):
    dat[i] = concatenate_bot_mid_top(dat[i][0], 
                                     dat[i][1], 
                                     dat[i][2], 
                                     axis=axis_y,
                                     pad_type=padding_type[1])
    padding_tensor[i] = concatenate_bot_mid_top(padding_tensor[i][0], 
                                     padding_tensor[i][1], 
                                     padding_tensor[i][2], 
                                     axis=axis_y,
                                     pad_type=padding_type[1],
                                     zero_value=1.0)
  dat = concatenate_bot_mid_top(dat[0],
                                dat[1],
                                dat[2],
                                axis=axis_x,
                                pad_type=padding_type[0])
  padding_tensor = concatenate_bot_mid_top(padding_tensor[0],
                                padding_tensor[1],
                                padding_tensor[2],
                                axis=axis_x,
                                pad_type=padding_type[0],
                                zero_value=1.0)


  if return_padding:
    return dat, padding_tensor
  else:
    return dat

def pad(dat, pos, size, axis):

  if dat is None:
    return [None, None, None]

  shape = dat.shape
  pad_bottom = abs(min(pos, 0))
  pad_top = max((pos + size) - shape[axis], 0)
  if (pad_bottom > dat.shape[axis]) or (pad_top > dat.shape[axis]):
    print("padding not working when pad is bigger then dat")
    print("dat shape: " + str(dat.shape[axis]))
    print("Pad bottom: " + str(pad_bottom))
    print("Pad top: " + str(pad_top))
    exit()
  if pad_bottom > 0:
    index = []
    for i in range(len(dat.shape)):
      if i == axis:
        index.append(slice(dat.shape[i]-pad_bottom, dat.shape[i]))
      else:
        index.append(slice(0, dat.shape[i]))
    dat_bottom = dat[index] 
  else:
    dat_bottom = None
  if pad_top > 0:
    index = []
    for i in range(len(dat.shape)):
      if i == axis:
        index.append(slice(0, pad_top))
      else:
        index.append(slice(0, dat.shape[i]))
    dat_top = dat[index] 
  else:
    dat_top = None
  index = []
  for i in range(len(dat.shape)):
    if i == axis:
      index.append(slice(max(pos, 0), pos+size))
    else:
      index.append(slice(0, dat.shape[i]))
  dat_middle = dat[index]

  return [dat_bottom, dat_middle, dat_top]

def concatenate_bot_mid_top(dat_bottom, dat_middle, dat_top, axis, pad_type='periodic', zero_value=0.0):
  if dat_middle is None:
    return None
  if dat_bottom is not None:
    if pad_type == "periodic":
      dat_middle = np.concatenate([dat_bottom, dat_middle], axis=axis)
    elif pad_type == "zero":
      dat_middle = np.concatenate([np.zeros_like(dat_bottom) + zero_value, dat_middle], axis=axis)
  if dat_top is not None:
    if pad_type == "periodic":
      dat_middle = np.concatenate([dat_middle, dat_top], axis=axis)
    elif pad_type == "zero":
      dat_middle = np.concatenate([dat_middle, np.zeros_like(dat_top) + zero_value], axis=axis)
  return dat_middle
 
def stack_grid(dat, shape, has_batch=False):
  if len(shape) == 2:
    if has_batch:
      axis=2
    else:
      axis=1
    # converts a list of numpy arrays to a single array acording to shape
    dat = [dat[i:i+shape[1]] for i in range(0, len(dat), shape[1])]
    for i in range(shape[0]):
      dat[i] = np.concatenate(dat[i], axis=axis) 
    dat = np.concatenate(dat, axis=axis-1)
  elif len(shape) == 3:
    if has_batch:
      axis=3
    else:
      axis=2
    dat = [dat[i:i+shape[2]] for i in range(0, len(dat), shape[2])]
    for i in range(shape[0] * shape[1]):
      dat[i] = np.concatenate(dat[i], axis=axis) 
    dat = [dat[i:i+shape[1]] for i in range(0, len(dat), shape[1])]
    for i in range(shape[0]):
      dat[i] = np.concatenate(dat[i], axis=axis-1) 
    dat = np.concatenate(dat, axis=axis-2)
  return dat

def energy_spectrum(vel, has_batch=False):

  # get shapes
  nx = vel.shape[:-1]
  if has_batch:
    nx = nx[1:]
  nxc = [x/2.0 + 1.0 for x in nx]

  kmax = int(round(np.sqrt((1-nxc[0])**2 + (1-nxc[1])**2 + (1-nxc[2])**2)) + 1)

  Eu = np.zeros(kmax)
  Ev = np.zeros(kmax)
  Ew = np.zeros(kmax)

  P = np.power(np.abs(np.fft.fftshift(np.fft.fftn(vel[...,0]))), 2.0)
  for i in range(1, nx[0]+1):
    for j in range(1, nx[1]+1):
      for k in range(1, nx[2]+1):
        km = int(round(np.sqrt((i - nxc[0])**2 + (j - nxc[1])**2 + (k - nxc[2])**2)))
        Eu[km] = Eu[km] + P[i-1,j-1,k-1]

  P = np.power(np.abs(np.fft.fftshift(np.fft.fftn(vel[...,1]))), 2.0)
  for i in range(1, nx[0]+1):
    for j in range(1, nx[1]+1):
      for k in range(1, nx[2]+1):
        km = int(round(np.sqrt((i - nxc[0])**2 + (j - nxc[1])**2 + (k - nxc[2])**2)))
        Ev[km] = Ev[km] + P[i-1,j-1,k-1]

  P = np.power(np.abs(np.fft.fftshift(np.fft.fftn(vel[...,2]))), 2.0)
  for i in range(1, nx[0]+1):
    for j in range(1, nx[1]+1):
      for k in range(1, nx[2]+1):
        km = int(round(np.sqrt((i - nxc[0])**2 + (j - nxc[1])**2 + (k - nxc[2])**2)))
        Ew[km] = Ew[km] + P[i-1,j-1,k-1]

  E = Eu + Ev + Ew

  return E

def np_stack_list(dat, axis=0):
  num_elements = len(dat[0])
  dat_tmp = []
  for i in range(num_elements):
    dat_tmp.append(np.stack([x[i] for x in dat], axis=axis))
  return dat_tmp

def np_stack_llist(dat, axis=0):
  num_elements_1 = len(dat[0])
  num_elements_2 = len(dat[0][0])
  dat_tmp = []
  for i in range(num_elements_1):
    dat_ttmp = []
    for j in range(num_elements_2):
      dat_ttmp.append(np.stack([x[i][j] for x in dat], axis=axis))
    dat_tmp.append(dat_ttmp)
  return dat_tmp

def flip_boundary_vel(boundary):
  boundary[...,1] = -boundary[...,1]
  return boundary

def rotate_boundary_vel(boundary, k):
  for i in range(k):
    store_boundary = boundary[...,0]
    boundary[...,0] = -boundary[...,1]
    boundary[...,1] = boundary[...,0]
  return boundary




"""
# short test
dat = []
for i in range(3):
  for j in range(2):
    dat.append(np.zeros((1,10,10,1)) + 2*i + j)
dat = stack_grid(dat, [3,2], True)
plt.imshow(dat[0,:,:,0])
plt.show()
dat = mobius_extract(dat, [5,5], [10,10], True)
plt.imshow(dat[0,:,:,0])
plt.show()
""" 



