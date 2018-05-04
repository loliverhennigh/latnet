

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
  dat = dat[..., new_pos_x:new_pos_x + subdomain.size[0], :, :]
  padding_tensor = padding_tensor[..., new_pos_x:new_pos_x + subdomain.size[0], :, :]
  
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
  dat = dat[..., new_pos_y:new_pos_y + subdomain.size[1], :]
  padding_tensor = padding_tensor[..., new_pos_y:new_pos_y + subdomain.size[1], :]
 
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
  padding_tensor = np.ones_like(dat[...,0:1])

  tic = time.time()
  # pad x
  axis = 0
  if has_batch:
    axis = 1
  dat = pad(dat, subdomain.pos[0], subdomain.size[0], axis, padding_type[0])
  padding_tensor = pad(padding_tensor, subdomain.pos[0], subdomain.size[0], axis, padding_type[0])
  #print("time for extract")
  #print(time.time() - tic)

  # pad y
  axis = 1
  if has_batch:
    axis = 2
  dat = pad(dat, subdomain.pos[1], subdomain.size[1], axis, padding_type[1])
  padding_tensor = pad(padding_tensor, subdomain.pos[1], subdomain.size[1], axis, padding_type[1])
 
  # pad z
  if len(subdomain.pos) == 3:
    axis = 2
    if has_batch:
      axis = 3
    dat = pad(dat, subdomain.pos[2], subdomain.size[2], axis, padding_type[2])
    padding_tensor = pad(padding_tensor, subdomain.pos[2], subdomain.size[2], axis, padding_type[2])

  padding_tensor = (1.0 - padding_tensor)

  if return_padding:
    return dat, padding_tensor
  else:
    return dat

def pad(dat, pos, size, axis, pad_type):
  shape = dat.shape
  pad_bottom = abs(min(pos, 0))
  pad_top = max((pos + size) - shape[axis], 0)
  if (pad_bottom > dat.shape[axis]) or (pad_top > dat.shape[axis]):
    print("padding not working when pad is bigger then dat")
    exit()
  if pad_bottom > 0:
    index = []
    for i in xrange(len(dat.shape)):
      if i == axis:
        index.append(slice(dat.shape[i]-pad_bottom, dat.shape[i]))
      else:
        index.append(slice(0, dat.shape[i]))
    dat_bottom = dat[index] 
  if pad_top > 0:
    index = []
    for i in xrange(len(dat.shape)):
      if i == axis:
        index.append(slice(0, pad_top))
      else:
        index.append(slice(0, dat.shape[i]))
    dat_top = dat[index] 
  index = []
  for i in xrange(len(dat.shape)):
    if i == axis:
      index.append(slice(max(pos, 0), pos+size))
    else:
      index.append(slice(0, dat.shape[i]))
  dat_middle = dat[index]
  if pad_bottom > 0:
    if pad_type == "periodic":
      dat_middle = np.concatenate([dat_bottom, dat_middle], axis=axis)
    elif pad_type == "zero":
      dat_middle = np.concatenate([np.zeros_like(dat_bottom), dat_middle], axis=axis)
  if pad_top > 0:
    if pad_type == "periodic":
      dat_middle = np.concatenate([dat_middle, dat_top], axis=axis)
    elif pad_type == "zero":
      dat_middle = np.concatenate([dat_middle, np.zeros_like(dat_top)], axis=axis)

  return dat_middle 

def stack_grid(dat, shape, has_batch=False):
  if has_batch:
    axis=2
  else:
    axis=1
  # converts a list of numpy arrays to a single array acording to shape
  dat = [dat[i:i+shape[1]] for i in range(0, len(dat), shape[1])]
  for i in xrange(shape[0]):
    dat[i] = np.concatenate(dat[i], axis=axis) 
  dat = np.concatenate(dat, axis=axis-1)
  return dat

def stack_grid(dat, shape, has_batch=False):
  if len(shape) == 2:
    if has_batch:
      axis=2
    else:
      axis=1
    # converts a list of numpy arrays to a single array acording to shape
    dat = [dat[i:i+shape[1]] for i in range(0, len(dat), shape[1])]
    for i in xrange(shape[0]):
      dat[i] = np.concatenate(dat[i], axis=axis) 
    dat = np.concatenate(dat, axis=axis-1)
  elif len(shape) == 3:
    if has_batch:
      axis=3
    else:
      axis=2
    dat = [dat[i:i+shape[2]] for i in range(0, len(dat), shape[2])]
    for i in xrange(shape[0] * shape[1]):
      dat[i] = np.concatenate(dat[i], axis=axis) 
    dat = [dat[i:i+shape[1]] for i in range(0, len(dat), shape[1])]
    for i in xrange(shape[0]):
      dat[i] = np.concatenate(dat[i], axis=axis-1) 
    dat = np.concatenate(dat, axis=axis-2)
  return dat


"""
# short test
dat = []
for i in xrange(3):
  for j in xrange(2):
    dat.append(np.zeros((1,10,10,1)) + 2*i + j)
dat = stack_grid(dat, [3,2], True)
plt.imshow(dat[0,:,:,0])
plt.show()
dat = mobius_extract(dat, [5,5], [10,10], True)
plt.imshow(dat[0,:,:,0])
plt.show()
""" 



