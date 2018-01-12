

import numpy as np
import matplotlib.pyplot as plt

def mobius_extract(dat, subdomain, has_batch=False):
  # extracts a chunk at pos and with size from a dat tensor 
  shape = dat.shape
  if has_batch:
    shape = shape[1:]
  pad_bottom_x = abs(min(subdomain.pos[0], 0))
  pad_top_x = max((subdomain.pos[0] + subdomain.size[0]) - shape[0], 0)
  pad_bottom_y = abs(min(subdomain.pos[1], 0))
  pad_top_y = max((subdomain.pos[1] + subdomain.size[1]) - shape[1], 0)
  padding = [[pad_bottom_x, pad_top_x], [pad_bottom_y, pad_top_y], [0,0]]
  if has_batch:
    padding = [[0,0]] + padding
  dat = np.pad(dat, padding, 'wrap')
  new_pos_x = subdomain.pos[0] + pad_bottom_x
  new_pos_y = subdomain.pos[1] + pad_bottom_y
  dat_extract_pad = dat[..., new_pos_x:new_pos_x + subdomain.size[0], 
                             new_pos_y:new_pos_y + subdomain.size[1], :]
  return dat_extract_pad

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



