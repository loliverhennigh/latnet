

import numpy as np

def mobius_extract_pad(dat, pos, radius, has_batch=False):
  shape = dat.shape
  if has_batch:
    shape = shape[1:]
  pad_bottom_x = int(max(-(pos[0] - radius), 0))
  pad_top_x = int(max(-(shape[0] - pos[0]) + radius, 0)) + 1
  pad_bottom_y = int(max(-(pos[1] - radius), 0))
  pad_top_y = int(max(-(shape[1] - pos[1]) + radius, 0)) + 1
  padding = [[pad_bottom_x, pad_top_x], [pad_bottom_y, pad_top_y], [0,0]]
  if has_batch:
    padding = [[0,0]] + padding
  dat = np.pad(dat, padding, 'wrap')
  new_pos_x = pos[0] + pad_bottom_x
  new_pos_y = pos[1] + pad_bottom_y
  dat_extract_pad = dat[new_pos_x-radius:new_pos_x+radius, new_pos_y-radius:new_pos_y+radius]
  return dat_extract_pad

def mobius_extract_pad_2(dat, pos, size, pad_length, has_batch=False):
  shape = dat.shape
  if has_batch:
    shape = shape[1:]
  pad_bottom_x = int(max(-(pos[0] - pad_length), 0))
  pad_top_x = int(max(-(shape[0] - pos[0] - size[0] - pad_length), 0))
  pad_bottom_y = int(max(-(pos[1] - pad_length), 0))
  pad_top_y = int(max(-(shape[1] - pos[1] - size[1] - pad_length), 0))
  padding = [[pad_bottom_x, pad_top_x], [pad_bottom_y, pad_top_y], [0,0]]
  if has_batch:
    padding = [[0,0]] + padding
  print(pos)
  print(pad_length)
  print(padding)
  print(dat.shape)
  dat = np.pad(dat, padding, 'wrap')
  print(dat.shape)
  new_pos_x = pos[0] - pad_length + pad_bottom_x
  new_pos_y = pos[1] - pad_length + pad_bottom_y
  if has_batch:
    dat_extract_pad = dat[:, new_pos_x:new_pos_x + size[0] + 2*pad_length, new_pos_y:new_pos_y + size[1] + 2*pad_length]
  else:
    print("not implemented")
    #dat_extract_pad = dat[new_pos_[0]:pos[0] + size[0] + 2*pad_length, pos[1]:pos[1] + size[1] + 2*pad_length]
  print(new_pos_x,new_pos_x + size[0] + 2*pad_length, new_pos_y,new_pos_y + size[1] + 2*pad_length)
  print(dat_extract_pad.shape)
  return dat_extract_pad


