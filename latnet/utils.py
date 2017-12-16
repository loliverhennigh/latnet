

import numpy as np

def mobius_extract_pad(dat, pos, radius):
  shape = dat.shape
  pad_bottom_x = int(max(-(pos[0] - radius), 0))
  pad_top_x = int(max(-(shape[0] - pos[0]) + radius, 0)) + 1
  pad_bottom_y = int(max(-(pos[1] - radius), 0))
  pad_top_y = int(max(-(shape[1] - pos[1]) + radius, 0)) + 1
  dat = np.pad(dat, [[pad_bottom_x, pad_top_x], [pad_bottom_y, pad_top_y], [0,0]], 'wrap')
  new_pos_x = pos[0] + pad_bottom_x
  new_pos_y = pos[1] + pad_bottom_y
  dat_extract_pad = dat[new_pos_x-radius:new_pos_x+radius, new_pos_y-radius:new_pos_y+radius]
  return dat_extract_pad


