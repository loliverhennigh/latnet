
import os
import numpy as np
import lxml.etree as etree
from tqdm import *
import glob
import subprocess
import random

import matplotlib.pyplot as plt

from Queue import Queue
import threading

# helper for voxelizing
def voxelize_file(filename, size):
  new_filename = filename[:-4] + "_size_" + str(size) + ".binvox"
  if os.path.isfile(new_filename):
    print("already there")
    return new_filename
  vox_cmd = "./binvox -d " + str(size) + " -cb -e " + filename
  with open(os.devnull, 'w') as devnull:
    ret = subprocess.check_call(vox_cmd.split(' '), stdout=devnull, stderr=devnull)
  rename_cmd = "mv " + filename[:-4] + ".binvox"  + " " 
  ret = subprocess.check_call((rename_cmd + new_filename).split(' '))
  return new_filename

# num new objects
all_vox_files = glob.glob('./train/**/*.obj')
random.shuffle(all_vox_files)
for i in tqdm(xrange(10000)):
  voxelize_file(all_vox_files[i], 256)



