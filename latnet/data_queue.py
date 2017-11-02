
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
import glob
from tqdm import *
import sys
import os.path
import gc
import skfmm
import time
import psutil as ps
import shutil

from Queue import Queue
import threading

class DataQueue:
  def __init__(self, config, sailfish_sim):

    # base dir where all the xml files are
    self.base_dir = config.sailfish_sim_dir
    self.sailfish_sim = sailfish_sim

    # configs
    self.batch_size      = config.batch_size
    self.num_simulations = config.num_simulations
    self.seq_length      = config.seq_length
    shape = config.shape.split('x')
    shape = map(int, shape)
    self.shape = shape
 
    # lists to store the datasets
    self.geometries    = []
    self.lat_states    = []

    # make queue
    self.max_queue = config.max_queue
    self.queue = Queue() # to stop halting when putting on the queue
    self.queue_batches = []

    # Start threads
    for i in xrange(config.nr_threads):
      get_thread = threading.Thread(target=self.data_worker)
      get_thread.daemon = True
      get_thread.start()

  def create_dataset(self):

    print("clearing old data...")
    self.geometries = []
    self.lat_states = []
    self.queue_batches = []
    with self.queue.mutex:
      self.queue.queue.clear()
    shutil.rmtree(self.base_dir)

    print("generating simulations...")
    for i in tqdm(xrange(self.num_simulations)):
      with open(os.devnull, 'w') as devnull:
        save_dir = self.base_dir + "sim_" + str(i)
        p = ps.subprocess.Popen(('mkdir -p ' + save_dir).split(' '), stdout=devnull, stderr=devnull)
        p.communicate()
        p = ps.subprocess.Popen((self.sailfish_sim + ' --checkpoint_file=' + save_dir + "/flow").split(' '), stdout=devnull, stderr=devnull)
        p.communicate()

    print("parsing new data")
    self.parse_data()

  def data_worker(self):
    while True:
      geometry_file, lat_files = self.queue.get()

      # load geometry file
      geometry_array = np.load(geometry_file)
      geometry_array = np.expand_dims(geometry_array, axis=0)
      geometry_array = geometry_array[:,1:-1,1:-1]


      # load flow file
      lat_array = []
      for lat_file in lat_files:
        lat = np.load(lat_file)
        lat = lat.f.dist0a[:,1:-1,1:self.shape[0]+1]
        lat = np.swapaxes(lat, 0, 1)
        lat = np.swapaxes(lat, 1, 2)
        lat = np.expand_dims(lat, axis=0)
        lat_array.append(lat)
      lat_array = np.concatenate(lat_array, axis=0)
      lat_array = lat_array.astype(np.float32)
  
      # add to que
      self.queue_batches.append((geometry_array, lat_array))
      self.queue.task_done()
  
  def parse_data(self): 
    # get list of all simulation runs
    sim_dir = glob.glob(self.base_dir + "*/")

    # clear lists
    self.geometries = []
    self.lat_states = []

    print("parsing dataset")
    for d in tqdm(sim_dir):
      # get needed filenames
      geometry_file    = d + "flow_geometry.npy"
      lat_file = glob.glob(d + "*.0.cpoint.npz")
      lat_file.sort()

      # check file for geometry
      if not os.path.isfile(geometry_file):
        continue

      if len(lat_file) == 0:
        continue

      # store name
      self.geometries.append(geometry_file)
      self.lat_states.append(lat_file)

  def minibatch(self, state=None, boundary=None):

    for i in xrange(self.max_queue - len(self.queue_batches) - self.queue.qsize()):
      sim_index = np.random.randint(0, self.num_simulations)
      sim_start_index = np.random.randint(0, len(self.lat_states[sim_index])-self.seq_length)
      self.queue.put((self.geometries[sim_index], self.lat_states[sim_index][sim_start_index:sim_start_index+self.seq_length]))
   
    while len(self.queue_batches) < self.batch_size:
      print("spending time waiting for queue")
      time.sleep(1.01)

    batch_boundary = []
    batch_data = []
    for i in xrange(self.batch_size): 
      batch_boundary.append(self.queue_batches[0][0].astype(np.float32))
      batch_data.append(self.queue_batches[0][1])
      self.queue_batches.pop(0)
    batch_boundary = np.stack(batch_boundary, axis=0)
    batch_data = np.stack(batch_data, axis=0)
    return {boundary:batch_boundary, state:batch_data}

class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        self._target = target
        self._args = args
        threading.Thread.__init__(self)
 
    def run(self):
        self._target(*self._args)
 

"""
#dataset = Sailfish_data("../../data/", size=32, dim=3)
dataset = Sailfish_data("/data/sailfish_flows/", size=512, dim=2)
#dataset.create_dataset()
dataset.parse_data()
batch_boundary, batch_data = dataset.minibatch(batch_size=8)
for i in xrange(100):
  batch_boundary, batch_data = dataset.minibatch(batch_size=8)
  time.sleep(.8)
  print("did batch")
  plt.imshow(batch_data[0,0,:,:,0])
  plt.show()
  plt.imshow(batch_data[0,0,:,:,1])
  plt.show()
  plt.imshow(batch_data[0,0,:,:,-1])
  plt.show()
  plt.imshow(batch_data[0,0,:,:,2])
  plt.show()
  plt.imshow(batch_data[0,0,:,:,-2])
  plt.show()
  plt.imshow(np.sum(batch_data[0,0], axis=2))
  print(np.sum(batch_data[0,0]))
  plt.show()
  plt.imshow(batch_boundary[0,0,:,:,0])
  plt.show()
"""


