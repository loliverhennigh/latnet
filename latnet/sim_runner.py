
import glob
import psutil as ps
import shutil

class SimRunner:
  # generates simulation data for training 

  def __init__(self, config, save_dir):
    self.save_dir = save_dir
    self.num_cpoints = config.seq_length * 2
    self.lb_to_ln = config.lb_to_ln
    self.max_iters = config.max_sim_iters
    self.max_iters_till_next_cpoint = 1000 # hard set for now

  def last_cpoint(self):
    cpoints = glob(self.save_dir + "/*.0.cpoint.npz")
    cpoints.sort()
    if len(cpoints):
      return cpoints[-1]
    else:
      return None

  def first_cpoint(self):
    cpoints = glob(self.save_dir + "/*.0.cpoint.npz")
    cpoints.sort()
    if len(cpoints):
      return cpoints[-1]
    else:
      return None

  def generate_cpoint(self):

    # iters till next cpoint save
    iters_till_next_cpoint = np.random.randint(0, self.max_iters_till_next_cpoint)

    # base cmd
    cmd = ('./' + self.train_sim.script_name 
         + ' --mode=generate_data'
         + ' --sailfish_sim_dir=' + save_dir + '/store/flow'

    # determine last checkpoint
    last_cpoint = self.last_cpoint()
    last_step = None
    if last_cpoint is not None:
      last_step = int(last_cpoint[5:-13])

    # add iteration till next cpoint and possible restor cpoint
    if last_step is not None:
      num_run_iters = last_step + iters_till_next_cpoint + (self.num_cpoints * self.lb_to_ln)
      if num_run_iters < max_iters:
        cmd += ' --checkpoint_from=' + save_dir + "/flow" + str(num_run_iters)
        cmd += ' --restore_from=' + last_cpoint
      else:
        self.clean_save_dir()

    # run cmd
    with open(os.devnull, 'w') as devnull:
      p = ps.subprocess.Popen(cmd.split(' '), 
                              stdout=devnull, stderr=devnull, 
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))

    # mv cpoints over
    if last_step is not None:
      self.clean_prev_cpoints()

    # if no cpoints in dir then need to restart simulation
    if len(glob(self.save_dir + "/*.0.cpoint.npz")):
      self.generate_cpoint()

  def read_geometry(self, pos, radius):
    geometry_file = self.save_dir + "/flow_geometry.npy"
    geometry_array = None
    if os.path.isfile(geometry_file):
      geometry_array = np.load(geometry_file)
      geometry_array = geometry_array.astype(np.float32)
      geometry_array = geometry_array[:,1:-1,1:-1]
      geometry_array = np.expand_dims(geometry_array, axis=0)
      geometry_array = mobius_extract_pad(geometry_array, pos, radius)
    return geometry_array

  def read_seq_states(self, seq_length, pos, radius, padding_decrease_seq):
    # load flow file
    state_out = []
    state_in = None
    state_files = glob(self.save_dir + "/*.0.cpoint.npz")
    state_files.sort()
    if len(state_files) >= seq_length:
      start_pos = np.random.randint(0, len(state_files) - self.seq_length)
      for i in xrange(seq_length):
        state = np.load(state_files[i])
        state = state.f.dist0a[:,1:-1,1:self.sim_shape[0]+1]
        state = state.astype(np.float32)
        state = np.swapaxes(state, 0, 1)
        state = np.swapaxes(state, 1, 2)
        state = np.expand_dims(state, axis=0)
        if i == 0:
          state_in = mobius_extract_pad(state, pos, radius)
        state = mobius_extract_pad(state, pos, radius - padding_decrease_seq[i])
        state_out.append(state)
    return state_in, state_out  

  def clean_prev_cpoints(self):
    old_cpoints = glob(self.save_dir + "/*.0.cpoint.npz")
    for c in old_cpoints:
      with open(os.devnull, 'w') as devnull:
        p = ps.subprocess.Popen(["rm", c], 
                                 stdout=devnull, stderr=devnull)
        p.communicate()
    new_cpoints = glob(self.save_dir + "/store/*.0.cpoint.npz")
    if len(new_cpoints) != self.num_cpoints:
      for c in new_cpoints:
        with open(os.devnull, 'w') as devnull:
          p = ps.subprocess.Popen(["mv", c, self.save_dir + "/"], 
                                   stdout=devnull, stderr=devnull)
          p.communicate()

  def clean_save_dir(self):
    shutil.rmtree(self.save_dir)

  def make_sim_dir(self):
    with open(os.devnull, 'w') as devnull:
      p = ps.subprocess.Popen(('mkdir -p ' + save_dir + "/store").split(' '), 
                               stdout=devnull, stderr=devnull)
      p.communicate()


def mobius_extract_pad(dat, pos, radius):
  shape = dat.shape
  pad_bottom_x = int(max(-(pos[0] - radius), 0))
  pad_top_x = int(max(-(shape[0] - pos[0] + radius), 0))
  pad_bottom_y = int(max(-(pos[1] - radius), 0))
  pad_top_y = int(max(-(shape[1] - pos[1] + radius), 0))
  dat = np.pad(dat, [[0,0], [pad_bottom_x, pad_top_x], [pad_bottom_y, pad_top_y], [0,0]], 'wrap')
  new_pos_x = pos[0] + pad_bottom_x
  new_pos_y = pos[1] + pad_bottom_y
  dat_extract_pad = dat[:,pos[0]-radius:pos[0]+radius, pos[1]-radius:pos[1]+radius]
  return dat_extract_pad



