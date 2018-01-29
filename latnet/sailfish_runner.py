
class SailfishRunner:

  def __init__(self, config, save_dir, script_name):
    self.save_dir = save_dir
    self.lb_to_ln = config.lb_to_ln
    self.max_sim_iters = config.max_sim_iters
    self.script_name = script_name
 
    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape
    self.DxQy = lattice.TYPES[config.DxQy]

  def list_cpoints(self):
    cpoints = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    return cpoints.sort()

  def boundary_file(self):
    return self.save_dir + "/flow_geometry.npy"

  def first_cpoint(self):
    cpoints = self.list_cpoints()
    return cpoints[0], self.cpoint_to_iter(cpoints[0])

  def last_cpoint(self):
    cpoints = self.list_cpoints()
    return cpoints[-1], self.cpoint_to_iter(cpoints[-1])
 
  def is_restorable(self):
    cpoints = self.list_cpoints()
    boundary_file = self.boundary_file()
    return ((len(cpoints) > 0) and os.path.isfile(boundary_file)):

  def cpoint_to_iter(self, cpoint_name):
    sailfish_iter = int(cpoint_name.split('.')[-4])
    return self.sailfish_iter_to_latnet_iter(sailfish_iter)

  def iter_to_cpoint(self, iteration):
    sailfish_iter = self.latnet_iter_to_sailfish_iter(iteration)
    cpoint = (self.save_dir + '/flow.' 
             + str(sailfish_iter)
             + '.0.cpoint.npz')

  def sailfish_iter_to_latnet_iter(self, iteration):
    return int(iteration/self.lb_to_ln)

  def latnet_iter_to_sailfish_iter(self, iteration):
    return iteration*self.lb_to_ln

  def make_sim_dir(self):
    with open(os.devnull, 'w') as devnull:
      p = ps.subprocess.Popen(('mkdir -p ' + self.save_dir + "/store").split(' '), 
                               stdout=devnull, stderr=devnull)
      p.communicate()

  def clean_dir(self):
    store_files = glob.glob(self.save_dir + "/*")
    self.rm_files(store_files)

  def clean_store_dir(self):
    store_files = glob.glob(self.save_dir + "/store/*")
    self.rm_files(store_files)
 
  def mv_store_dir(self):
    store_files = glob.glob(self.save_dir + "/store/*")
    for f in store_files:
      p = ps.subprocess.Popen(["mv", f, self.save_dir + "/"])
      p.communicate()

  def rm_files(self, file_list):
    for f in file_list:
      with open(os.devnull, 'w') as devnull:
        p = ps.subprocess.Popen(["rm", f], 
                                 stdout=devnull, stderr=devnull)
        p.communicate()
 
  def new_sim(self, num_iters):

    self.make_sim_dir()
    self.clean_dir()
    self.clean_store_dir()

    cmd = ('./' + self.script_name 
         + ' --run_mode=generate_data'
         + ' --train_sim_dir=' + self.save_dir + '/store/flow'
         + ' --max_sim_iters=' + str(self.lb_to_ln*num_iters)
         + ' --checkpoint_from=0')
    p = ps.subprocess.Popen(cmd.split(' '), 
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
    p.communicate()
   
    self.mv_store_dir(self)
 
  def restart_sim(self, num_iters, keep_old=False):

    assert self.is_restorable(), "trying to restart sim without finding proper save"
    self.clean_store_dir()

    last_cpoint, last_iter = self.last_cpoint()

    cmd = ('./' + self.script_name 
         + ' --run_mode=generate_data'
         + ' --train_sim_dir=' + self.save_dir + '/store/flow'
         + ' --max_sim_iters=' + str(self.latnet_iter_to_sailfish_iter(num_iters
                                                                    + last_iter))
         + ' --checkpoint_from=0'
         + ' --restore_geometry=True'
         + ' --restore_from=' + last_cpoint[:-13])
    p = ps.subprocess.Popen(cmd.split(' '), 
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'))
    p.communicate()
  
    if not keep_old:
      self.clean_dir()
    self.mv_store_dir(self)

  def read_geometry(self, subdomain):
    geometry_file = self.boundary_file()
    geometry = None
    if os.path.isfile(geometry_file):
      geometry = np.load(geometry_file)
      geometry = geometry.astype(np.float32)
      geometry = geometry[1:-1,1:-1]
      geometry = numpy_utils.mobius_extract(geometry, subdomain)
    return geometry

  def read_state(self, iteration, subdomain):
    # load flow file
    state_file = self.iter_to_cpoint(iteration)
    state = np.load(state_file)
    state = state.f.dist0a[:,1:-1,1:self.sim_shape[0]+1]
    state = state.astype(np.float32)
    state = np.swapaxes(state, 0, 1)
    state = np.swapaxes(state, 1, 2)
    state = self.DxQy.subtract_lattice(state)
    state = numpy_utils.mobius_extract(state, subdomain)
    return state

class TrainSailfishRunner(SailfishRunner):

  def __init__(self, config, save_dir, script_name):
    SailfishRunner.__init__(self, config, save_dir, script_name)
    self.num_cpoints = 400
    # more configs will probably be added later

  def read_train_data(self, state_subdomain, geometry_subdomain, seq_state_subdomain):

    # if read geometry too many times generate new data
    #self.times_called += 1
    #if self.times_called > self.max_times_called:
    #  self.generate_cpoint()
    #  self.times_called = 0

    # read state
    state_files = glob.glob(self.save_dir + "/*.0.cpoint.npz")
    ind = np.random.randint(0, len(state_files) - self.seq_length)
    state = self.read_state(ind, state_subdomain)

    # read geometry
    geometry = self.read_geometry(geometry_subdomain)

    # read seq states
    seq_state = []
    for i in xrange(len(seq_state_subdomain)):
      seq_state.append(self.read_state(ind + i, seq_state_subdomain[i]))

    return state, geometry, seq_state

  def generate_train_data(self):
    self.new_sim(self, num_iters)

