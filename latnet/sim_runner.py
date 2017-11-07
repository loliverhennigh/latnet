
import glob
import psutil as ps
import shutil

class SimRunner:
  # generates simulation data for training 

  def __init__(self, save_dir, num_cpoints, lb_to_ln, max_iters):
    self.save_dir = save_dir
    self.num_cpoints = num_cpoints
    self.lb_to_ln = lb_to_ln
    self.max_iters = max_iters
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
         + ' --sailfish_sim_dir=' + save_dir + '/flow'

    # determine last checkpoint
    last_cpoint = self.last_cpoint()
    last_step = None
    if last_cpoint is not None:
      last_step = int(last_cpoint[5:-13])

    # add iteration till next cpoint and possible restor cpoint
    if last_step is not None:
      num_run_iters = last_step + iters_till_next_cpoint + (self.num_cpoints * self.lb_to_ln)
      if num_run_iters < max_iters:
        cmd += ' --checkpoint_from=' + str(num_run_iters)
        cmd += ' --restore_from=' + last_cpoint
      else:
        self.clean_save_dir()

    # run cmd
    with open(os.devnull, 'w') as devnull:
      p = ps.subprocess.Popen(cmd.split(' '), stdout=devnull, stderr=devnull)

    # rm previous cpoints
    if last_step is not None:
      self.clean_prev_cpoints()

  def _clean_prev_cpoints(self):
    cpoints = glob(self.save_dir + "/*.0.cpoint.npz")
    cpoints.sort()
    if len(cpoints) != 2*self.num_cpoints:
    for i in xrange(self.
      save_dirs[0]
     
  def clean_save_dir(self):
    shutil.rmtree(self.save_dir)

  def make_sim_dir(self):
    with open(os.devnull, 'w') as devnull:
      p = ps.subprocess.Popen(('mkdir -p ' + save_dir).split(' '), 
                               stdout=devnull, stderr=devnull)
      p.communicate()



