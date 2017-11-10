

class subgrid:

  def __init__(self, grid_shape, sub_grid_shape):
    self.grid_shape = grid_shape
    self.sub_grid_shape = sub_grid_shape

  def _find_subgrids(self):
    self.blocks = []
    for i in xrange(int(self.grid_shape[0]/self.sub_grid_shape[0]) + 1):
      for j in xrange(int(self.grid_shape[1]/self.sub_grid_shape[1]) + 1):
        pos = (self.sub_grid_shape[0] * i, self.sub_grid_shape[1] * j)
        length = 
        self.blocks.append((pos, radius
        
    


