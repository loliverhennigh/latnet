
import tensorflow as tf
import numpy as np
from nn import int_shape, simple_conv_2d, simple_conv_3d, simple_trans_conv_2d, simple_trans_conv_3d

# helper function
def is_numpy(x):
  return type(x) == np.ndarray

class DxQy:

  @classmethod
  def add_lattice(cls, lattice):
    w = cls._expand(lattice, cls.weights)
    w = cls._convert(lattice, w)
    lattice = lattice + w
    return lattice

  @classmethod
  def subtract_lattice(cls, lattice):
    w = cls._expand(lattice, cls.weights)
    w = cls._convert(lattice, w)
    lattice = lattice - w
    return lattice

  @classmethod
  def lattice_to_vel(cls, lattice):
    c = cls._expand(lattice, cls.c)
    c = cls._convert(lattice, c)
    lattice = cls._expand_lattice(lattice)
    vel = cls._reduce_lattice(c * lattice)
    return vel

  @classmethod
  def lattice_to_norm(cls, lattice):
    vel = cls.lattice_to_vel(lattice)
    norm = cls.vel_to_norm(vel)
    return norm

  @classmethod
  def lattice_to_rho(cls, lattice):
    if is_numpy(lattice):
      rho = np.sum(lattice, axis=-1)
      rho = np.expand_dims(rho, axis=-1)
    else:
      rho = tf.reduce_sum(lattice, axis=len(lattice.get_shape())-1)
      rho = tf.expand_dims(rho, axis=-1)
    return rho

  def lattice_to_divergence(cls, lattice):
    assert not is_numpy(lattice), "divergence not supported for numpy"
    velocity = cls.lattice_to_vel(lattice)
    if cls.dims == 2:
      divergence = nn.simple_conv_2d(vel, cls.divergence_kernel)
      divergence = divergence[:,1:-1,1:-1,:]
    elif cls.dims == 3:
      divergence = simple_conv_3d(vel, cls.divergence_kernel)
      divergence = divergence[:,1:-1,1:-1,1:-1,:]
    return divergence

  @classmethod
  def vel_to_norm(cls, vel):
    if is_numpy(vel):
      norm = np.linalg.norm(vel, axis=-1)
      norm = np.expand_dims(norm, axis=-1)
    else:
      norm = tf.norm(vel, axis=-1)
      norm = tf.expand_dims(norm, axis=-1)
    return norm

  @classmethod
  def _expand(cls, lattice, vec):
    if is_numpy(lattice):
      vec = vec.reshape((len(lattice.shape)-1)*[1] + list(vec.shape))
    else:
      vec = vec.reshape((len(lattice.get_shape())-1)*[1] + list(vec.shape))
    return vec
 
  @classmethod
  def _expand_lattice(cls, lattice):
    if is_numpy(lattice):
      lattice = np.expand_dims(lattice, axis=len(lattice.shape))
    else:
      lattice = tf.expand_dims(lattice, axis=len(lattice.get_shape()))
    return lattice
 
  @classmethod
  def _reduce_lattice(cls, lattice):
    if is_numpy(lattice):
      lattice = np.sum(lattice, axis=len(lattice.shape)-2)
    else:
      lattice = tf.reduce_sum(lattice, axis=len(lattice.get_shape())-2)
    return lattice

  @classmethod
  def _convert(cls, lattice, vec):
    if not is_numpy(lattice):
      vec = tf.constant(vec, dtype=1)
    return vec

  @classmethod
  def vel_to_freq(cls, vel):
    pass

class D2Q9(DxQy):
  dims = 2
  Q = 9
  weights = np.array([4./9.,  1./9.,  1./9., 
                      1./9.,  1./9.,  1./36.,
                      1./36., 1./36., 1./36.])
  
  c = np.array([[0 ,0], [ 0, 1], [ 1,0],
                [0,-1], [-1, 0], [ 1,1],
                [1,-1], [-1,-1], [-1,1]])

  divergence_kernel = np.zeros((3,3,2,1))
  divergence_kernel[2,1,0,0] =  1.0
  divergence_kernel[0,1,0,0] = -1.0
  divergence_kernel[1,2,1,0] =  1.0
  divergence_kernel[1,0,1,0] = -1.0

  force_kernel = np.zeros((3,3,9,1))
  force_kernel[1,0,1,0] = 1.0 # right
  force_kernel[0,1,2,0] = 1.0 # up
  force_kernel[1,2,3,0] = 1.0 # left
  force_kernel[2,1,4,0] = 1.0 # down
  force_kernel[0,0,5,0] = 1.0 # up right
  force_kernel[0,2,6,0] = 1.0 # up left
  force_kernel[2,2,7,0] = 1.0 # down left
  force_kernel[2,0,8,0] = 1.0 # down right

  def vel_to_feq(self, vel):
    vel = np.array(vel)
    vel_dot_vel = np.sum(vel * vel)
    vel_dot_c = np.sum(np.expand_dims(vel, axis=0) * self.c, axis=-1)
    feq = self.weights * (1.0 + 
                          3.0*vel_dot_c + 
                          4.5*vel_dot_c*vel_dot_c - 
                          1.5*vel_dot_vel)
    feq = feq - self.weights
    return feq 

  def rotate_lattice(self, lattice, rotation):
    for i in xrange(rotation):
      lattice = self.rotate_lattice_left(lattice)
    return lattice

  def rotate_lattice_left(self, lattice):
    lattice_split = np.split(lattice, 9, -1)
    (lattice_split[1], lattice_split[2], 
     lattice_split[3], lattice_split[4], 
     lattice_split[5], lattice_split[6], 
     lattice_split[7], lattice_split[8]) = (
     lattice_split[2], lattice_split[3], 
     lattice_split[4], lattice_split[1], 
     lattice_split[6], lattice_split[7], 
     lattice_split[8], lattice_split[5])
    lattice = np.concatenate(lattice_split, axis = -1)
    lattice = np.rot90(lattice, k=1, axes=(0,1))
    return lattice

  def flip_lattice(self, lattice):
    lattice_split = np.split(lattice, 9, -1)
    (lattice_split[1], lattice_split[2], 
     lattice_split[3], lattice_split[4], 
     lattice_split[5], lattice_split[6], 
     lattice_split[7], lattice_split[8]) = (
     lattice_split[1], lattice_split[4], 
     lattice_split[3], lattice_split[2], 
     lattice_split[8], lattice_split[7], 
     lattice_split[6], lattice_split[5])
    lattice = np.concatenate(lattice_split, axis = -1)
    lattice = np.flipud(lattice)
    return lattice

class D3Q15(DxQy):
  dims = 3
  Q = 15
  weights = np.array([2./9.,  1./9.,  1./9.,
                      1./9.,  1./9.,  1./9., 
                      1./9.,  1./72., 1./72.,
                      1./72., 1./72., 1./72.,
                      1./72., 1./72., 1./72.])
  c = np.array([[ 0, 0, 0], [ 1, 0, 0], [-1, 0, 0],
                [ 0, 1, 0], [ 0,-1, 0], [ 0, 0, 1],
                [ 0, 0,-1], [ 1, 1, 1], [-1,-1,-1],
                [ 1, 1,-1], [-1,-1, 1], [ 1,-1, 1],
                [-1, 1,-1], [ 1,-1,-1], [-1, 1, 1]])

  divergence_kernel = np.zeros((3,3,3,3,1))
  divergence_kernel[2,1,1,2,0] =  1.0
  divergence_kernel[0,1,1,2,0] = -1.0
  divergence_kernel[1,2,1,1,0] =  1.0
  divergence_kernel[1,0,1,1,0] = -1.0
  divergence_kernel[1,1,2,0,0] =  1.0
  divergence_kernel[1,1,0,0,0] = -1.0

  force_kernel = np.zeros((3,3,3,15,1))
  force_kernel[1,1,0,1 ,0] = 1.0 # down
  force_kernel[1,1,2,2 ,0] = 1.0 # up
  force_kernel[1,0,1,3 ,0] = 1.0 # down
  force_kernel[1,2,1,4 ,0] = 1.0 # up
  force_kernel[0,1,1,5 ,0] = 1.0 # down
  force_kernel[2,1,1,6 ,0] = 1.0 # up
  force_kernel[0,0,0,7 ,0] = 1.0 # down left out
  force_kernel[2,2,2,8 ,0] = 1.0 # up right in
  force_kernel[2,0,0,9 ,0] = 1.0 # down left in 
  force_kernel[0,2,2,10,0] = 1.0 # up right out
  force_kernel[0,2,0,11,0] = 1.0 # down right out
  force_kernel[2,0,2,12,0] = 1.0 # up left in 
  force_kernel[2,2,0,13,0] = 1.0 # down right in 
  force_kernel[0,0,2,14,0] = 1.0 # up left out

  def vel_to_feq(self, vel):
    print("vel_to_feq not implemented for D3Q15 yet")
    exit()

TYPES = {}
TYPES['D2Q9']  = D2Q9
TYPES['D3Q15'] = D3Q15

"""
def lattice_to_flux(lattice, boundary):
  Lveloc = get_lveloc(int(lattice.get_shape()[-1]))
  rho = lattice_to_rho(lattice)
  velocity = lattice_to_vel(lattice)
  flux = velocity * rho * (-boundary + 1.0)
  return flux

def lattice_to_force(lattice, boundary):
  Lveloc = get_lveloc(int(lattice.get_shape()[-1]))
  dims = len(lattice.get_shape())-1
  Lveloc_shape = list(map(int, Lveloc.get_shape()))
  Lveloc = tf.reshape(Lveloc, dims*[1] + Lveloc_shape)
  boundary_shape = list(map(int, boundary.get_shape()))
  boundary_edge_kernel = get_edge_kernel(int(lattice.get_shape()[-1]))
  if len(boundary.get_shape()) == 4:
    # no padding because no overlap in on edges
    edge = simple_trans_conv_2d(boundary,boundary_edge_kernel) 
    edge = edge[:,1:-1,1:-1,:]
    boundary = boundary[:,1:-1,1:-1,:]
    lattice = lattice[:,1:-1,1:-1,:]
  else: 
    # need padding because overlap in on edges in 3D
    top = boundary[:,-1:]
    boundary = tf.concat([top, boundary], axis=1)
    left = boundary[:,:,-1:]
    boundary = tf.concat([left, boundary], axis=2)
    edge = simple_trans_conv_3d(boundary, boundary_edge_kernel)
    top = lattice[:,-1:]
    lattice = tf.concat([top, lattice], axis=1)
    left = lattice[:,:,-1:]
    lattice = tf.concat([left, lattice], axis=2)
  edge = edge * (-boundary + 1.0)
  edge = edge * lattice
  edge_shape = list(map(int, edge.get_shape()))
  edge = tf.reshape(edge, edge_shape + [1])
  force = tf.reduce_sum(edge * Lveloc, axis=dims)
  return force, edge[...,0]
"""
