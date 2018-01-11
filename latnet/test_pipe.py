

from copy import copy
import tensorflow as tf

import lattice as lat
import network_architectures.nn as nn

from network_architectures.shape_converter import ShapeConverter

class Pipe:
  def __init__(self):
    # in and out tensors
    self.in_tensors = {}
    self.out_tensors = {}

    # shape converter from in_tensor to out_tensor
    self.shape_converters = {}
    for name in tensors:
      self.shape_converters[name,name] = ShapeConverter()

  def train_unroll(self):
    self.in_tensors['state'] = 

    self.encoder_state(pipe, in_name="state", out_name="cstate_0")
    self.encoder_boundary(pipe, in_name="boundary", out_name="cboundary")

    return loss

  def train_step(self):
    pass

  def eval_unroll(self):
    pass

  def encoder_state(self, in_name, out_name):
    pass

  def encoder_boundary(self, in_name, out_name):
    pass

  def conv(self, in_name, out_name,
           kernel_size, stride, num_features, 
           weight_name="conv", nonlinearity=None):

    # add conv to tensor computation
    self.out_tensor[in_name] =  nn.conv_layer(self.out_tensor[in_name],
                                              kernel_size, stride, num_features, 
                                              name=weight_name, nonlinearity=None)

    # add conv to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name
        self.shape_converters(name).add_conv(kernel_size, stride)

    # rename tensor
    self.rename_out_tensor(in_name, out_name)

  def trans_conv(self, in_name, out_name,
                 kernel_size, stride, num_features, 
                 weight_name="trans_conv", nonlinearity=None):

    # add conv to tensor computation
    self.out_tensor[in_name] =  nn.transpose_conv_layer(self.out_tensor[in_name],
                                                        kernel_size, stride, num_features, 
                                                        name=weight_name, nonlinearity=None)

    # add conv to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name
        self.shape_converters[name].add_trans_conv(kernel_size, stride)

    # rename tensor
    self.rename_out_tensor(in_name, out_name)

  def res_block(self, in_name, out_name,
                filter_size=16, 
                nonlinearity=nn.concat_elu, 
                keep_p=1.0, stride=1, 
                gated=True, weight_name="resnet", 
                begin_nonlinearity=True, 
                normalize=None):

    # add res block to tensor computation
    self.out_tensor[in_name] = nn.res_block(self.out_tensor[in_name],
                                            filter_size, 
                                            nonlinearity, 
                                            keep_p, stride, 
                                            gated, weight_name, 
                                            begin_nonlinearity, 
                                            normalize)

    # add res block to the shape converter
    for name in self.shape_converters.keys():
      if name[1] == in_name
        self.shape_converters[name].add_res_block(stride)

    # rename tensor
    self.rename_out_tensor(in_name, out_name)

  def split_tensor(self, in_name,
                   a_out_name, b_out_name,
                   num_split, axis):

    # perform split on tensor
    self.out_tensors[a_out_name], self.out_tensors[b_out_name]  = tf.split(self.out_tensor[in_name],
                                                                           num_split, axis)
    # add to shape converters
    for name in self.shape_converters.keys():
      if name[1] == in_name:
        self.shape_converters[name[0], a_out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], b_out_name] = copy(self.shape_converters[name])

    # rm old tensor
    #self.rm_tensor(in_name)

  def image_combine(self, a_name, b_name, mask_name, out_name):
    # as seen in "Generating Videos with Scene Dynamics" figure 1
    self.out_tensors[out_name] = ((self.out_tensors[a_name] *      self.out_tensors[mask_name] )
                                + (self.out_tensors[b_name] * (1 - self.out_tensors[mask_name])))

    # take shape converters from a_name
    # TODO add tools to how shape converters are merged to make safer
    for name in self.shape_converters.keys():
      if name[1] == a_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == b_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == mask_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])

    # rm old tensors
    #self.rm_tensor(   a_name)
    #self.rm_tensor(   b_name)
    #self.rm_tensor(mask_name)

  def nonlinearity(self, name, nonlinarity_name):
    nonlin = nn.set_nonlinearity(nonlinarity_name)
    self.out_tensors[name] = nonlin(self.out_tensors[name])

  def combine_pipe(self, other_pipe):
    self.in_tensors.update(other_pipe.in_tensors)
    self.out_tensors.update(other_pipe.out_tensors)
    self.shape_converters.update(other_pipe.shape_converters)

  def split_pipe(self, old_name, new_name):
    self.out_tensors[new_name] = self.out_tensors[old_name]
    for name in self.shape_converters.keys():
      if name[1] == old_name
        self.shape_converters[name[0],new_name] = copy(self.shape_converters[name])

  def remove_tensor(self, rm_name):
    self.out_tensors.pop(rm_name)
    for name in self.shape_converters.keys():
      if name[1] == rm_name
        self.shape_converters.pop(name)
    
  def rename_out_tensor(self, old_name, new_name):
    self.out_tensors[new_name] = self.out_tensors.pop(old_name)
    for name in self.shape_converters.keys():
      if name[1] == old_name
        self.shape_converters[name[0],new_name] = self.shape_converters.pop(name)





