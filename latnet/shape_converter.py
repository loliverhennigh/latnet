
import tensorflow as tf
import numpy as np
from nn import *


class ShapeConverter:
  # This class allows the shape and pos to be converted from one position 
  # in the computational graph to another. For example, suppose you have 
  # a desired piece of the compressed state to extract and you need to know
  # what portion of the compressed state to feed through the decoder.

  def __init__(self):

    # function that takes in input subdomain and returns output subdomain
    self.in_out_subdomain = lambda x: x
    # function that takes in output shape and returns input shape
    self.out_in_subdomain = lambda x: x

  def add_conv(self, kernel_size, stride):

    def in_out_helper(subdomain):
      if stride == 1:
        # remove edges from padding
        edge_padding = kernel_size/2
        subdomain.remove_edges(edge_padding)

      elif stride == 2:
        # remove one from right edges if odd
        subdomain.remove_make_size_even()

        # remove edges
        edge_padding = (kernel_size-1)/2
        subdomain.remove_edges(edge_padding)

        # downsample
        subdomain.downsample()

      return subdomain

    def out_in_helper(subdomain):
      if stride == 1:
        # add edges from padding
        edge_padding = kernel_size/2
        subdomain.add_edges(edge_padding)

      elif stride == 2:
        # downsample
        subdomain.upsample()

        # add edges
        edge_padding = (kernel_size-1)/2
        subdomain.add_edges(edge_padding)

      return subdomain

    # add functions to subdomain converters
    self.concat_function_in_out(in_out_helper)
    self.concat_function_out_in(out_in_helper)

  def add_trans_conv(self, kernel_size, stride):

    # only suports filter size 2 and stride 2 for now
    #assert (kernel_size == 2) and (stride == 2), "filter size and stride need to be 2 for trans conv"

    def in_out_helper(subdomain):
      if stride == 2:
        # downsample
        subdomain.upsample()

        # add edges
        edge_padding = (kernel_size-1)/2
        subdomain.remove_edges(edge_padding)

      return subdomain

    def out_in_helper(subdomain):
      if stride == 2:
        # remove one from right edges if odd
        subdomain.add_make_size_even()

        # remove edges
        edge_padding = (kernel_size-1)/2
        subdomain.remove_edges(edge_padding)

        # downsample
        subdomain.downsample()

      return subdomain

    # add functions to subdomain converters
    self.concat_function_in_out(in_out_helper)
    self.concat_function_out_in(out_in_helper)

  def add_res_block(self, stride):
    if stride == 1:
      self.add_conv(3, stride)
    elif stride == 2:
      self.add_conv(4, stride)
    self.add_conv(3, 1)

  def concat_function_in_out(self, func):
    old_in_out_subdomain = self.in_out_subdomain
    self.in_out_subdomain = lambda x: func(old_in_out_subdomain(x))

  def concat_function_out_in(self, func):
    old_out_in_subdomain = self.out_in_subdomain
    self.out_in_subdomain = lambda x: old_out_in_subdomain(func(x))

  #def concat_shape_converter(self, shape_converter):


class SubDomain:
  # This class 
  def __init__(self, pos, size):
    # pos: the cordinates of the subdomain in the full domain (e.g. [128,128])
    # size: size of subdomain
    self.pos = pos
    self.size = size

  def add_edges(self, edge_length):
    self.pos =  [x-edge_length for x in self.pos ]
    self.size = [x+2*edge_length for x in self.size]

  def remove_edges(self, edge_length):
    self.pos =  [x+edge_length for x in self.pos ]
    self.size = [x-2*edge_length for x in self.size]

  def add_make_size_even(self):
    odd_edge = [(x % 2) for x in self.size]
    self.size = [x + y for x, y in zip(self.size, odd_edge)]

  def remove_make_size_even(self):
    odd_edge = [(x % 2) for x in self.size]
    self.size = [x - y for x, y in zip(self.size, odd_edge)]

  def downsample(self):
    self.pos  = [x/2 for x in self.pos ]
    self.size = [x/2 for x in self.size]

  def upsample(self):
    self.pos  = [x*2 for x in self.pos ]
    self.size = [x*2 for x in self.size]

"""
# quick test
s = ShapeConverter()
s.add_conv(2,2)
s.add_conv(3,1)
s.add_conv(2,2)
s.add_conv(3,1)
s.add_conv(2,2)
s.add_conv(3,1)
s.add_trans_conv(2,2)
s.add_trans_conv(2,2)
s.add_trans_conv(2,2)
in_out_subdomain = SubDomain([128,128], [256,256])
out_in_subdomain = SubDomain([0,0], [128,128])
s.in_out_subdomain(in_out_subdomain)
s.out_in_subdomain(out_in_subdomain)
print(out_in_subdomain.pos)
print(out_in_subdomain.size)
#print(s.in_out_subdomain(subdomain))
"""







