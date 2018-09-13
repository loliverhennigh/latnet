
import tensorflow as tf
import numpy as np
from copy import copy

import lattice
from shape_converter import ShapeConverter, SubDomain
import nn as nn

class NetArch(object):

  def __init__(self, config):
    super(NetArch, self).__init__()

    # in and out tensors
    self.in_tensors = {}
    self.out_tensors = {}
    self.in_pad_tensors = {}
    self.out_pad_tensors = {}
    self.shape_converters = {}

    # get configs
    self.run_mode = config.run_mode
    if config.run_mode == "train":
      self.train_mode = config.train_mode
    self.DxQy = lattice.TYPES[config.DxQy]()

    # make templates for peices of network
    self.make_network_templates()

  def make_network_templates(self):
    # piecese of network
    self.encoder_state                = tf.make_template('encoder_state', self._encoder_state)
    self.encoder_boundary             = tf.make_template('encoder_boundary', self._encoder_boundary)
    self.compression_mapping          = tf.make_template('compression_mapping', self._compression_mapping)
    self.decoder_state                = tf.make_template('decoder_state', self._decoder_state)
    if self.run_mode == "train":
      if self.train_mode == "full":
        self.full_seq_pred                = tf.make_template('unroll', self._full_seq_pred, create_scope_now_=True)
      elif self.train_mode == "compression":
        self.comp_seq_pred                = tf.make_template('unroll', self._comp_seq_pred, create_scope_now_=True)

  def _full_seq_pred(self, in_state_name, in_boundary_name, out_cstate_names, out_names, gpu_id=0):

    # name names compressesion peices in unroll
    if in_boundary_name is not None:
      cboundary_name = "comp_" + in_boundary_name
    else:
      cboundary_name = None

    ### encode ###
    self.encoder_state(in_name=in_state_name, 
                       out_name=out_cstate_names[0])
    if in_boundary_name is not None:
      self.encoder_boundary(in_name=in_boundary_name, 
                            out_name=cboundary_name)
      
    ### unroll on the compressed state ###
    for j in range(len(out_names)-1):
      self.add_shape_converter(out_cstate_names[j])
      if (j == 0) and (in_boundary_name is not None):
        self.compression_mapping(in_cstate_name=out_cstate_names[0],
                                 in_cboundary_name=cboundary_name,
                                 out_name=out_cstate_names[1], 
                                 start_apply_boundary=True)
      else:
        self.compression_mapping(in_cstate_name=out_cstate_names[j],
                                 in_cboundary_name=cboundary_name,
                                 out_name=out_cstate_names[j+1])
    self.add_shape_converter(out_cstate_names[len(out_names)-1])

    ### decode all compressed states ###
    for j in range(len(out_names)):
       self.match_trim_tensor(in_name=out_cstate_names[j], 
                              match_name=out_cstate_names[-1], 
                              out_name=out_cstate_names[j])
       self.decoder_state(in_name=out_cstate_names[j], 
                          out_name=out_names[j],
                          lattice_size=self.DxQy.Q)
       #if gpu_id == 0:
       #  self.lattice_summary(in_name=out_names[j], summary_name=out_names[j])

  def _comp_seq_pred(self, in_cstate_name, in_cboundary_name, out_names, gpu_id=0):

    ### unroll on the compressed state ###
    for j in range(len(out_names)-1):
      self.add_shape_converter(out_names[j])
      if (j == 0) and (in_cboundary_name is not None):
        self.compression_mapping(in_cstate_name=in_cstate_name,
                                 in_cboundary_name=in_cboundary_name,
                                 out_name=out_names[1], 
                                 start_apply_boundary=True)
      else:
        self.compression_mapping(in_cstate_name=out_names[j],
                                 in_cboundary_name=in_cboundary_name,
                                 out_name=out_names[j+1])
    self.add_shape_converter(out_names[-1])

    ### decode all compressed states ###
    for j in range(1, len(out_names)):
       self.match_trim_tensor(in_name=out_names[j], 
                              match_name=out_names[-1], 
                              out_name=out_names[j])


  def fc(self, in_name, out_name,
         hidden, weight_name='fc', 
         nonlinearity=None, flat=False):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.fc_layer(self.out_tensors[in_name],
                                              hidden, name=weight_name, 
                                              nonlinearity=nonlinearity, flat=flat)

  def simple_conv(self, in_name, out_name, kernel):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.simple_conv_2d(self.out_tensors[in_name], k=kernel)

    # add conv to the shape converter
    for name in list(self.shape_converters.keys()):
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_conv(3, 1)

  def conv(self, in_name, out_name,
           kernel_size, stride, filter_size, 
           weight_name="conv", nonlinearity=None,
           normalize=None):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.conv_layer(self.out_tensors[in_name],
                                              kernel_size, stride, filter_size, 
                                              name=weight_name, nonlinearity=nonlinearity,
                                              normalize=normalize)

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_conv_pad(self.out_pad_tensors[in_name], kernel_size, stride)

    # ensure zeros padding
    self.out_tensors[out_name] = nn.apply_pad(self.out_tensors[out_name], self.out_pad_tensors[out_name])

    # add conv to the shape converter
    for name in list(self.shape_converters.keys()):
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_conv(kernel_size, stride)

  def trans_conv(self, in_name, out_name,
                 kernel_size, stride, filter_size, 
                 weight_name="trans_conv", nonlinearity=None):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.transpose_conv_layer(self.out_tensors[in_name],
                                                        kernel_size, stride, filter_size, 
                                                        name=weight_name, nonlinearity=nonlinearity)

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_trans_conv_pad(self.out_pad_tensors[in_name], kernel_size, stride)

    # ensure zeros padding
    self.out_tensors[out_name] = nn.apply_pad(self.out_tensors[out_name], self.out_pad_tensors[out_name])

    # add conv to the shape converter
    for name in list(self.shape_converters.keys()):
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_trans_conv(kernel_size, stride)

  def upsample(self, in_name, out_name):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.upsampleing_resize(self.out_tensors[in_name])

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_trans_conv_pad(self.out_pad_tensors[in_name], 1, 2)

    # add conv to the shape converter
    for name in list(self.shape_converters.keys()):
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_trans_conv(0, 2)

  def downsample(self, in_name, out_name, sampling='ave'):

    # add conv to tensor computation
    self.out_tensors[out_name] =  nn.downsample(self.out_tensors[in_name], sampling=sampling)

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_conv_pad(self.out_pad_tensors[in_name], 1, 2)

    # add conv to the shape converter
    for name in list(self.shape_converters.keys()):
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_conv(1, 2)

  def res_block(self, in_name, out_name,
                a_name = None,
                filter_size=16, 
                kernel_size=3, 
                nonlinearity=nn.concat_elu, 
                keep_p=1.0, stride=1, 
                gated=True, weight_name="resnet", 
                begin_nonlinearity=True, 
                normalize=None):
                #normalize='batch_norm'):

    # add res block to tensor computation
    self.out_tensors[out_name] = nn.res_block(self.out_tensors[in_name],
                                            filter_size=filter_size, 
                                            kernel_size=kernel_size, 
                                            nonlinearity=nonlinearity, 
                                            keep_p=keep_p, stride=stride, 
                                            gated=gated, name=weight_name, 
                                            begin_nonlinearity=begin_nonlinearity, 
                                            normalize=normalize)

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_res_pad(self.out_pad_tensors[in_name], kernel_size, stride)

    # ensure zeros padding
    self.out_tensors[out_name] = nn.apply_pad(self.out_tensors[out_name], self.out_pad_tensors[out_name])

    # add res block to the shape converter
    for name in list(self.shape_converters.keys()):
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_res_block(kernel_size, stride)

  def fast_res_block(self, in_name, out_name,
                     filter_size=16, 
                     filter_size_conv=4, 
                     kernel_size=7, 
                     nonlinearity=nn.concat_elu, 
                     weight_name="resnet", 
                     begin_nonlinearity=True):

    # add res block to tensor computation
    self.out_tensors[out_name] = nn.fast_res_block(self.out_tensors[in_name],
                                                   filter_size=filter_size, 
                                                   filter_size_conv=filter_size_conv, 
                                                   kernel_size=kernel_size, 
                                                   nonlinearity=nonlinearity, 
                                                   name=weight_name, 
                                                   begin_nonlinearity=begin_nonlinearity)

    # remove edges or pool of pad tensor
    self.out_pad_tensors[out_name] = nn.mimic_res_pad(self.out_pad_tensors[in_name], kernel_size, stride)

    # ensure zeros padding
    self.out_tensors[out_name] = nn.apply_pad(self.out_tensors[out_name], self.out_pad_tensors[out_name])

    # add res block to the shape converter
    for name in list(self.shape_converters.keys()):
      if name[1] == in_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
        self.shape_converters[name[0], out_name].add_res_block(kernel_size, 1)


  def split_tensor(self, in_name,
                   out_names,
                   num_split, axis):

    # perform split on tensor
    if axis == -1:
      axis = len(self.out_tensors[in_name].get_shape())-1
    splited_tensors  = tf.split(self.out_tensors[in_name],
                                num_split, axis)
    for i in range(len(out_names)):
      self.out_tensors[out_names[i]] = splited_tensors[i]

    # add to shape converters
    for name in list(self.shape_converters.keys()):
      if name[1] == in_name:
        for i in range(len(out_names)):
          self.shape_converters[name[0], out_names[i]] = copy(self.shape_converters[name])

  def concat_tensors(self, in_names, out_name, axis=-1):
    in_tensors = [self.out_tensors[name] for name in in_names]
    self.out_tensors[out_name] = tf.concat(in_tensors, 
                                           axis=axis)

    # add to shape converters
    for name in list(self.shape_converters.keys()):
      if name[1] in in_names:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])

  def trim_tensor(self, in_name, out_name, trim):
    if trim > 0:
      if len(self.out_tensors[in_name].get_shape()) == 4:
        self.out_tensors[out_name] = self.out_tensors[in_name][:,trim:-trim, trim:-trim]
        self.out_pad_tensors[out_name] = self.out_pad_tensors[in_name][:,trim:-trim, trim:-trim]
      elif len(self.out_tensors[in_name].get_shape()) == 5:
        self.out_tensors[out_name] = self.out_tensors[in_name][:,trim:-trim, trim:-trim, trim:-trim]
        self.out_pad_tensors[out_name] = self.out_pad_tensors[in_name][:,trim:-trim, trim:-trim, trim:-trim]

  def match_trim_tensor(self, in_name, match_name, out_name, in_out=False):

    if in_out: 
      subdomain = SubDomain(self.DxQy.dims*[0], self.DxQy.dims*[128])
      new_subdomain = self.shape_converters[in_name, match_name].in_out_subdomain(subdomain)
      #self.trim_tensor(in_name, out_name, abs(new_subdomain.pos[0])+1)
      self.trim_tensor(in_name, out_name, abs(new_subdomain.pos[0]))
    else:
      subdomain = SubDomain(self.DxQy.dims*[0], self.DxQy.dims*[1])
      new_subdomain = self.shape_converters[in_name, match_name].out_in_subdomain(subdomain)
      self.trim_tensor(in_name, out_name, abs(new_subdomain.pos[0]))

  def image_combine(self, a_name, b_name, mask_name, out_name):
    # as seen in "Generating Videos with Scene Dynamics" figure 1
    self.out_tensors[out_name] = ((self.out_tensors[a_name] *      self.out_tensors[mask_name] )
                                + self.out_tensors[b_name])
                                #+ (self.out_tensors[b_name] * (1 - self.out_tensors[mask_name])))

    # update padding name
    self.out_pad_tensors[out_name] = self.out_pad_tensors[a_name]

    # take shape converters from a_name
    for name in self.shape_converters.keys():
      if name[1] == a_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == b_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])
      if name[1] == mask_name:
        self.shape_converters[name[0], out_name] = copy(self.shape_converters[name])

  def lattice_shape(self, in_name):
    shape = tf.shape(self.out_tensors[in_name])[1:-1]
    return shape

  def num_lattice_cells(self, in_name, return_float=False):
    lattice_shape = self.lattice_shape(in_name)
    lattice_cells = tf.reduce_prod(lattice_shape)
    if return_float:
      lattice_cells = tf.cast(lattice_cells, tf.float32)
    return lattice_cells

  def add_shape_converter(self, name):
    self.shape_converters[name, name] = ShapeConverter()

  def nonlinearity(self, name, nonlinearity_name):
    nonlin = nn.set_nonlinearity(nonlinearity_name)
    self.out_tensors[name] = nonlin(self.out_tensors[name])

  def l1_loss(self, true_name, pred_name, loss_name, factor=None):
    self.out_tensors[loss_name] = tf.reduce_mean(tf.abs(tf.stop_gradient(self.out_tensors[ true_name])
                                              - self.out_tensors[pred_name]))
    if factor is not None:
      self.out_tensors[loss_name] = factor * self.out_tensors[loss_name]


    self.out_tensors[loss_name] = tf.nn.l2_loss(tf.stop_gradient(self.out_tensors[ true_name]) 
                                                - self.out_tensors[pred_name])
    if factor is not None:
      self.out_tensors[loss_name] = factor * self.out_tensors[loss_name]

  def gen_loss(self, class_name, loss_name, factor=None):
    self.out_tensors[loss_name] = -tf.reduce_mean(tf.log(self.out_tensors[class_name]))
    if factor is not None:
      self.out_tensors[loss_name] = factor * self.out_tensors[loss_name]

  def disc_loss(self, true_class_name, pred_class_name, loss_name, factor=None):
    self.out_tensors[loss_name] = -tf.reduce_mean(tf.log(self.out_tensors[true_class_name])
                                          + tf.log(1.0 - self.out_tensors[pred_class_name]))
    if factor is not None:
      self.out_tensors[loss_name] = factor * self.out_tensors[loss_name]

  def combine_pipe(self, other_pipe):
    self.in_tensors.update(other_pipe.in_tensors)
    self.out_tensors.update(other_pipe.out_tensors)
    self.shape_converters.update(other_pipe.shape_converters)

  def split_pipe(self, old_name, new_name):
    self.out_tensors[new_name] = self.out_tensors[old_name]
    for name in self.shape_converters.keys():
      if name[1] == old_name:
        self.shape_converters[name[0],new_name] = copy(self.shape_converters[name])

  def remove_tensor(self, rm_name):
    self.out_tensors.pop(rm_name)
    for name in self.shape_converters.keys():
      if name[1] == rm_name:
        self.shape_converters.pop(name)

  def add_step_counter(self, name):
    counter = tf.get_variable(name, [], 
              initializer=tf.constant_initializer(0), trainable=False)
    self.in_tensors[name] = counter
    self.out_tensors[name] = counter
    self.shape_converters[name,name] = ShapeConverter()
 
  def add_tensor(self, name, shape):
    tensor     = tf.placeholder(tf.float32, shape, name=name)
    pad_tensor = tf.placeholder(tf.float32, shape[:-1] + [1], name=name + "_pad")
    self.in_tensors[name] = tensor
    self.out_tensors[name] = tensor
    self.in_pad_tensors[name] = pad_tensor
    self.out_pad_tensors[name] = pad_tensor
    self.shape_converters[name,name] = ShapeConverter()
  
  def add_lattice(self, name, gpu_id=0):
    self.add_tensor(name, (1 + self.DxQy.dims) * [None] + [self.DxQy.Q])
    if gpu_id == 0:
      self.lattice_summary(in_name=name, summary_name=name)
    
  def add_boundary(self, name, gpu_id=0):
    self.add_tensor(name, (1 + self.DxQy.dims) * [None] + [self.DxQy.boundary_dims])
    if gpu_id == 0:
      self.boundary_summary(in_name=name, summary_name=name)
   
  def add_cstate(self, name):
    self.add_tensor(name, (1 + self.DxQy.dims) * [None] + [self.cstate_depth])
   
  def add_cboundary(self, name):
    self.add_tensor(name, (1 + self.DxQy.dims) * [None] + [self.cboundary_depth])

  def rename_tensor(self, old_name, new_name):
    # this may need to be handled with more care
    self.out_tensors[new_name] = self.out_tensors[old_name]
    if old_name in self.out_pad_tensors.keys():
      self.out_pad_tensors[new_name] = self.out_pad_tensors[old_name]
    
    # add to shape converter
    for name in list(self.shape_converters.keys()):
      if name[1] == old_name:
        self.shape_converters[name[0], new_name] = copy(self.shape_converters[name])

  def lattice_summary(self, in_name, summary_name, 
                      display_norm=True, display_vel=True, display_pressure=True):
    with tf.device('/cpu:0'):
      if display_norm:
        if self.DxQy.dims == 2:
          tf.summary.image(summary_name + '_norm', self.DxQy.lattice_to_norm(self.out_tensors[in_name]))
        elif self.DxQy.dims == 3:
          tf.summary.image(summary_name + '_norm', self.DxQy.lattice_to_norm(self.out_tensors[in_name])[:,0])
      if display_pressure:
        if self.DxQy.dims == 2:
          tf.summary.image(summary_name + '_rho', self.DxQy.lattice_to_rho(self.out_tensors[in_name]))
        elif self.DxQy.dims == 3:
          tf.summary.image(summary_name + '_rho', self.DxQy.lattice_to_rho(self.out_tensors[in_name])[:,0])
      if display_vel:
        if self.DxQy.dims == 2:
          vel = self.DxQy.lattice_to_vel(self.out_tensors[in_name])
          tf.summary.image(summary_name + '_vel_x', vel[...,0:1])
          tf.summary.image(summary_name + '_vel_y', vel[...,1:2])
        elif self.DxQy.dims == 3:
          vel = self.DxQy.lattice_to_vel(self.out_tensors[in_name])[:,0]
          tf.summary.image(summary_name + '_vel_x', vel[...,0:1])
          tf.summary.image(summary_name + '_vel_y', vel[...,1:2])
  
  def boundary_summary(self, in_name, summary_name):
    with tf.device('/cpu:0'):
      if self.DxQy.dims == 2:
        tf.summary.image('physical_boundary', self.out_tensors[in_name][...,0:1])
        tf.summary.image('vel_x_boundary', self.out_tensors[in_name][...,1:2])
        tf.summary.image('vel_y_boundary', self.out_tensors[in_name][...,2:3])
        if len(self.out_tensors[in_name].get_shape()) == 5:
          tf.summary.image('vel_z_boundary', self.out_tensors[in_name][...,3:4])
        tf.summary.image('density_boundary', self.out_tensors[in_name][...,-1:])
 
  def gradients(self, loss_name, grad_name, params):
    self.out_tensors[grad_name] = tf.gradients(self.out_tensors[loss_name], params)
 
  def l2_loss(self, true_name, pred_name, loss_name, normalize=True):

    if not (type(true_name) is list):
      true_name = [true_name]
      pred_name = [pred_name]
    self.out_tensors[loss_name] = 0.0 
    for i in range(len(true_name)):
      self.out_tensors[loss_name] += tf.nn.l2_loss(tf.stop_gradient(self.out_tensors[ true_name[i]]) 
                                                  - self.out_tensors[pred_name[i]])
    if normalize:
      #normalize_loss_factor = float(len(true_name)) * tf.cast(tf.reduce_prod(tf.shape(self.out_tensors[true_name[0]])), dtype=tf.float32)
      normalize_loss_factor = float(len(true_name) * self.batch_size)
      self.out_tensors[loss_name] = self.out_tensors[loss_name] / normalize_loss_factor


  def sum_losses(self, loss_names, out_name, normalize=True):
    self.out_tensors[out_name] = 0.0 
    for n in loss_names:
      self.out_tensors[out_name] += self.out_tensors[n]
    if normalize:
      self.out_tensors[out_name] += self.out_tensors[out_name] / float(len(loss_names))
    with tf.device('/cpu:0'):
      tf.summary.scalar(out_name, self.out_tensors[out_name])

  def sum_gradients(self, gradient_names, out_name):
    for i in range(1, len(gradient_names)):
      for j in range(len(self.out_tensors[gradient_names[i]])):
        if self.out_tensors[gradient_names[i]][j] is not None:
          self.out_tensors[gradient_names[0]][j] += self.out_tensors[gradient_names[i]][j]
    self.out_tensors[out_name] = self.out_tensors[gradient_names[0]]




