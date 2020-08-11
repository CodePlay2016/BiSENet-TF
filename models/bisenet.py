#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ 2019 Liming Liu     HuNan University
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
from Dataset.dataset import DataLoader
from builders import frontend_builder
import numpy as np

colors_camvid = np.array(
    [[64,128,64],
    [192,0,128],
    [0,128, 192],
    [0, 128, 64],
    [128, 0, 0],
    [64, 0, 128],
    [64, 0, 192],
    [192, 128, 64],
    [192, 192, 128],
    [64, 64, 128],
    [128, 0, 192],
    [192, 0, 64],
    [128, 128, 64],
    [192, 0, 192],
    [128, 64, 64],
    [64, 192, 128],
    [64, 64, 0],
    [128, 64, 128],
    [128, 128, 192],
    [0, 0, 192],
    [192, 128, 128],
    [128, 128, 128],
    [64, 128,192],
    [0, 0, 64],
    [0, 64, 64],
    [192, 64, 128],
    [128, 128, 0],
    [192, 128, 192],
    [64, 0, 64],
    [192, 192, 0],
    [0, 0, 0],
    [64, 192, 0]], dtype=np.float32)

colors_mvd = np.array(
    [[165, 42, 42],
    [0, 192, 0],
    [196, 196, 196],
    [190, 153, 153],
    [180, 165, 180],
    [90, 120, 150],
    [102, 102, 156],
    [128, 64, 255],
    [140, 140, 200],
    [170, 170, 170],
    [250, 170, 160],
    [96, 96, 96],
    [230, 150, 140],
    [128, 64, 128],
    [110, 110, 110],
    [244, 35, 232],
    [150, 100, 100],
    [70, 70, 70],
    [150, 120, 90],
    [220, 20, 60],
    [255, 0, 0],
    [255, 0, 100],
    [255, 0, 200],
    [200, 128, 128],
    [255, 255, 255],
    [64, 170, 64],
    [230, 160, 50],
    [70, 130, 180],
    [190, 255, 255],
    [152, 251, 152],
    [107, 142, 35],
    [0, 170, 30],
    [255, 255, 128],
    [250, 0, 30],
    [100, 140, 180],
    [220, 220, 220],
    [220, 128, 128],
    [222, 40, 40],
    [100, 170, 30],
    [40, 40, 40],
    [33, 33, 33],
    [100, 128, 160],
    [142, 0, 0],
    [70, 100, 150],
    [210, 170, 100],
    [153, 153, 153],
    [128, 128, 128],
    [0, 0, 80],
    [250, 170, 30],
    [192, 192, 192],
    [220, 220, 0],
    [140, 140, 20],
    [119, 11, 32],
    [150, 0, 255],
    [0, 60, 100],
    [0, 0, 142],
    [0, 0, 90],
    [0, 0, 230],
    [0, 80, 100],
    [128, 64, 64],
    [0, 0, 110],
    [0, 0, 70],
    [0, 0, 192],
    [32, 32, 32],
    [120, 10, 10],
    [0, 0, 0]], dtype=np.float32)

colors_avmp = np.array(
    [[0,0,0],
    [0,0,255],
    [255,255,255],
    [255,0,0],
    [0,255,0]], dtype=np.float32)

colors_dict = dict({
    'CamVid': colors_camvid,
    'MVD': colors_mvd,
    'AVMP': colors_avmp
})

def Upsampling(inputs, scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale])


def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=[strides, strides], activation_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

############ added depthwise separatable convolution module ###############
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)

def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish

def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid

def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, reuse=None):
    return tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                           name=name, reuse=reuse)

def _global_avg(inputs, pool_size, strides, padding='valid', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)

def _squeeze_excitation_layer(input, out_dim, ratio, layer_name, is_training=True, reuse=None):
    with tf.variable_scope(layer_name, reuse=reuse):
        squeeze = _global_avg(input, pool_size=input.get_shape()[1:-1], strides=1)

        excitation = _fully_connected_layer(squeeze, units=out_dim / ratio, name=layer_name + '_excitation1',
                                            reuse=reuse)
        excitation = relu6(excitation)
        excitation = _fully_connected_layer(excitation, units=out_dim, name=layer_name + '_excitation2', reuse=reuse)
        excitation = hard_sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input * excitation
        return scale

def DepthSepConv(net,out_channel,kernel=[3, 3],use_bias=False,stride=1,rate=1):
    input_channel= net.get_shape().as_list()[-1]
    filters = int(out_channel)
    return tf.layers.separable_conv2d(net, filters,kernel,
                                      strides=stride, padding="SAME",
                                      data_format='channels_last', dilation_rate=(rate,rate),
                                      depth_multiplier=1, activation=None,
                                      use_bias=use_bias)
############ added depthwise separatable convolution module ###############

############ Modified ARM and FFM ###############
def AttentionRefinementModule_Custom(inputs, n_filters):
    
    exp_size = _make_divisible(n_filters * 2)
    inputs = slim.conv2d(inputs, exp_size, [1, 1], stride=[1, 1], activation_fn=None)
    inputs = slim.batch_norm(inputs, fused=True)
    inputs = DepthSepConv(inputs, n_filters, kernel=[3, 3], stride=1)
    inputs = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    return net

def FeatureFusionModule_Custom(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    exp_size = _make_divisible(n_filters*1.5)
    inputs = slim.conv2d(inputs, exp_size, [1, 1], stride=[1, 1], activation_fn=None)
    inputs = slim.batch_norm(inputs, fused=True)
    inputs = DepthSepConv(inputs,n_filters, kernel=[3, 3], stride=1)
    inputs = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    #inference/combine_path/Conv_7/Conv2D run 1 average cost 73.475998 ms, 7.450 %, FlopsRate: 35.517 %

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)

    return net
############ Modified ARM and FFM ###############

def AttentionRefinementModule(inputs, n_filters):
    inputs = slim.conv2d(inputs, n_filters, [3, 3], activation_fn=None)
    inputs = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    return net


def FeatureFusionModule(input_1, input_2, n_filters):
    #print("-"*10)
    #input_2 = tf.Print(input_2, [tf.shape(input_2)], message="input_2::", summarize=4)
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)

    return net


class BiseNet(object):
    def __init__(self, model_config, train_config, num_classes, mode):
        self.model_config = model_config
        self.train_config = train_config
        self.num_classes = num_classes
        self.mode = mode
        assert mode in ['train', 'validation', 'inference', 'test']
        if self.mode == 'train':
            self.data_config = self.train_config['train_data_config']
        elif self.mode == 'validation':
            self.data_config = self.train_config['validation_data_config']
        elif self.mode == 'test':
            self.data_config = self.train_config['test_data_config']

        self.images = None
        self.images_feed = None
        self.labels = None
        self.net = None
        self.sup1 = None
        self.sup2 = None
        self.init_fn = None
        self.loss = None
        self.total_loss = []
        self.response = None

        with tf.device("/cpu:0"):
            self.dataset = DataLoader(self.data_config, self.train_config['DataSet'], self.train_config['dataset_dir'], self.train_config['class_dict'])

    def build_inputs(self):
        """Input fetching and batching

        Outputs:
          self.images: image batch of shape [batch, hz, wz, 3]
          labels: image batch of shape [batch, hx, wx, num_classes]
        """
        if self.mode in ['train', 'validation', 'test']:
            # Put data loading and preprocessing in CPU is substantially faster
            # DataSet prepare
                self.images, labels = self.dataset.get_one_batch()
                # labels = tf.Print(labels, [tf.unique(tf.reshape(labels,[-1,]))[0]], message="labels:", summarize=10)
                self.labels = tf.one_hot(labels, self.num_classes)

        else:
            self.images_feed = tf.placeholder(shape=[None, None, None, 3],
                                    dtype=tf.uint8, name='images_input')

            self.images = tf.to_float(self.images_feed)/255

    def is_training(self):
        """Returns true if the model is built for training mode"""
        return self.mode == 'train'

    def setup_global_step(self):
        global_step = tf.Variable(
            initial_value=0,
            name='global_step',
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build_bisenet_custom(self, reuse=False):
        """
        Builds the BiSeNet model.

        Arguments:
          reuse: Reuse variable or not

        Returns:
          BiSeNet model
        """
        ### The spatial path
        ### The number of feature maps for each convolution is not specified in the paper
        ### It was chosen here to be equal to the number of feature maps of a classification
        ### model at each corresponding stage
        batch_norm_params = self.model_config['batch_norm_params']
        init_method = self.model_config['conv_config']['init_method']
        down_16x_end_points=self.model_config['net_node']['16xdown:50']
        down_32x_end_points=self.model_config['net_node']['32xdown:25']
        if init_method == 'kaiming_normal':
            initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        else:
            initializer = slim.xavier_initializer()

        with tf.variable_scope('spatial_net', reuse=reuse):
            with slim.arg_scope([slim.conv2d], biases_initializer=None, weights_initializer=initializer):
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training(), **batch_norm_params):
                    # inference/spatial_net/Conv/Conv2D run 1 average cost 250.552994 ms, 25.405 %, FlopsRate: 9.064 %
                    # conv2d
                    spatial_net = slim.conv2d(self.images,16,[3,3], stride=[2,2], activation_fn=None)
                    spatial_net=hard_swish(slim.batch_norm(spatial_net,fused=True))

                    # bneck1
                    exp_size = _make_divisible(16)
                    spatial_net = slim.conv2d(spatial_net,exp_size, [1,1], stride=[1,1], activation_fn=None)
                    spatial_net=slim.batch_norm(spatial_net, fused=True)
                    spatial_net =DepthSepConv(spatial_net,16,kernel=[3, 3],stride=2)
                    spatial_net = tf.nn.relu(slim.batch_norm(spatial_net, fused=True))

                    # bneck2
                    exp_size = _make_divisible(72)
                    spatial_net = slim.conv2d(spatial_net, exp_size, [1, 1], stride=[1, 1], activation_fn=None)
                    spatial_net = slim.batch_norm(spatial_net, fused=True)
                    spatial_net = DepthSepConv(spatial_net,24, kernel=[3, 3], stride=2)
                    spatial_net = tf.nn.relu(slim.batch_norm(spatial_net, fused=True))
                    # bneck3
                    exp_size = _make_divisible(88)
                    spatial_net = slim.conv2d(spatial_net, exp_size, [1, 1], stride=[1, 1], activation_fn=None)
                    spatial_net = slim.batch_norm(spatial_net, fused=True)
                    spatial_net = DepthSepConv(spatial_net, 24, kernel=[3, 3], stride=1)
                    spatial_net = tf.nn.relu(slim.batch_norm(spatial_net, fused=True))
                    # bneck4
                    exp_size = _make_divisible(96)
                    spatial_net = slim.conv2d(spatial_net, exp_size, [1, 1], stride=[1, 1], activation_fn=None)
                    spatial_net = slim.batch_norm(spatial_net, fused=True)
                    spatial_net = DepthSepConv(spatial_net,40, kernel=[3, 3], stride=1)
                    spatial_net = tf.nn.relu(slim.batch_norm(spatial_net, fused=True))
                    # bneck5
                    spatial_net = DepthSepConv(spatial_net,80, kernel=[3, 3], stride=1)
                    spatial_net = tf.nn.relu(slim.batch_norm(spatial_net, fused=True))
                    # bneck6
                    spatial_net = DepthSepConv(spatial_net,128, kernel=[3, 3], stride=1)
                    spatial_net = tf.nn.relu(slim.batch_norm(spatial_net, fused=True))

        frontend_config = self.model_config['frontend_config']
        ### Context path
        logits, end_points, frontend_scope, init_fn = frontend_builder.build_frontend(self.images, frontend_config,
                                                                                      self.is_training(), reuse)

        ### Combining the paths
        with tf.variable_scope('combine_path', reuse=reuse):
            with slim.arg_scope([slim.conv2d], biases_initializer=None, weights_initializer=initializer):
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training(), **batch_norm_params):
                    # tail part
                    global_context = tf.reduce_mean(end_points[down_32x_end_points], [1, 2], keep_dims=True)
                    global_context = slim.conv2d(global_context, 128, 1, [1, 1], activation_fn=None)
                    global_context = tf.nn.relu(slim.batch_norm(global_context, fused=True))
                    ARM_out1 = AttentionRefinementModule_Custom(end_points[down_32x_end_points], n_filters=128)
                    ARM_out2 = AttentionRefinementModule_Custom(end_points[down_16x_end_points], n_filters=128)

                    ARM_out1 = tf.add(ARM_out1, global_context)
                    ARM_out1 = Upsampling(ARM_out1, scale=2)
                    # inference/combine_path/Conv_6/Conv2D run 1 average cost 23.034000 ms, 2.336 %, FlopsRate: 8.879 %
                    exp_size = _make_divisible(256)
                    ARM_out1 = slim.conv2d(ARM_out1, exp_size, [1, 1], stride=[1, 1], activation_fn=None)
                    ARM_out1 = slim.batch_norm(ARM_out1, fused=True)
                    ARM_out1 = DepthSepConv(ARM_out1, 128, kernel=[3, 3], stride=1)
                    ARM_out1 = tf.nn.relu(slim.batch_norm(ARM_out1, fused=True))
                    ARM_out2 = tf.add(ARM_out2, ARM_out1)
                    ARM_out2 = Upsampling(ARM_out2, scale=2)
                    # inference/combine_path/Conv_13/Conv2D run 1 average cost 23.034000 ms, 2.336 %, FlopsRate: 8.879 %
                    exp_size = _make_divisible(256)
                    ARM_out2 = slim.conv2d(ARM_out2, exp_size, [1, 1], stride=[1, 1], activation_fn=None)
                    ARM_out2 = slim.batch_norm(ARM_out2, fused=True)
                    ARM_out2 = DepthSepConv(ARM_out2, 128, kernel=[3, 3], stride=1)
                    ARM_out2 = tf.nn.relu(slim.batch_norm(ARM_out2, fused=True))
                    context_net = ARM_out2

                    FFM_out = FeatureFusionModule_Custom(input_1=spatial_net, input_2=context_net, n_filters=256)

                    ARM_out1 = ConvBlock(ARM_out1, n_filters=128, kernel_size=[3, 3])
                    ARM_out2 = ConvBlock(ARM_out2, n_filters=128, kernel_size=[3, 3])
                    exp_size = _make_divisible(128)
                    FFM_out = slim.conv2d(FFM_out, exp_size, [1, 1], stride=[1, 1], activation_fn=None)
                    FFM_out = slim.batch_norm(FFM_out, fused=True)
                    FFM_out = DepthSepConv(FFM_out,64, kernel=[3, 3], stride=1)
                    FFM_out = tf.nn.relu(slim.batch_norm(FFM_out, fused=True))
                    # Upsampling + dilation or only Upsampling
                    FFM_out = Upsampling(FFM_out, scale=2)
                    # inference/combine_path/Conv_12/Conv2D run 1 average cost 32.151001 ms, 3.260 %, FlopsRate: 8.879 %
                    exp_size = _make_divisible(128)
                    FFM_out = slim.conv2d(FFM_out, exp_size, [1, 1], stride=[1, 1], activation_fn=None)
                    FFM_out = DepthSepConv(FFM_out, 64, kernel=[3, 3], stride=1,rate=2)
                    FFM_out = tf.nn.relu(slim.batch_norm(FFM_out, fused=True))
                    FFM_out = slim.conv2d(FFM_out, self.num_classes, [1, 1], activation_fn=None, scope='logits')
                    self.net = Upsampling(FFM_out, 4)

                    if self.mode in ['train', 'validation', 'test']:
                        sup1 = slim.conv2d(ARM_out1, self.num_classes, [1, 1], activation_fn=None, scope='supl1')
                        sup2 = slim.conv2d(ARM_out2, self.num_classes, [1, 1], activation_fn=None, scope='supl2')
                        self.sup1 = Upsampling(sup1, scale=16)
                        self.sup2 = Upsampling(sup2, scale=8)
                        self.init_fn = init_fn

    def build_bisenet(self, reuse=False):
        """
        Builds the BiSeNet model.

        Arguments:
          reuse: Reuse variable or not

        Returns:
          BiSeNet model
        """

        ### The spatial path
        ### The number of feature maps for each convolution is not specified in the paper
        ### It was chosen here to be equal to the number of feature maps of a classification
        ### model at each corresponding stage
        batch_norm_params = self.model_config['batch_norm_params']
        init_method = self.model_config['conv_config']['init_method']
        down_16x_end_points=self.model_config['net_node']['16xdown:50']
        down_32x_end_points=self.model_config['net_node']['32xdown:25']
        if init_method == 'kaiming_normal':
            initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        else:
            initializer = slim.xavier_initializer()

        with tf.variable_scope('spatial_net', reuse=reuse):
            with slim.arg_scope([slim.conv2d], biases_initializer=None, weights_initializer=initializer):
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training(), **batch_norm_params):
                    #print("*"*20)
                    print(self.images)
                    #print("*" * 20)
                    spatial_net = ConvBlock(self.images, n_filters=64, kernel_size=[7, 7], strides=2)
                    spatial_net = ConvBlock(spatial_net, n_filters=64, kernel_size=[3, 3], strides=2)
                    spatial_net = ConvBlock(spatial_net, n_filters=64, kernel_size=[3, 3], strides=2)
                    spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[1, 1])

        frontend_config = self.model_config['frontend_config']
        ### Context path
        logits, end_points, frontend_scope, init_fn = frontend_builder.build_frontend(self.images, frontend_config,
                                                                                      self.is_training(), reuse)

        ### Combining the paths
        with tf.variable_scope('combine_path', reuse=reuse):
            with slim.arg_scope([slim.conv2d], biases_initializer=None, weights_initializer=initializer):
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training(), **batch_norm_params):
                    # tail part
                    size = tf.shape(end_points[down_32x_end_points])[1:3]
                    global_context = tf.reduce_mean(end_points[down_32x_end_points], [1, 2], keep_dims=True)
                    global_context = slim.conv2d(global_context, 128, 1, [1, 1], activation_fn=None)
                    global_context = tf.nn.relu(slim.batch_norm(global_context, fused=True))
                    net_5 = AttentionRefinementModule(end_points[down_32x_end_points], n_filters=128)
                    net_4 = AttentionRefinementModule(end_points[down_16x_end_points], n_filters=128)

                    net_5 = tf.add(net_5, global_context)
                    net_5 = Upsampling(net_5, scale=2)
                    net_5 = ConvBlock(net_5, n_filters=128, kernel_size=[3, 3])
                    #net_4=net_5
                    net_4 = tf.add(net_4, net_5)
                    net_4 = Upsampling(net_4, scale=2)
                    net_4 = ConvBlock(net_4, n_filters=128, kernel_size=[3, 3])

                    context_net = net_4

                    net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, n_filters=256)
                    net_5 = ConvBlock(net_5, n_filters=128, kernel_size=[3, 3])
                    net_4 = ConvBlock(net_4, n_filters=128, kernel_size=[3, 3])
                    net = ConvBlock(net, n_filters=64, kernel_size=[3, 3])
                    
                    # Upsampling + dilation or only Upsampling
                    net = Upsampling(net, scale=2)
                    net = slim.conv2d(net, 64, [3, 3], rate=2, activation_fn=tf.nn.relu, biases_initializer=None,
                                      normalizer_fn=slim.batch_norm)

                    net = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None, scope='logits')
                    self.net = Upsampling(net, 4)

                    # net = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None, scope='logits')
                    # self.net = Upsampling(net, scale=8)

                    if self.mode in ['train', 'validation', 'test']:
                        sup1 = slim.conv2d(net_5, self.num_classes, [1, 1], activation_fn=None, scope='supl1')
                        sup2 = slim.conv2d(net_4, self.num_classes, [1, 1], activation_fn=None, scope='supl2')
                        self.sup1 = Upsampling(sup1, scale=16)
                        self.sup2 = Upsampling(sup2, scale=8)
                        self.init_fn = init_fn

    def build_loss(self):
        # self.labels = tf.Print(self.labels, [tf.unique(tf.reshape(self.labels, (-1,)))[0]], message="label:", summarize=10)
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.net, labels=self.labels))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.sup1, labels=self.labels))
        loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.sup2, labels=self.labels))
        loss = tf.add_n([loss1, loss2, loss3])

        # self.loss = loss1
        # self.total_loss = tf.losses.get_total_loss()
        return loss

    def summarize(self):
        shape = tf.shape(self.labels)

        # Tensorboard inspection
        tf.summary.image('image', self.images, family=self.mode, max_outputs=1)
        # tf.Print(self.labels, [tf.shape(self.labels)], message="label size:", summarize=10)
        color_map = colors_dict[self.train_config['DataSet']]
        tf.summary.image('GT', tf.reshape(
            tf.matmul(tf.reshape(self.labels, [-1, self.num_classes]), color_map), [-1, shape[1], shape[2], 3]),
                         family=self.mode, max_outputs=1)
        tf.summary.image('response', tf.reshape(tf.matmul(
            tf.reshape(tf.one_hot(tf.argmax(self.net, -1), self.num_classes), [-1, self.num_classes]), color_map),
            [-1, shape[1], shape[2], 3]), family=self.mode, max_outputs=1)
        tf.summary.scalar('total_loss', tf.reduce_mean(self.total_loss), family=self.mode)
        # tf.summary.scalar('loss', self.loss, family=self.mode)

        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(tf.argmax(self.net, -1),
                                                                          tf.argmax(self.labels, -1))
        mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=tf.argmax(self.net, -1),
                                                                          labels=tf.argmax(self.labels, -1),
                                                                          num_classes=self.num_classes)
        with tf.control_dependencies([accuracy_update, mean_IOU_update]):
            tf.summary.scalar('accuracy', accuracy, family=self.mode)
            tf.summary.scalar('mean_IOU', mean_IOU, family=self.mode)

    def predict(self):
        self.response = self.net

    def build(self, num_gpus=1, reuse=False):
        """Creates all ops for training and evaluation"""
        with tf.name_scope(self.mode):
            if self.mode in ['train', 'validation', 'test']:
                tower_losses = []
                for i in range(num_gpus):
                    self.build_inputs()
                    with tf.device('/gpu:%d' % i):
                    # First tower has default name scope.
                        name_scope = ('clone_%d' % i) if i else ''
                        with tf.name_scope(name_scope) as scope:
                            with tf.variable_scope(
                                    tf.get_variable_scope(), reuse=True if i!=0 else None):
                                if self.model_config['use_custom']:
                                    self.build_bisenet_custom(reuse=reuse)
                                else:
                                    self.build_bisenet(reuse=reuse)
                        loss = self.build_loss()
                        self.total_loss.append(loss)
                with tf.device('/cpu:0'):
                    self.summarize()
            else:
                self.build_inputs()
                self.build_bisenet(reuse=reuse)
                self.predict()

            if self.is_training():
                self.setup_global_step()




