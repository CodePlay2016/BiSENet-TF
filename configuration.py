#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ 2019 Liming Liu     HuNan University
#

"""Default configurations of model specification, training and tracking

For most of the time, DO NOT modify the configurations within this file.
Use the configurations here as the default configurations and only update
them following the examples in the `experiments` directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))

LOG_DIR = 'Logs_AVMP'  # where checkpoints, logs are saved
RUN_NAME = 'bisenet-v2'  # identifier of the experiment
train_batch_size=24
val_batch_size=8
MODEL_CONFIG = {
  'use_custom': True,

  'frontend_config': {'frontend': 'MobilenetV3',     #MobilenetV3,MobilenetV3-0.75,MobilenetV3-SMALL-1.0
                      'pretrained_dir': 'pretrainModel/',  #MobilenetV3:pretrain,
                                                     # MobilenetV3-0.75:pretrain pretrain/v3-large_224_0.75_float/pristine
                                                     #path of the pretrained frontend model.
                      'train_frontend': True,
                      'use_bn': True,
                      'bn_scale': True,
                      'bn_momentum': 0.05,
                      'bn_epsilon': 1e-6,
                      'weight_decay': 5e-4,
                      'stride': 8, },
  'conv_config': {"init_method": "kaiming_normal",
                  },
  'batch_norm_params': {"scale": True,
                        # Decay for the moving averages.
                        "decay": 0.9,
                        # Epsilon to prevent 0s in variance.
                        "epsilon": 1e-5,
                        'updates_collections': tf.GraphKeys.UPDATE_OPS,  # Ensure that updates are done within a frame
                        },
    'net_node':{    #large
                    "16xdown:50":'layer_11/output',
                    "32xdown:25":'layer_16/output'
                    #small
                    # "16xdown:50":'layer_7/output',
                    # "32xdown:25":'layer_12/output'
               }
}

TRAIN_CONFIG = {
  'DataSet': 'CamVid',
  'dataset_dir': './data/CamVid',
  'class_dict': './data/CamVid/class_dict.csv',
  'num_classes': 32,
  'train_dir': osp.join(LOG_DIR, 'checkpoints', RUN_NAME),

  'seed': 123,  # fix seed for reproducing experiments

  'train_data_config': {'preprocessing_name': 'augment',
                        'input_dir': 'train',
                        'output_dir': 'train_labels',
                        'crop_h': 800,
                        'crop_w': 800,
                        'random_scale': True,
                        'random_mirror': True,
                        'num_examples_per_epoch': 421,
                        'epoch': 400,
                        'num_gpus': num_gpus,
                        'batch_size': train_batch_size,
                        'prefetch_threads': 4, },

  'validation_data_config': {'preprocessing_name': 'None',
                             'input_dir': 'val',
                             'output_dir': 'val_labels',
                             'crop_h': 736,
                             'crop_w': 960,
                             'num_gpus': 1,
                             'batch_size': val_batch_size,
                             'prefetch_threads': 4, },

  'test_data_config': {'preprocessing_name': 'None',
                       'input_dir': 'test',
                       'output_dir': 'test_labels',
                       'crop_h': 736,
                       'crop_w': 960,
                       'num_examples_per_epoch': 421,
                       'batch_size': 8,
                       'prefetch_threads': 4,
                       'test_dir': osp.join(LOG_DIR, 'checkpoints', RUN_NAME+'test')},


  # Optimizer for training the model.
  'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD, RMSProp and MOMENTUM are supported
                       'momentum': 0.9,
                       'use_nesterov': False,
                       'decay': 0.9, },          # Discounting factor for history gradient(useful in RMSProp Mode)

  # Learning rate configs
  'lr_config': {'policy': 'polynomial',         # piecewise_constant, exponential, polynomial and cosine
                'initial_lr': 0.01,
                'power': 0.9,                   # Only useful in polynomial
                'num_epochs_per_decay': 1,
                'lr_decay_factor': 0.8685113737513527,
                'staircase': True, },

  # If not None, clip gradients to this value.
  'clip_gradients': None,

  # Frequency at which loss and global step are logged
  'log_every_n_steps': 10,

  # Frequency to save model
  'save_model_every_n_step': 421 // train_batch_size,  # save model every epoch

  # How many model checkpoints to keep. No limit if None.
  'max_checkpoints_to_keep': 20,
}

TRAIN_CONFIG_AVMP = {
  'DataSet': 'AVMP',
  'dataset_dir': '/home/weihh/python/bise-net/data/avm_seg',
  'class_dict': './data/avm_seg/class_dict.csv',
  'num_classes': 5,
  'train_dir': osp.join(LOG_DIR, 'checkpoints', RUN_NAME),

  'seed': 123,  # fix seed for reproducing experiments

  'train_data_config': {'preprocessing_name': 'augment',
                        'split': 'train',
                        'crop_h': 160,
                        'crop_w': 320,
                        'random_scale': False,
                        'random_mirror': True,
                        'num_examples_per_epoch': 4057,
                        'epoch': 300,
                        'num_gpus': num_gpus,
                        'batch_size': train_batch_size,
                        'prefetch_threads': 4, },

  'validation_data_config': {'preprocessing_name': 'None',
                             'split': 'valid',
                             'crop_h': 160,
                             'crop_w': 320,
                             'num_gpus': 1,
                             'batch_size': val_batch_size,
                             'prefetch_threads': 4, },


  # Optimizer for training the model.
  'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD, RMSProp and MOMENTUM are supported
                       'momentum': 0.9,
                       'use_nesterov': False,
                       'decay': 0.9, },          # Discounting factor for history gradient(useful in RMSProp Mode)

  # Learning rate configs
  'lr_config': {'policy': 'polynomial',         # piecewise_constant, exponential, polynomial and cosine
                'initial_lr': 1e-4,
                'power': 0.9,                   # Only useful in polynomial
                'num_epochs_per_decay': 1,
                'lr_decay_factor': 0.8685113737513527,
                'staircase': True, },

  # If not None, clip gradients to this value.
  'clip_gradients': None,

  # Frequency at which loss and global step are logged
  'log_every_n_steps': 10,

  # Frequency to save model
  'save_model_every_n_step': 4057 // train_batch_size,  # save model every epoch

  # How many model checkpoints to keep. No limit if None.
  'max_checkpoints_to_keep': 5,
}

TRAIN_CONFIG_MVD = {
  'DataSet': 'MVD',
  'dataset_dir': './data/MVD/tfrecord.full',
  'class_dict': './data/MVD/class_dict.csv',
  'num_classes': 66,
  'train_dir': osp.join(LOG_DIR, 'checkpoints', RUN_NAME),

  'seed': 123,  # fix seed for reproducing experiments

  'train_data_config': {'preprocessing_name': 'augment',
                        'split': 'training',
                        'crop_h': 800,
                        'crop_w': 800,
                        'random_scale': True,
                        'random_mirror': True,
                        'num_examples_per_epoch': 18000,
                        'epoch': 400,
                        'num_gpus': num_gpus,
                        'batch_size': train_batch_size,
                        'prefetch_threads': 4, },

  'validation_data_config': {'preprocessing_name': 'None',
                             'split': 'validation',
                             'crop_h': 736,
                             'crop_w': 960,
                             'num_gpus': 1,
                             'batch_size': val_batch_size,
                             'prefetch_threads': 4, },

  'test_data_config': {'preprocessing_name': 'None',
                       'input_dir': 'test',
                       'output_dir': 'test_labels',
                       'crop_h': 736,
                       'crop_w': 960,
                       'num_examples_per_epoch': 421,
                       'batch_size': 8,
                       'prefetch_threads': 4,
                       'test_dir': osp.join(LOG_DIR, 'checkpoints', RUN_NAME+'test')},


  # Optimizer for training the model.
  'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD, RMSProp and MOMENTUM are supported
                       'momentum': 0.9,
                       'use_nesterov': False,
                       'decay': 0.9, },          # Discounting factor for history gradient(useful in RMSProp Mode)

  # Learning rate configs
  'lr_config': {'policy': 'polynomial',         # piecewise_constant, exponential, polynomial and cosine
                'initial_lr': 0.01,
                'power': 0.9,                   # Only useful in polynomial
                'num_epochs_per_decay': 1,
                'lr_decay_factor': 0.8685113737513527,
                'staircase': True, },

  # If not None, clip gradients to this value.
  'clip_gradients': None,

  # Frequency at which loss and global step are logged
  'log_every_n_steps': 10,

  # Frequency to save model
  'save_model_every_n_step': 18000 // train_batch_size,  # save model every epoch
  # 'save_model_every_n_step': 10,  # save model every epoch

  # How many model checkpoints to keep. No limit if None.
  'max_checkpoints_to_keep': 20,
}

