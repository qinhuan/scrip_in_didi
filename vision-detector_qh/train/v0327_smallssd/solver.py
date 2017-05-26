#!/usr/bin/env python
# coding=utf-8

import caffe
from caffe.proto import caffe_pb2

Params = {
        # Train parameters
        'base_lr': 0.01,
        'weight_decay': 0.0005,
        # 'lr_policy': "fixed",
        'lr_policy': "multistep",
        'stepvalue': [80000, 100000],
        'gamma': 0.1,
        'momentum': 0.9,
        'iter_size': 1,
        'max_iter': 120000,
        'snapshot': 50000,
        'display': 40,
        'average_loss': 40,
        'type': "SGD",
#        'solver_mode': None,
#        'device_id': None,
        'debug_info': False,
        'snapshot_after_train': True,
        # Test parameters
        'test_iter': None,
        'test_interval': 10000,
        'eval_type': "detection",
        'ap_version': "11point",
        'test_initialization': False,
        'show_per_class_result': True,
        }



def create():
    #assert os.path.exists(os.path.dirname(Params['snapshot_prefix']))
    solver = caffe_pb2.SolverParameter(**Params)

    return solver
