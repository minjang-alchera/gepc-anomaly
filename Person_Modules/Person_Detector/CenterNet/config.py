from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class config:
    task = 'multi_pose'
    arch = 'hardnet_85'
    heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
    head_conv = 256
    load_model = './../Weights/Person_Weights/multi-pose-hardnet-85-R.pth'
    device = 'cuda'
    # device = 'cpu'
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    num_classes = 1
    test_scales = [1]
    input_h = 512
    input_w = 512
    down_ratio = 4
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    reject_small_person = False
    reject_ratio = 0.03

    # segmentation
    hm_hp = True
    mse_loss = False
    reg_offset = True
    reg_hp_offset = True
    flip_test = False
    cat_spec_wh = False
    K = 10