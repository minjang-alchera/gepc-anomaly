from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
from pycocotools import  mask as mask_utils
from ..config import config
from ..Models.decode import mpseg_decode
from ..Models.utils import flip_tensor
from ..Utils.image import get_affine_transform
from ..Utils.post_process import mpseg_post_process

from .base_detector import BaseDetector


class MPSegDetector(BaseDetector):
    def __init__(self):
        super(MPSegDetector, self).__init__()

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            if self.config.hm_hp and not self.config.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()
            seg_feat = output['seg_feat']
            conv_weigt = output['conv_weight']
            reg = output['reg'] if self.config.reg_offset else None
            hm_hp = output['hm_hp'] if self.config.hm_hp else None
            hp_offset = output['hp_offset'] if self.config.reg_hp_offset else None
            assert not self.config.flip_test,"not support flip_test"
            torch.cuda.synchronize()
            forward_time = time.time()
            # def mpseg_decode(
            #         heat, wh, kps, seg_feat, conv_weight, reg=None, cat_spec_wh=False, hm_hp=None, hp_offset=None,
            #         K=100):
            dets, masks, debug_heatmap = mpseg_decode(hm, wh, output['hps'], seg_feat, conv_weigt, reg=reg,
                                      cat_spec_wh=self.config.cat_spec_wh, hm_hp=hm_hp, hp_offset=hp_offset, K=self.config.K)
            # dets,masks = ctseg_decode(hm, wh,seg_feat, conv_weigt, reg=reg, cat_spec_wh=self.config.cat_spec_wh, K=self.config.K)
        return output, (dets,masks), debug_heatmap

    def post_process(self, det_seg, meta, scale=1):
        assert scale == 1, "not support scale != 1"
        dets,seg = det_seg
        dets = dets.detach().cpu().numpy()
        seg = seg.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = mpseg_post_process(
            dets.copy(), seg.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'],*meta['img_size'], self.config.num_classes)
        return dets[0]

    def merge_outputs(self, detections):
        return detections[0]
