from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from ..config import config
from ..Models.decode import multi_pose_decode
from ..Utils.post_process import multi_pose_post_process

from .base_detector import BaseDetector


class MultiPoseDetector(BaseDetector):
    def __init__(self):
        super(MultiPoseDetector, self).__init__()
        self.flip_idx = config.flip_idx

    def process(self, images):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            output['hm_hp'] = output['hm_hp'].sigmoid_()

            reg = output['reg']
            hm_hp = output['hm_hp']
            hp_offset = output['hp_offset']
            torch.cuda.synchronize()
            dets = multi_pose_decode(
                output['hm'], output['wh'], output['hps'],
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.config.K)

        return dets

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        results[1] = results[1].tolist()
        return results

    def postprocess(self, dets, meta):
        batch = dets.size(0)
        detections = {}
        for b in range(batch):
            detections[b] = []
            det = dets[b, (dets[b, :, 4] > 0.5), :].detach().cpu().numpy()
            for i in range(0, 4, 2):
                det[:, i] = np.clip(((det[:, i] * 4) - meta[b]['left']) * (1.0 / meta[b]['s']), 0.0, meta[b]['original_size'][0])
                det[:, i + 1] = np.clip(((det[:, i + 1] * 4) - meta[b]['top']) * (1.0 / meta[b]['s']), 0.0, meta[b]['original_size'][1])

            # det[:, 5:39] = (det[:, 5:39] * 4) * (1.0 / meta[b]['s'])
            for i in range(5, 39, 2):
                det[:, i] = np.clip(((det[:, i] * 4) - meta[b]['left']) * (1.0 / meta[b]['s']), 0.0, meta[b]['original_size'][0])
                det[:, i + 1] = np.clip(((det[:, i + 1] * 4) - meta[b]['top']) * (1.0 / meta[b]['s']), 0.0, meta[b]['original_size'][1])

            if self.config.reject_small_person is True:
                for i in range(det.shape[0]):
                    if ((det[i, 2] - det[i, 0]) * (det[i, 3] - det[i, 1])) / (meta[b]['original_size'][0] * meta[b]['original_size'][1]) > self.config.reject_ratio:
                        detections[b].append({'box': det[i, :4].tolist(),
                                              'skeleton': det[i, 5:39].tolist(),
                                              'confidence': det[i, 4],
                                              'skeleton_confidence':det[i, 39:56]})
            else:
                for i in range(det.shape[0]):
                    detections[b].append({'box':det[i, :4].tolist(),
                                          'skeleton':det[i, 5:39].tolist(),
                                          'confidence':det[i, 4],
                                          'skeleton_confidence':det[i, 39:56]})
        return detections