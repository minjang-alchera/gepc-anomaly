import torch

from .models.dc_gcae.dc_gcae import load_ae_dcec
from collections import deque
from .config import config
import numpy as np
from .utils.data_utils import normalize_pose
from .utils.pose_seg_dataset import keypoints17_to_coco18

class Anomaly_Detector():
    def __init__(self, image_size, channel=1):
        self.model = load_ae_dcec(config.load_model)
        self.data = {}
        self.image_size = image_size
        self.channels = channel
        for c in range(channel):
            self.data[c] = {}

    def run(self, detections):
        dataset_args = {'transform_list': None, 'debug': False, 'headless': False,
                        'scale': False, 'scale_proportional': False, 'seg_len': 12,
                        'patch_size': 64, 'return_indices': True, 'return_metadata': True, 'start_ofst': 0,
                        'seg_stride': 1,
                        'train_seg_conf_th': 0.0}

        vid_res = dataset_args.get('vid_res', [self.image_size[0], self.image_size[1]])

        for channel in range(self.channels):
            for idx in range(len(detections[channel])):
                detection = detections[channel][idx]
                detections[channel][idx]['Action'] = None
                if detection['id'] is None:
                    continue
                if detection['id'] not in self.data[channel]:
                    self.data[channel][detection['id']] = deque(maxlen=config.frame_length)
                self.data[channel][detection['id']].append(self.preprocess(detection=detection))
                if len(self.data[channel][detection['id']]) >= config.frame_length:
                    pts = np.array(self.data[channel][detection['id']], dtype=np.float32)
                    pts = normalize_pose(pts, vid_res=vid_res, **dataset_args)
                    pts = keypoints17_to_coco18(pts)
                    pts = np.transpose(pts, (2, 0, 1)).astype(np.float32)
                    pts = torch.from_numpy(pts).cuda()
                    out = self.model.predict(pts)
                    print("=====================================")
                    print(out)
        return detections

    def preprocess(self, detection):
        skeleton = detection['skeleton']
        skeleton_confidence = detection['skeleton_confidence']
        output = np.concatenate([np.array(skeleton).reshape((-1, 2)), np.array(skeleton_confidence).reshape((-1, 1))], axis=1)
        return output