from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import torch
import torchvision.transforms as transforms
from ..config import config
from ..Models.model import create_model, load_model

class BaseDetector(object):
    def __init__(self):
        print('Creating model...')
        self.model = create_model(config.task, config.arch, config.heads, config.head_conv)
        self.model = load_model(self.model, config.load_model)
        self.model = self.model.to(config.device)
        self.model.eval()

        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.max_per_image = 100
        self.num_classes = config.num_classes
        self.scales = config.test_scales
        self.config = config
        self.pause = True

    def preprocess(self, images):
        tensor = []
        meta = []
        for image in images:
            height, width = image.shape[0:2]
            scale = min(config.input_w / width, config.input_h / height)
            new_width = int(scale * width)
            new_height = int(scale * height)

            delta_width = abs(self.config.input_w - new_width)
            delta_height = abs(self.config.input_h - new_height)

            top = round(delta_height / 2 + 0.5)
            bottom = round(delta_height / 2 - 0.5)
            left = round(delta_width / 2 + 0.5)
            right = round(delta_width / 2 - 0.5)
            meta.append({'s': scale, 'original_size': (width, height), 'top': top, 'left': left})
            tensor.append(
                self.norm(torch.from_numpy(cv2.copyMakeBorder(cv2.resize(image, (new_width, new_height)), top, bottom, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))).to(config.device).float().permute(2, 0, 1) / 255.0).unsqueeze(0))
        return torch.cat(tensor, 0), meta


    def run(self, images):
        tensor, meta = self.preprocess(images)
        dets = self.process(tensor)
        dets = self.postprocess(dets, meta)
        return dets