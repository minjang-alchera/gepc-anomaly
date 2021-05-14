from .ActionsEstLoader import TSSTG
from collections import deque
from .config import config
import numpy as np

class Fall_Detector():
    def __init__(self, image_size, channel=1):
        self.model = TSSTG(config.load_model, config.device)
        self.data = {}
        self.image_size = image_size
        self.channels = channel
        for c in range(channel):
            self.data[c] = {}

    def run(self, detections):
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
                    out = self.model.predict(pts, self.image_size)
                    action_name = self.model.class_names[out[0].argmax()]
                    detections[channel][idx]['Action'] = {'action':action_name, 'confidence':out[0].max() * 100}
        return detections

    def preprocess(self, detection):
        skeleton = detection['skeleton'][:2] + detection['skeleton'][10:]
        skeleton_confidence = [detection['skeleton_confidence'][0]] + detection['skeleton_confidence'][5:]
        output = np.concatenate([np.array(skeleton).reshape((-1, 2)), np.array(skeleton_confidence).reshape((-1, 1))], axis=1)
        return output