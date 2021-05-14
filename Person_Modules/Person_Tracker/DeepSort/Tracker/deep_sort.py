import numpy as np

from .nn_matching import NearestNeighborDistanceMetric
from .preprocessing import non_max_suppression
from .detection import Detection
from .tracker import Tracker


class DeepSort(object):
    def __init__(self, channel=1):
        max_cosine_distance = 0.2
        nn_budget = 100
        self.channel = channel
        self.tracker = []
        for c in range(channel):
            self.tracker.append(Tracker(NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)))

    def to_cxcywh(self, box):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        cx = (x2 - x1) / 2 + x1
        cy = (y2 - y1) / 2 + y1
        w = x2 - x1
        h = y2 - y1
        return [cx, cy, w, h]

    def detection_to_input(self, detection):
        out = []
        for d, det in enumerate(detection):
            out.append(Detection(self.to_cxcywh(det['box']), det['confidence'], det['feature'], det['skeleton'], det['skeleton_confidence']))
        return out

    # def run(self, bbox_xywh, confidences, ori_img):
    def run(self, detections):
        # update tracker
        outputs = {}
        for channel in range(self.channel):
            detection = self.detection_to_input(detection=detections[channel])
            self.tracker[channel].predict()
            self.tracker[channel].update(detection)

            # output bbox identities
            outputs[channel] = []
            for track in self.tracker[channel].tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                outputs[channel].append({'box': self._xywh_to_xyxy(track.current_box),
                                         'id': track.track_id,
                                         'skeleton': track.skeleton,
                                         'skeleton_confidence': track.skeleton_confidence.tolist(),
                                         'feature': track.detection_feature,
                                         'confidence': track.confidence,
                                         'segmentation': track.segmentation})
        return outputs

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = int(x - w / 2)
        x2 = int(x + w / 2)
        y1 = int(y - h / 2)
        y2 = int(y + h / 2)
        return x1, y1, x2, y2

if __name__ == '__main__':
    pass
