'''
This code are person / face modules
- developer: jh.son, si.jo, yj.song in alchera BAL
- date: 2021.03.19
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)

import cv2
import time
import copy
from Utils.display import *
from Person_Modules.Person_Detector.CenterNet.Detector.multi_pose import MultiPoseDetector
from Person_Modules.Person_FeatureExtractor.DeepSort.feature_extractor import Feature_Extractor
from Person_Modules.Person_Tracker.DeepSort.Tracker.deep_sort import DeepSort
from Person_Modules.Person_Action_Recognizer.Fall_Detector.fall_detector import Fall_Detector
from Person_Modules.Person_Action_Recognizer.Anomaly_detector.Anomaly_detector import Anomaly_Detector

def crop_box(img, box):
    return copy.copy(img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :])

def crop_detection_boxes(imgs, detections):
    output = {}
    for channel in range(len(detections)):
        output[channel] = []
        for n in range(len(detections[channel])):
            det = detections[channel][n]
            output[channel].append(crop_box(imgs[channel], det['box']))
    return output

def assign_feature_one_channel(detections, features):
    if len(detections) != len(features):
        print('not matched length')
        return
    for n in range(len(detections)):
        detections[n]['feature'] = features[n]
    return detections

def assign_feature(body_feature_extractor, imgs, detections):
    crop_imgs = crop_detection_boxes(imgs=imgs, detections=detections)
    for channel in range(len(crop_imgs)):
        if not crop_imgs[channel]:
            continue
        features = body_feature_extractor(crop_imgs[channel])
        detections[channel] = assign_feature_one_channel(detections[channel], features)
    return detections

def demo():
    demo = 'cam1.mp4'
    #demo = 'S015C002P019R001A043_rgb.avi'
    cap = cv2.VideoCapture(demo)
    detector = MultiPoseDetector()
    feature_extractor = Feature_Extractor()
    tracker = DeepSort()

    fall_detector = Fall_Detector(image_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    anomaly_detector = Anomaly_Detector(image_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, img = cap.read()
        if ret is False:
            break
        imgs = [img]
        detections = detector.run(imgs)
        detections = assign_feature(body_feature_extractor=feature_extractor, imgs=imgs, detections=detections)
        detections = tracker.run(detections)
        #detections = fall_detector.run(detections=detections)
        detections = anomaly_detector.run(detections=detections)

        draw_person_tracking(imgs=imgs, detections=detections)
        cv2.imshow('person_detected', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print('init')



if __name__ == '__main__':
    demo()