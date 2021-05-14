import cv2

COLORS_10 = [(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

POINT_COLOR = ['00A5E3', '8DD7BF', 'FF96C5', 'FF5768', 'FFBF65',
               'FC6238', 'FFD872', 'F2D4CC', 'E77577', '00CDAC',
               'CFF800', '4DD091', 'FF60A8', 'C05780', '00B0BA',
               'FFEC59', '0065A2']

EDGE_COLOR = ['ABDEE6', 'CBAACB', 'FFFFB5', 'FFCCB6', 'F3B0C3',
              'C6DBDA', 'FEE1E8', 'FED7C3', 'F6EAC2', 'ECD5E3',
              'FF968A', 'FFAEA5', 'FFC5BF', 'FFD8BE', 'FFC8A2',
              'D3F0F0', '8FCACA', 'CCE2CB']

EDGE = [[0, 1], [0, 2], [1, 3], [2, 4],
        [3, 5], [4, 6], [5, 6],
        [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12],
        [11, 13], [13, 15], [12, 14], [14, 16]]

def draw_person_detection(imgs, detections):
    for channel in range(len(detections)):
        for n in range(len(detections[channel])):
            det = detections[channel][n]
            draw_box(img=imgs[channel], box=det['box'])

def draw_person_tracking(imgs, detections):
    for channel in range(len(detections)):
        for n in range(len(detections[channel])):
            det = detections[channel][n]
            if 'id' in det.keys():
                draw_box(img=imgs[channel], box=det['box'], id=det['id'])
            else:
                draw_box(img=imgs[channel], box=det['box'])

            if 'Action' in det.keys():
                draw_action_results(img=imgs[channel], detection=det)
            draw_skeleton(img=imgs[channel], skeleton=det['skeleton'])
            draw_person_name(img=imgs[channel], detection=det)

def draw_action_results(img, detection):
    if detection['Action'] is not None:
        text = '{}: {:.2f}%'.format(detection['Action']['action'], detection['Action']['confidence'])
        text_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=1)
        org = (int(detection['box'][0]), int(detection['box'][1] - text_size[0][1]))
        if detection['Action']['action'] == 'Fall Down':
            cv2.putText(img=img, text=text, org=org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 255),
                        thickness=1)
        else:
            cv2.putText(img=img, text=text, org=org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(255, 0, 0),
                        thickness=1)

def draw_person_head(img, detection):
    skeleton = detection['skeleton']
    # 1
    cv2.circle(img=img, center=(int(skeleton[2]), int(skeleton[3])), radius=3, color=(255, 0, 0), thickness=2)
    # 3
    cv2.circle(img=img, center=(int(skeleton[6]), int(skeleton[7])), radius=3, color=(255, 0, 0), thickness=2)
    # 2
    cv2.circle(img=img, center=(int(skeleton[4]), int(skeleton[5])), radius=3, color=(0, 255, 255), thickness=2)
    # 4
    cv2.circle(img=img, center=(int(skeleton[8]), int(skeleton[9])), radius=3, color=(0, 255, 255), thickness=2)

def draw_person_with_face(imgs, detections):
    for channel in range(len(detections)):
        for n in range(len(detections[channel])):
            det = detections[channel][n]
            draw_box(img=imgs[channel], box=det['box'], id=det['id'])

def draw_person_name(img, detection):
    if 'face_distance' in detection.keys():
    # if detection['face_distance'] < 0.28:
        text_size = cv2.getTextSize(text=detection['name'], fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                                    thickness=1)
        cv2.rectangle(img, pt1=(detection['box'][0], detection['box'][1] + 10), pt2=(detection['box'][2], detection['box'][1] - 30),
                      color=(0, 139, 0), thickness=cv2.FILLED)
        org = [int(((detection['box'][0] + detection['box'][2]) / 2) - int((text_size[0][0]) / 2)), detection['box'][1]]
        cv2.putText(img, text=detection['name'], org=(org[0], org[1]), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=1)
        cv2.putText(img, text=str(round(detection['face_distance'], 3)), org=(org[0]+20, org[1] + 40), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=1)

def draw_box(img, box, id=None):
    if id is None:
        cv2.rectangle(img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 200, 0), thickness=2)
    else:
        cv2.rectangle(img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=COLORS_10[(len(COLORS_10) - 1) % id], thickness=2)

def draw_skeleton(img, skeleton):
    for point in range(0, len(skeleton), 2):
        color = tuple(int(POINT_COLOR[point // 2][i:i + 2], 16) for i in (0, 2, 4))
        cv2.circle(img=img, center=(int(skeleton[point]), int(skeleton[point + 1])), radius=3, color=color, thickness=3)
    for e in range(len(EDGE)):
        p1 = EDGE[e][0] * 2
        p2 = EDGE[e][1] * 2
        color = tuple(int(EDGE_COLOR[e][i:i + 2], 16) for i in (0, 2, 4))
        cv2.line(img=img, pt1=(int(skeleton[p1]), int(skeleton[p1 + 1])), pt2=(int(skeleton[p2]), int(skeleton[p2 + 1])), color=color, thickness=3)
