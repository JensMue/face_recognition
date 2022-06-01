import numpy as np
import os
import tqdm
import time
import cv2
import dlib
from mtcnn import MTCNN
import tensorflow as tf
from face_detection.models import RetinaFaceModel
from face_detection.utils import load_yaml, pad_input_image, recover_pad_output

## Create homogeneous classes and functions for different open source face detectors for easier benchmarking

class OpenCVHaarFaceDetector():
    def __init__(self,
                 scaleFactor=1.3,
                 minNeighbors=5):

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, self.scaleFactor,
                                                   self.minNeighbors)

        faces = [[x, y, x + w, y + h] for x, y, w, h in faces]

        return np.array(faces)


class DlibHOGFaceDetector():
    def __init__(self, nrof_upsample=0, det_threshold=0):
        self.hog_detector = dlib.get_frontal_face_detector()
        self.nrof_upsample = nrof_upsample
        self.det_threshold = det_threshold

    def detect_face(self, image):

        dets, scores, idx = self.hog_detector.run(image, self.nrof_upsample,
                                                  self.det_threshold)

        faces = []
        for i, d in enumerate(dets):
            x1 = int(d.left())
            y1 = int(d.top())
            x2 = int(d.right())
            y2 = int(d.bottom())

            faces.append(np.array([x1, y1, x2, y2]))

        return np.array(faces)


class DlibCNNFaceDetector():
    def __init__(self,
                 nrof_upsample=0,
                 model_path='face_detection/data/mmod_human_face_detector.dat'):

        self.cnn_detector = dlib.cnn_face_detection_model_v1(model_path)
        self.nrof_upsample = nrof_upsample

    def detect_face(self, image):

        dets = self.cnn_detector(image, self.nrof_upsample)

        faces = []
        for i, d in enumerate(dets):
            x1 = int(d.rect.left())
            y1 = int(d.rect.top())
            x2 = int(d.rect.right())
            y2 = int(d.rect.bottom())

            faces.append(np.array([x1, y1, x2, y2]))

        return np.array(faces)


class MTCNNFaceDetector():
    def __init__(self, min_face_size = 15, steps_threshold = [0.6, 0.7, 0.7]):

        self.min_face_size = min_face_size
        self.steps_threshold = steps_threshold  # three steps's threshold
        self.factor = 0.709  # scale factor
        
        self.detector = MTCNN(None, self.min_face_size, self.steps_threshold)

    def detect_face(self, image):
        
        faces = []
        res = self.detector.detect_faces(image)
        
        for face in res:
            x, y, w, h = face['box']
            faces.append(np.array([x, y, x+w, y+h]))

        return np.array(faces)


class RetinaFaceDetector():
    def __init__(self, cfg_path = 'configs/retinaface_res50.yaml', iou_th = 0.4, score_th = 0.02):

        self.cfg_path = cfg_path
        self.cfg = load_yaml(self.cfg_path)
        self.iou_th = iou_th  
        self.score_th = score_th
        
        self.detector = RetinaFaceModel(self.cfg, 
                                   False,         # trainable
                                   self.iou_th,
                                   self.score_th)
        
        # load last checkpoint
        checkpoint_dir = 'face_detection/checkpoints/' + self.cfg['sub_name']
        checkpoint = tf.train.Checkpoint(model=self.detector)

        if tf.train.latest_checkpoint(checkpoint_dir):
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            print("[*] load ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
        else:
            print("[*] Cannot find checkpoint")
    def detect_face(self, image):
        
        img_h, img_w , _ = image.shape
        frame = np.float32(image.copy())# convert to float32 
        frame = frame[..., ::-1] #convert BGR to RGB

        frame, pad_params = pad_input_image(frame, max_steps=max(self.cfg['steps']))

        det_faces = self.detector(frame[np.newaxis, ...]).numpy()

        det_faces = recover_pad_output(det_faces, pad_params)

        
        faces = []
        
        for face in det_faces:
            x1, y1 = int(face[0]*img_w), int(face[1]*img_h)
            x2, y2 = int(face[2]*img_w), int(face[3]*img_h)
            faces.append(np.array([x1, y1, x2, y2]))

        return np.array(faces)
    
#####################################################################
#############  Benchmarking Evaluation Tools  #######################
#####################################################################
    
def get_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    inputs:
        boxA = np.array( [ xmin,ymin,xmax,ymax ] )
        boxB = np.array( [ xmin,ymin,xmax,ymax ] )
    outputs:
        float in [0, 1]
    """

    bb1 = dict()
    bb1['x1'] = boxA[0]
    bb1['y1'] = boxA[1]
    bb1['x2'] = boxA[2]
    bb1['y2'] = boxA[3]

    bb2 = dict()
    bb2['x1'] = boxB[0]
    bb2['y1'] = boxB[1]
    bb2['x2'] = boxB[2]
    bb2['y2'] = boxB[3]

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes area
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # Compute the intersection over union 
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou

def extract_and_filter_data():
    """
    Extract bounding box ground truth from dataset annotations (val) set, 
    obtain each image path and maintain all information in one dictionary
    """
    bb_gt_collection = dict()

    with open(os.path.join('face_detection', 'data', 'widerface', 'val',
                         'label.txt'), 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip('\n')
        if '#' in line:
            image_path = os.path.join('face_detection', 'data', 'widerface', 'val',
                                      'images', line.strip(' # '))
            bb_gt_collection[image_path] = []
        else:
            line_components = line.split(' ')
            if len(line_components) > 1:

                x1 = int(line_components[0])
                y1 = int(line_components[1])
                w = int(line_components[2])
                h = int(line_components[3])

                # filter out faces smaller than 32x32 pixels
                if w > 32 and h > 32:
                    bb_gt_collection[image_path].append(
                        np.array([x1, y1, x1 + w, y1 + h]))

    return bb_gt_collection

def evaluate(face_detector, bb_gt_collection):
    """
    Evaluates the selected face detector on the collection of images
    """
    total_data = len(bb_gt_collection.keys())
    data_total_iou = 0
    data_total_precision = 0
    data_total_inference_time = 0

    # Evaluate face detector and iterate it over dataset
    for i, key in tqdm.tqdm(enumerate(bb_gt_collection), total=total_data):
        image_data = cv2.imread(key)
        face_bbs_gt = np.array(bb_gt_collection[key])
        total_gt_face = len(face_bbs_gt)

        start_time = time.time()
        face_pred = face_detector.detect_face(image_data)
        inf_time = time.time() - start_time
        data_total_inference_time += inf_time

        ### Calc average IOU, Precision, and Average inferencing time ####
        total_iou = 0
        tp = 0
        pred_dict = dict()
        for gt in face_bbs_gt:
            max_iou_per_gt = 0
            cv2.rectangle(image_data, (gt[0], gt[1]), (gt[2], gt[3]),
                          (255, 0, 0), 2)
            for i, pred in enumerate(face_pred):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                cv2.rectangle(image_data, (pred[0], pred[1]),
                              (pred[2], pred[3]), (0, 0, 255), 2)
                iou = get_iou(gt, pred)
                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou = total_iou + max_iou_per_gt

        if total_gt_face != 0:
            if len(pred_dict.keys()) > 0:
                for i in pred_dict:
                    if pred_dict[i] >= 0.5:
                        tp += 1
                precision = float(tp) / float(total_gt_face)

            else:
                precision = 0

            image_average_iou = total_iou / total_gt_face
            image_average_precision = precision

            data_total_iou += image_average_iou
            data_total_precision += image_average_precision

    result = dict()
    result['average_iou'] = float(data_total_iou) / float(total_data)
    result['average_precision'] = float(data_total_precision) / float(
        total_data)
    result['average_inferencing_time'] = float(
        data_total_inference_time) / float(total_data)

    return result