#!/usr/bin/python
# -*- coding: UTF-8 -*-

''' Test a detection models '''
import detection
from app import detector as Det 

import argparse
import os

import cv2
#import pdb

ground_truth_dir = 'data/KITTI/images'
dataset = 'KITTI'

class BBoxes:
    def __init__(self, image_name, classes):
        self.image_name = image_name
        self.bboxes = {}
        for c in classes:
            self.bboxes[c] = []

    def show(self):
        img = cv2.imread(os.path.join(ground_truth_dir, self.image_name))
        for c, bboxes in self.bboxes.items():
            for box in bboxes:
                cv2.rectangle(img, (int(box[0]), int(box[1])),
                              (int(box[0] + box[2]), int(box[1] + box[3])),
                              (255, 0, 0), 3)

        cv2.imshow('image', img)
        cv2.waitKey(0)

def extract_box(sp, is_gt=True):
    assert dataset in ['KITTI', 'DIDI']
    if dataset == 'KITTI':
        if is_gt:
            return sp[0], float(sp[4]), float(sp[5]), float(sp[6]), float(sp[7])
        else:
            return sp[1], float(sp[2]), float(sp[3]), float(sp[4]), float(sp[5]), float(sp[6])
    elif dataset == 'DIDI':
        if is_gt:
            return sp[1], float(sp[3]), float(sp[4]), float(sp[5]), float(sp[6])
        else:
            return sp[1], float(sp[2]), float(sp[3]), float(sp[4]), float(sp[5]), float(sp[6])
    else:
        pass

# Load ground truth
def load_ground_truth(gt_file, classes):
    gt_bboxes = []
    with open(gt_file, 'r') as f:
        it = iter(f.readlines())
        while True:
            try:
                img_name = it.next().split()[0]
                box_num = int(it.next())
                bbs = BBoxes(img_name, classes)
                for i in range(box_num):
                    sp = it.next().split()
                    tp, tfx, tfy, brx, bry = extract_box(sp)
                    assert tfx <= brx and tfy <= bry, 'Illegal box {} {} {} {}'.format(tfx, tfy, brx, bry)
                    if tp not in classes:
                        continue
                    bbs.bboxes[tp].append([tfx, tfy, brx - tfx, bry - tfy])
                gt_bboxes.append(bbs)
            except StopIteration:
                break

    return gt_bboxes

def calc_predict_bboxes(image_names, detector, classes):
    pred_bboxes = []
    cnt = 0
    for image_name in image_names:
        bbs = BBoxes(image_name, classes)
        boxes = detector.detect_single_image(
                os.path.join(ground_truth_dir, image_name))
        for box in boxes:
            cls = classes[box.label - 1]
            bbs.bboxes[cls].append([box.x, box.y, box.w, box.h, box.score])
        pred_bboxes.append(bbs)
        cnt += 1
        if cnt % 100 == 0:
            print('{} done!'.format(cnt))

    return pred_bboxes

# Calculate ROC and Average Precision
def calc_roc_ap(detector, gt_file, save_dir='/tmp/roc_', classes=[]):
    gt_bboxes = load_ground_truth(gt_file, classes)
    pred_bboxes = calc_predict_bboxes([x.image_name for x in gt_bboxes], detector,
                                      classes)
    rocs = detection.multi_objects([x.bboxes for x in gt_bboxes],
                                   [x.bboxes for x in pred_bboxes],
                                   classes)
    for c in classes:
        print('Average precision of ' + c.ljust(15) + ': {}'.format(rocs[c]['AP']))
        detection.draw_ROC(rocs[c], save_dir + '-{}.jpg'.format(c), c)
    return rocs

def argument():
    parser = argparse.ArgumentParser(description='Test a detection model')
    parser.add_argument('model_defs', action='store',
                        help='prototxt')
    parser.add_argument('model_weights', action='store',
                        help='caffemodel')
    parser.add_argument('output_layer', action='store',
                        help='Detector output layer')
    parser.add_argument('--dataset', action='store', default='KITTI',
                        help='Testing dataset')
    parser.add_argument('--save_dir', action='store', default='/tmp/roc',
                        help='Show or save ROC')
    parser.add_argument('--classes', action='store', default='Car,Pedestrian,Cyclist',
                        help='Classes for testing')
    return parser.parse_args()

if __name__ == '__main__':
    parser = argument()
    detector = Det.Detector(parser.model_defs, parser.model_weights, 
            parser.output_layer) 
    gt_file = '/home/work/tester/data/' + parser.dataset + '/ground_truth.txt'
    dataset = parser.dataset
    ground_truth_dir = '/home/work/tester/data/' + dataset + '/images'

    calc_roc_ap(detector, gt_file, save_dir=parser.save_dir,
                classes=parser.classes.split(','))

