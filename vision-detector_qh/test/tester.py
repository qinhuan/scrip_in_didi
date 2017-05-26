#!/usr/bin/python
# -*- coding: UTF-8 -*-

''' Detection tester '''
import detection

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


# Load predict bounding boxes
def load_pred_bboxes_n(result_dir, img_names, classes):
    pred_bboxes = []
    for img_name in img_names:
        pred_file = os.path.join(result_dir, os.path.basename(img_name) + '.txt')
        bbs = BBoxes(img_name, classes)
        with open(pred_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                sp = l.split()
                if sp[0] not in classes:
                    continue
                tfx = float(sp[1])
                tfy = float(sp[2])
                brx = float(sp[3])
                bry = float(sp[4])
                score = float(sp[5])
                bbs.bboxes[sp[0]].append([tfx, tfy, brx - tfx, bry - tfy, score])
        pred_bboxes.append(bbs)
    return pred_bboxes


def load_pred_bboxes_1(pred_file, img_names, classes):
    pred_bboxes = {}
    for img_name in img_names:
        pred_bboxes[img_name] = BBoxes(img_name, classes)

    with open(pred_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            sp = l.split()
            img_name = os.path.basename(sp[0]) 
            tp, tfx, tfy, brx, bry, score = extract_box(sp, False)
            assert tfx <= brx and tfy <= bry, 'Illegal box {}'.format(sp)
            if tp not in classes:
                continue
            pred_bboxes[img_name].bboxes[tp].append(
                [tfx, tfy, brx - tfx, bry - tfy, score])
            #pred_bboxes[sp[0]].show()

    return [pred_bboxes[img_name] for img_name in img_names]


# Calculate ROC and Average Precision
def calc_roc_ap(result_file, gt_file, show=False, classes=[]):
    print(result_file)
    gt_bboxes = load_ground_truth(gt_file, classes)
    pred_bboxes = load_pred_bboxes_1(result_file, [x.image_name for x in gt_bboxes],
                                     classes)
    rocs = detection.multi_objects([x.bboxes for x in gt_bboxes],
                                   [x.bboxes for x in pred_bboxes],
                                   classes)
    for c in classes:
        print('Average precision of ' + c.ljust(15) + ': {}'.format(rocs[c]['AP']))
        detection.draw_ROC(rocs[c], result_file + '-{}.jpg'.format(c), c)
    return rocs

def calc_roc_aps(result_files, gt_file, show=False, classes=[]):
    rocs = []
    labels = []
    roc_fig_temp = 'results/' + dataset + '/{}'
    for rf in result_files:
        rocs.append(calc_roc_ap(rf, gt_file, show=show, classes=classes))
        label = os.path.splitext(os.path.basename(rf))[0] 
        roc_fig_temp = roc_fig_temp + '_' + label 
        labels.append(label)
    
    roc_fig_temp += '.jpg'
    
    for c in classes:
        rocs_c = [roc[c] for roc in rocs]
        detection.draw_ROCs(rocs_c, labels, roc_fig_temp.format(c), c)

def argument():
    parser = argparse.ArgumentParser(description='Detection tester')
    parser.add_argument('result_files', action='store',
                        help='Detection for predict bounding boxes, multi result files split by ","')
    parser.add_argument('--dataset', action='store', default='KITTI',
                        help='Testing dataset')
    parser.add_argument('--show', action='store', default=False,
                        help='Show or save ROC')
    parser.add_argument('--classes', action='store', default='Car,Pedestrian,Cyclist',
                        help='Classes for testing')
    return parser.parse_args()

if __name__ == '__main__':
    parser = argument()
    gt_file = 'data/' + parser.dataset + '/ground_truth.txt'
    dataset = parser.dataset
    ground_truth_dir = 'data/' + dataset + '/images'
    result_files = parser.result_files.split(',')
    if len(result_files) == 1:
        calc_roc_ap(result_files[0], gt_file, show=parser.show,
                    classes=parser.classes.split(','))
    else:
        calc_roc_aps(result_files, gt_file, show=parser.show,
                     classes=parser.classes.split(','))

