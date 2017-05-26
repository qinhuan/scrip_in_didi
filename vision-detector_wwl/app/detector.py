#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os
import sys
import cv2
import argparse

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import box_proc as BoxProc

class Detector(caffe.Net):
    """
    Detector extends Net for ssd.
    """
    def __init__(self, model_file, pretrained_file, out_layer):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        self.out_layer = out_layer
        self.threshod = 0.0

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        mean=np.array([114, 115, 108])
        self.transformer.set_mean(in_, mean)
        # scale 0~1 to 0~255
        self.transformer.set_raw_scale(in_, 255.0)
        self.transformer.set_channel_swap(in_, (2, 1, 0))
    
    def set_threshod(self, threshod):
        self.threshod = threshod

    def detect_single_image(self, image_name, show=False):
        print(image_name) 
        image = caffe.io.load_image(image_name).astype(np.float32)
        h = image.shape[0]
        w = image.shape[1]
        self.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        out = self.forward()
        orig_boxes = out[self.out_layer][0][0]
        box_num = orig_boxes.shape[0]
        boxes = []

        for i in range(box_num):
            box = [0]*6
            box[0] = orig_boxes[i, 3] * w
            box[1] = orig_boxes[i, 4] * h
            box[2] = orig_boxes[i, 5] * w
            box[3] = orig_boxes[i, 6] * h
            box[4] = orig_boxes[i, 2]
            box[5] = orig_boxes[i, 1]
            box = BoxProc.Box(box, tp='xyxy') 
            if box.score > self.threshod:
                boxes.append(box)

        if show:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            BoxProc.show(image, boxes)

        return boxes

def argument():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('model_defs', action='store',
                        help='prototxt')
    parser.add_argument('model_weights', action='store',
                        help='caffemodel')
    parser.add_argument('output_layer', action='store',
                        help='Detector output layer')
    parser.add_argument('file_list', action='store',
                        help='File list')
    parser.add_argument('--threshod', action='store', default=0.0, type=float,
                        help='Threshod')
    return parser.parse_args()

if __name__ == '__main__':
    parser = argument()
    det = Detector(parser.model_defs, parser.model_weights, 
            parser.output_layer)
    det.set_threshod(parser.threshod)
    with open(parser.file_list, 'r') as f:
        for l in f.readlines():
            det.detect_single_image(l.split()[0], show=True)
