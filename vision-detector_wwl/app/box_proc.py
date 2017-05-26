#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2

class Box:
    def __init__(self, box, tp='xywh'):
        assert len(box) >= 4
        self.x = box[0]
        self.y = box[1]
        if tp == 'xywh':
            self.w = box[2]
            self.h = box[3]
        elif tp == 'xyxy':
            self.w = max(0, box[2] - box[0])
            self.h = max(0, box[3] - box[1])
        else:
            assert False, 'Wrong type {}'.format(tp)

        self.score = 0
        if len(box) >= 5:
            self.score = box[4]
        
        self.label = 0
        if len(box) >= 6:
            self.label = int(box[5])

    def copy(self):
        return Box(self.box())

    def box(self):
        return [self.x, self.y, self.w, self.h]

    def upper_left(self):
        return (int(self.x + 0.5), int(self.y + 0.5))

    def bottom_right(self):
        return (int(self.x + self.w + 0.5), int(self.y + self.h + 0.5))

    def area(self):
        assert self.w >= 0
        assert self.h >= 0
        return self.w * self.h

    def cross(self, box):
        u = max(self.x, box.x)
        d = min(self.x + self.w, box.x + box.w)
        l = max(self.y, box.y)
        r = min(self.y + self.h, box.y + box.h)
        return Box([u, l, d, r], tp='xyxy')

    def iou(self, box):
        cross_box = self.cross(box)
        area_i = cross_box.area()
        area_u = self.area() + box.area() - area_i
        assert area_u > 0, 'illegal box {}, {}'.format(self.box(), box.box())
        return area_i / area_u

def crop_roi(image, crop_box, new_height, new_width):
    # to integer
    crop_box.x = int(crop_box.x)
    crop_box.y = int(crop_box.y)
    crop_box.w = int(crop_box.w)
    crop_box.h = int(crop_box.h)
    if crop_box.area() <= 0:
        return None
    # get roi
    rect = np.zeros((crop_box.h, crop_box.w, image.shape[2]), dtype=image.dtype)
    image_box = Box([0, 0, image.shape[1], image.shape[0]])
    roi_box = crop_box.cross(image_box)
    dst_box = roi_box.copy()
    dst_box.x -= crop_box.x
    dst_box.y -= crop_box.y
    # copy
    try:
        ul_dst = dst_box.upper_left()
        br_dst = dst_box.bottom_right()
        ul_roi = roi_box.upper_left()
        br_roi = roi_box.bottom_right()
        #print('{} {} {} {}'.format(ul_dst, br_dst, ul_roi, br_roi))

        rect[ul_dst[1]:br_dst[1], ul_dst[0]:br_dst[0]] = \
            image[ul_roi[1]:br_roi[1], ul_roi[0]:br_roi[0]]
        if new_height == 0 and new_width == 0:
            dst_image = rect
        else:
            dst_image = cv2.resize(rect, (new_height, new_width))
        #cv2.imshow('crop image', dst_image)
        #cv2.waitKey()
        return dst_image
    except ValueError:
        import pdb
        pdb.set_trace()
        return None

def show(image, bboxes):
    """Show bounding boxes.
    """
    def get_color(label):
        return (255*((label&1)==1), 255*((label&4)==4), 255*((label&2)==2))
    for box in bboxes:
        cv2.rectangle(image, box.upper_left(), box.bottom_right(), get_color(box.label), 2)

    # show in the limited size
    MAX_HEIGHT=720
    MAX_WIDTH=1080
    if image.shape[0] > MAX_HEIGHT or image.shape[1] > MAX_WIDTH:
        ratio = min(MAX_HEIGHT * 1.0 / image.shape[0], MAX_WIDTH * 1.0 / image.shape[1])
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

    cv2.imshow('bounding box', image)
    cv2.waitKey(0)

