#!/usr/bin/env python
# coding=utf-8

import box_proc as BoxProc

def read_bboxes(annotation_file, classes=['Car'], min_size=0):
    """ Read bounding boxes from annotation file list.
    """
    def get_box(box_info):
        if box_info[0] not in classes:
            return None
        box = [float(x) for x in box_info[4:8]]
        return BoxProc.Box(box, tp='xyxy')

    samples = []
    with open(annotation_file) as f:
        lines = iter(f.readlines())
        while True:
            try:
                # image name
                image_name = lines.next().split()[0]
                # image num
                bboxes_num = int(lines.next().split()[0])
                # bboxes
                bboxes = []
                for i in range(bboxes_num):
                    box_info = lines.next().split()
                    box = get_box(box_info)
                    if box is not None and box.w > min_size and box.h > min_size:
                        bboxes.append(box)

                samples.append({'name':image_name, 'bboxes':bboxes})
            except StopIteration:
                break

    return samples


def save_bboxes(annotation_file):
    """ Save bounding boxes from annotation file list.
    """
    pass
