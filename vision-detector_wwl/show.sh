#!/bin/sh
# setting caffe root folder
CAFFE_ROOT=/home/jszhujun2010/code/caffe-ssd-cmake
CURRENT_DIR=`pwd`
# add pycaffe path
export PYTHONPATH=$CAFFE_ROOT/python:$CURRENT_DIR

model_defs='../Merge/models/detection/kitti_128x512_vgg16-channel16-nopooling12_iter_240000.prototxt'
model_weights='../Merge/models/detection/kitti_128x512_vgg16-channel16-nopooling12_iter_240000.caffemodel'
detection_out='detection_out_kitti'
#model_defs='../Merge/models/detection/ssd-vgg16-8-kitti-v1.prototxt'
#model_weights='../Merge/models/detection/ssd-vgg16-8-kitti-v1.caffemodel'
test_list=data/KITTI/test_list.txt

python ./app/detector.py $model_defs $model_weights $detection_out $test_list --threshod 0.1 
