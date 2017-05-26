#!/bin/sh
# setting caffe root folder
CAFFE_ROOT=/home/work/qinhuan/git/caffe-ssd
CURRENT_DIR=`pwd`
# add pycaffe path
export PYTHONPATH=$CAFFE_ROOT/python:$CURRENT_DIR
echo $PYTHONPATH
model_defs='/home/work/qinhuan/git/vision-detector/test/test_wwl/toqinhuan/baseline/deploy.prototxt'
model_weights='/tmp/toqinhuan/baseline/final.caffemodel'

python ./test/test_models.py $model_defs $model_weights detection_out

