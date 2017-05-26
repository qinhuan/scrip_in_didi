#!/bin/sh
# setting caffe root folder
CAFFE_ROOT=/home/work/wwl/code/caffe-ssd
CURRENT_DIR=`pwd`
# add pycaffe path
export PYTHONPATH=$CAFFE_ROOT/python:$CURRENT_DIR

device_id=1
nohup python ./train/v0327_smallssd/run.py $device_id > log_smallssd.txt & 
