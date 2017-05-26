#!/bin/sh
# setting caffe root folder
CAFFE_ROOT=/home/work/qinhuan/git/caffe-ssd
CURRENT_DIR=`pwd`
# add pycaffe path
export PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python:$CURRENT_DIR

device_id=3
nohup python ./train/v0327_smallssd/run.py > log_smallssd_min_size_sample.txt &
