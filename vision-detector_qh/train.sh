#!/bin/sh
# setting caffe root folder
CAFFE_ROOT=/home/work/qinhuan/git/caffe-ssd
CURRENT_DIR=`pwd`
# add pycaffe path
export PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python:$CURRENT_DIR

python ./train/ssd/run.py
