import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '/home/work/qinhuan/git/caffe-ssd'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(3)
caffe.set_mode_gpu()
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = '/home/work/qinhuan/git/vision-detector/test/test_wwl/toqinhuan/labelmap.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


model_def = '/home/work/qinhuan/git/vision-detector/test/test_wwl/toqinhuan/smallssd_specsample/deploy.prototxt'
model_weights = '/tmp/toqinhuan/smallssd_specsample/final.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([114,115,108])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].reshape(1,3,64,256)
import cv2

fout = open('/home/work/qinhuan/git/vision-detector/test/test_wwl/toqinhuan/txts/res_baseline_smallssd_specsample.txt', 'w')
img_dir = '/home/work/tester/data/KITTI'
test_list = '/home/work/qinhuan/git/vision-detector/test/test_wwl/toqinhuan/list.txt'
with open(test_list) as f:
    while True:
        line = f.readline()
        if line == '':
            break
        line = line.strip()
        print line
        img = caffe.io.load_image(img_dir + '/images/' + line)
        t1 = transformer.preprocess('data', img)
        for index, t in enumerate([t1]):
            net.blobs['data'].data[...] = t
            # Forward pass.
            detections = net.forward()['detection_out']
            # Parse the outputs.
            det_label = detections[0,0,:,1]
            det_conf = detections[0,0,:,2]
            det_xmin = detections[0,0,:,3]
            det_ymin = detections[0,0,:,4]
            det_xmax = detections[0,0,:,5]
            det_ymax = detections[0,0,:,6]
            
            print det_conf
            # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.0]
            print top_indices
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_labels = get_labelname(labelmap, top_label_indices)
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            print top_conf.shape[0]
            for i in xrange(top_conf.shape[0]):
                xmin = (top_xmin[i] * img.shape[1])
                ymin = (top_ymin[i] * img.shape[0])
                xmax = (top_xmax[i] * img.shape[1])
                ymax = (top_ymax[i] * img.shape[0])
                score = top_conf[i]
                label = top_labels[i]
                fout.write(line + ' ')
                fout.write('%s %f %f %f %f %f' % (label, score, xmin, ymin, xmax, ymax))
                fout.write('\n')
fout.close()
