import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/work/qinhuan/git/caffe-ssd'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = '/home/work/qinhuan/code/dataPro/labelmap_all.prototxt'
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


model_def = sys.argv[2] #'/home/work/qinhuan/git/vision-detector/jobs/ssd/TrafficLight_128x512_vgg16-channel16-nopooling12_v1/deploy.prototxt'
model_weights = sys.argv[3] #'/home/work/qinhuan/git/vision-detector/jobs/ssd/TrafficLight_128x512_vgg16-channel16-nopooling12_v1/TrafficLight_128x512_vgg16-channel16-nopooling12_iter_200000.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].reshape(1,3,128,512)
import cv2

fout = open('/home/work/qinhuan/git/vision-detector/test/res.txt', 'w')
test_list = '/home/work/qinhuan/code/dataPro/test_light.txt'
with open(test_list) as f:
    while True:
        line = f.readline()
        if line == '':
            break
        line = line.strip().split(' ')[0]
        #img = cv2.imread(line)
        xextra = int(0.35 * 2732)
        xmax = int(0.7 * 2732)
        yextra = int(0.3 * 836)
        ymax = int(0.65 * 836)
        #crop_img = img[ymin:ymax, xmin:xmax]
        save_path = '/home/work/qinhuan/code/dataPro/crop/'
        #cv2.imwrite(save_path + line.split('/')[-1], crop_img)
        #continue

        #if sys.argv[1] == '1':
        #img = caffe.io.load_image(save_path + line.split('/')[-1])
        #else:
        img = caffe.io.load_image(line)
        plt.imshow(img)
        exit()
        img = img[yextra:ymax, xextra:xmax]
        #import pdb
        #pdb.set_trace()
        transformed_image = transformer.preprocess('data', img)
        net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.0]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        if sys.argv[1] == '0':
            xextra = 0
            yextra = 0
        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1])) + xextra
            ymin = int(round(top_ymin[i] * img.shape[0])) + yextra
            xmax = int(round(top_xmax[i] * img.shape[1])) + xextra
            ymax = int(round(top_ymax[i] * img.shape[0])) + yextra
            score = top_conf[i]
            fout.write(line.split('/')[-1].split('.')[0] + ' ')
            fout.write('%f %d %d %d %d' % (score, xmin, ymin, xmax, ymax))
            fout.write('\n')
fout.close()
