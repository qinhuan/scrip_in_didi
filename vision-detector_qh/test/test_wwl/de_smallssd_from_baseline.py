import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import array
# Make sure that caffe is on the python path:
caffe_root = '/home/work/qinhuan/git/caffe-ssd'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

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

def non_max_suppression_fast(x1, y1, x2, y2, conf, overlapThresh):
    if len(x1) == 0:
        return []
    pick = []
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (area[idxs[:last]] + area[i] - (w * h))
        idxs = np.delete(idxs, np.where(overlap > overlapThresh)[0])
        
        overlap2 = (w * h) / (area[i])
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap1 = (w * h) / (area[idxs[:last]])
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap1 > 0.8)[0])))
    return pick

if __name__ == '__main__':
    caffe.set_device(2)
    caffe.set_mode_gpu()
    # load PASCAL VOC labels
    labelmap_file = '/home/work/qinhuan/git/vision-detector/test/test_wwl/toqinhuan/labelmap.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    model_def = '/home/work/qinhuan/git/vision-detector/test/test_wwl/mymodel/baseline-smallssd/deploy.prototxt'
    model_weights = '/home/work/qinhuan/git/vision-detector/test/test_wwl/mymodel/baseline-smallssd/final.caffemodel'
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

    fout = open('/home/work/qinhuan/git/vision-detector/test/test_wwl/toqinhuan/txts/res_smallssd_from_baseline_0.5__0.8.txt', 'w')
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
            yl = int(img.shape[0] * 0.3)
            yr = int(img.shape[0] * 0.7)
            xl1 = 0
            xr1 = int(img.shape[1] * (1.0 / 3 + 0.1))
            xl2 = int(img.shape[1] * (1.0 / 3 - 0.1))
            xr2 = int(img.shape[1] * (2.0 / 3 + 0.1))
            xl3 = int(img.shape[1] * (2.0 / 3 - 0.1))
            xr3 = int(img.shape[1])
            t1 = transformer.preprocess('data', img)
            t2 = transformer.preprocess('data', img[yl:yr, xl1:xr1])
            t3 = transformer.preprocess('data', img[yl:yr, xl2:xr2])
            t4 = transformer.preprocess('data', img[yl:yr, xl3:xr3])
            Label = []
            Conf = []
            Xmin = []
            Ymin = []
            Xmax = []
            Ymax = []
            cnt1 = 0
            cnt2 = 0
            cnt3 = 0
            cnt4 = 0
            for index, t in enumerate([t1, t2, t3, t4]):
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

                # Get detections with confidence higher than 0.6.
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.0]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_labels = get_labelname(labelmap, top_label_indices)
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]
                
                #import pdb;
                #pdb.set_trace()
                for i in xrange(top_conf.shape[0]):
                    if top_labels[i] != 'Car':
                        continue
                    if index == 0:
                        xmin = (top_xmin[i] * img.shape[1])
                        ymin = (top_ymin[i] * img.shape[0])
                        xmax = (top_xmax[i] * img.shape[1])
                        ymax = (top_ymax[i] * img.shape[0])
                        if (ymax - ymin) / img.shape[0] <= 0.2:
                            continue
                        cnt1 = cnt1 + 1
                        Label.append(top_labels[i])
                        Conf.append(top_conf[i])
                        Xmin.append(xmin)
                        Ymin.append(ymin)
                        Xmax.append(xmax)
                        Ymax.append(ymax)
                    elif index == 1:
                        xmin = (top_xmin[i] * (xr1 - xl1))
                        ymin = (top_ymin[i] * (yr - yl)) + yl
                        xmax = (top_xmax[i] * (xr1 - xl1))
                        ymax = (top_ymax[i] * (yr - yl)) + yl
                        if (ymax - ymin) / img.shape[0] >= 0.25:
                            continue
                        cnt2 = cnt2 + 1
                        Label.append(top_labels[i])
                        Conf.append(top_conf[i])
                        Xmin.append(xmin)
                        Ymin.append(ymin)
                        Xmax.append(xmax)
                        Ymax.append(ymax)
                    elif index == 2:
                        xmin = (top_xmin[i] * (xr2 - xl2)) + xl2
                        ymin = (top_ymin[i] * (yr - yl)) + yl
                        xmax = (top_xmax[i] * (xr2 - xl2)) + xl2
                        ymax = (top_ymax[i] * (yr - yl)) + yl
                        if (ymax - ymin) / img.shape[0] >= 0.25:
                            continue
                        cnt3 = cnt3 + 1
                        Label.append(top_labels[i])
                        Conf.append(top_conf[i])
                        Xmin.append(xmin)
                        Ymin.append(ymin)
                        Xmax.append(xmax)
                        Ymax.append(ymax)
                    elif index == 3:
                        xmin = (top_xmin[i] * (xr3 - xl3)) + xl3
                        ymin = (top_ymin[i] * (yr - yl)) + yl
                        xmax = (top_xmax[i] * (xr3 - xl3)) + xl3
                        ymax = (top_ymax[i] * (yr - yl)) + yl
                        if (ymax - ymin) / img.shape[0] >= 0.25:
                            continue
                        cnt4 = cnt4 + 1
                        Label.append(top_labels[i])
                        Conf.append(top_conf[i])
                        Xmin.append(xmin)
                        Ymin.append(ymin)
                        Xmax.append(xmax)
                        Ymax.append(ymax)
            overlapThresh = 0.5
            ids = non_max_suppression_fast(np.array(Xmin), np.array(Ymin), np.array(Xmax), np.array(Ymax), np.array(Conf), overlapThresh)
            for i in ids:
                fout.write(line + ' ')
                fout.write('%s %f %f %f %f %f' % (Label[i], Conf[i], Xmin[i], Ymin[i], Xmax[i], Ymax[i]))
                fout.write('\n')
            #print cnt1,cnt2,cnt3,cnt4
            #print len(ids)
            #print ids
            #exit()
    fout.close()
