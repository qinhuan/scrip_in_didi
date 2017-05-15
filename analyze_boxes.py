# convert Didi txt annotations to xml
from xml.dom.minidom import Document

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import pdb

caffe_root = '/home/work/wwl/code/caffe-ssd'
os.chdir(os.path.join(caffe_root, 'data/didi'))

def count(split_lines, img_size, class_ind, total):
    width = img_size[1]
    height = img_size[0]

    for split_line in split_lines:
        line = split_line.strip().split()
        if line[1] in class_ind:
            xmin = float(line[3])
            ymin = float(line[4])
            xmax = float(line[5])
            ymax = float(line[6])
            w = xmax - xmin
            h = ymax - ymin
            if h < 1e-6 or w / h > 5:
                continue
            ox = (xmin + xmax) / 2.0
            oy = (ymin + ymax) / 2.0
            total[line[1]]['w'].append(w / width)
            total[line[1]]['h'].append(h / height)
            total[line[1]]['ox'].append(ox / width)
            total[line[1]]['oy'].append(oy / height)
            total[line[1]]['aspect_ratio'].append(w / h)

def save_hist(total, hist_dir):
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)
    
    for cls, lists in total.items():
        for pro, lis in lists.items():
            plt.clf()
            params = {'bins':100}
            plt.hist(lis, **params)
            plt.savefig(os.path.join(hist_dir, '{}_{}.jpg'.format(cls, pro)))
            

if __name__ == '__main__':

    class_ind = ('Pedestrian', 'Car', 'Cyclist')
    prop_idx = ('w', 'h', 'ox', 'oy', 'aspect_ratio')
    total = {}
    for c in class_ind:
        total[c] = {} 
        for p in prop_idx:
            total[c][p] = []
    
    hist_dir = ('hist')
    data_dir = os.path.abspath('/home/work/data/dididata/label/object')
    train_label_file = 'annos_train.txt'
    test_label_file = 'annos_test.txt'
    if not os.path.exists(train_label_file) or not os.path.exists(test_label_file):
        os.system('python split_train_test.py {} {}'.format(train_label_file, test_label_file))

    for label_file in [train_label_file, test_label_file]:
        with open(label_file, 'r') as lb:
            lines = lb.readlines()
            it = iter(lines)
            while True:
                try:
                    img_path = it.next().split()[0]
                    img_abs_path = os.path.join(data_dir, img_path)
                    obj_num = int(it.next().split()[0])
                    split_lines = []
                    for i in range(obj_num):
                        split_lines.append(it.next())
                    assert os.path.exists(img_abs_path), \
                        'Path does not exist: {}'.format(img_path)
                    img_size = cv2.imread(img_abs_path).shape
                    count(split_lines, img_size, class_ind, total)
                except StopIteration:
                    break
    
    save_hist(total, hist_dir)
