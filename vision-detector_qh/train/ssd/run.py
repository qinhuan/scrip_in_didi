from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#!/usr/bin/env python
# coding=utf-8
import os, sys
import solver as Solver
import caffe
import numpy as np
import math

resize_height = 128
resize_width = 512

data_dir = '/home/work/qinhuan/git/caffe-ssd/data'
train_lmdb = os.path.join(data_dir, 'train_lmdb')
test_lmdb = os.path.join(data_dir, 'test_lmdb')
label_map_file = os.path.join(data_dir, 'labelmap_all.prototxt')
name_size_file = os.path.join(data_dir, 'test_name_size.txt')
pretrain_model = '/home/work/qinhuan/git/vision-detector/models/vgg16_pretrain_imagenet_iter_350000.caffemodel' 
debug_iter = 101
mbox_loss = []
x = []
sample_occupy = []

num_test_image = len([l for l in open(name_size_file, 'r').readlines()]) 
job_name = 'TrafficLight_{}x{}_vgg16-channel16-nopooling12_baseline_test'.format(resize_height, resize_width)
job_dir = 'jobs/ssd/{}/'.format(job_name)
save_file_path = 'jobs/ssd/{}/loss.png'.format(job_name)

if not os.path.exists(job_dir):
    os.makedirs(job_dir)

def create_jobs():
    import ssd_vggnopool12 as Net 
    Net.Params = {
            'model_name': job_name,
            'train_lmdb': train_lmdb,
            'test_lmdb': test_lmdb,
            'label_map_file': label_map_file,
            'name_size_file': name_size_file,
            'output_result_dir': os.path.join(job_dir, 'output'),
            'resize_height': resize_height,
            'resize_width': resize_width,
            'batch_size_per_device': 64,
            'test_batch_size': 8,
            'freeze_layers': [],                    # Which layers to freeze (no backward) during training.
            'use_batchnorm': False,                 # If true, use batch norm for all newly added layers.
            'num_classes': 2,
            'background_label_id': 0,
            'neg_pos_ratio': 3.0,
            'num_test_image': num_test_image,
    }

    # train/test/deploy.prototxt
    for phase in ['train', 'test', 'deploy']:
        proto_file = os.path.join(job_dir, '{}.prototxt'.format(phase))
        with open(proto_file, 'w') as f:
            print(Net.create_net(phase), file=f)

    # solver.prototxt
    Solver.Params['train_net'] = os.path.join(job_dir, 'train.prototxt')
    Solver.Params['test_net'] = [os.path.join(job_dir, 'test.prototxt')]
    Solver.Params['snapshot_prefix'] = os.path.join(job_dir, job_name)

    
    Solver.Params['base_lr'] = Net.get_baselr() 
    Solver.Params['test_iter'] = [Net.get_testiter()] 
    solver_file = os.path.join(job_dir, 'solver.prototxt')
    with open(solver_file, 'w') as f:
        print(Solver.create(), file=f)

def debug(net, flag):
    net.blobs['label']
    if len(mbox_loss) == 0:
        mbox_loss.append(0.0)
    mbox_loss[-1] = mbox_loss[-1] + float(net.blobs['mbox_loss'].data)
    if flag is False:
        return 
    if len(x) == 0:
        x.append(debug_iter)
    else:
        x.append(x[-1] + debug_iter)
    mbox_loss[-1] = mbox_loss[-1] / debug_iter
    plt.clf()
    plt.title('loss per 1000')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.plot(x, mbox_loss)
    plt.savefig(save_file_path)
    mbox_loss.append(0.0)
    plt.clf()
    plt.hist(sample_occupy, 1000)
    plt.savefig('/home/work/qinhuan/git/vision-detector/sample_occupy.png')
    exit()

def training(device_id=0):
    caffe.set_mode_gpu()
    caffe.set_device(device_id)
    solver_file = os.path.join(job_dir, 'solver.prototxt')
    sgd_solver = caffe.get_solver(solver_file)
    
    if pretrain_model is not None:
        print('Finetune from {}.'.format(pretrain_model)) 
        sgd_solver.net.copy_from(pretrain_model)

    for i in range(Solver.Params['max_iter'] + 1):
        sgd_solver.step(1)
        net = sgd_solver.net
        #for layer_name, blob in net.blobs.iteritems():
        #    print (layer_name + '\t' + blob)
        for label in net.blobs['label'].data[0][0]:
            sample_occupy.append(math.sqrt((label[5]-label[3]) * (label[6]-label[4])))
            print (int(label[0]), label[1], label[2], label[3], label[4], label[5]-label[3], label[6]-label[4])
        exit()
        #import pdb;pdb.set_trace()
        if debug_iter != -1:
            if (i + 1) % debug_iter == 0:
                debug(sgd_solver.net, True)
            else:
                debug(sgd_solver.net, False)

if __name__ == '__main__':
    create_jobs()
    training(3)
