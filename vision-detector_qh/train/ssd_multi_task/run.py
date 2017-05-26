from __future__ import print_function
#!/usr/bin/env python
# coding=utf-8
import os, sys
import solver as Solver
import caffe 

resize_height = 128
resize_width = 512

data_dir = '/home/work/wwl/data/kitti/data_object_image_2/training'
train_lmdb = os.path.join(data_dir, 'train_lmdb')
test_lmdb = os.path.join(data_dir, 'test_lmdb')
label_map_file = 'data/KITTI/labelmap_kitti.prototxt'
name_size_file = 'data/KITTI/test_name_size.txt'
num_test_image = len([l for l in open(name_size_file, 'r').readlines()]) 

data_dir_lisa = '/home/work/wwl/data/LISA/lmdb'
train_lmdb_lisa = os.path.join(data_dir_lisa, 'train_lmdb')
test_lmdb_lisa = os.path.join(data_dir_lisa, 'test_lmdb')
label_map_file_lisa = 'data/LISA/labelmap_lisa.prototxt'
name_size_file_lisa = 'data/LISA/test_name_size.txt'
num_test_image_lisa = len([l for l in open(name_size_file_lisa, 'r').readlines()]) 

pretrain_model = 'data/vgg16_pretrain_imagenet_iter_350000.caffemodel' 
debug_iter = -1

job_name = 'kitti_{}x{}_vgg16-channel16-nopooling12'.format(resize_height, resize_width)
job_dir = 'jobs/ssd/{}/'.format(job_name)

batch_size = 8 

if not os.path.exists(job_dir):
    os.makedirs(job_dir)

def create_jobs():
    import ssd_vggnopool12 as Net 
    Net.Params = {
            'model_name': job_name,    
            'resize_height': resize_height,
            'resize_width': resize_width,
            'test_batch_size': batch_size,
            'batch_size_per_device': batch_size,
            'freeze_layers': [],                    # Which layers to freeze (no backward) during training.
            'use_batchnorm': False,                 # If true, use batch norm for all newly added layers.
            'neg_pos_ratio': 3.0,
            'train_lmdb': train_lmdb,
            'test_lmdb': test_lmdb,
            'label_map_file': label_map_file,
            'name_size_file': name_size_file,
            'output_result_dir': os.path.join(job_dir, 'output'),
            'num_classes': 4,
            'background_label_id': 0,
            'num_test_image': num_test_image,
            'LISA': {
                'train_lmdb': train_lmdb_lisa,
                'test_lmdb': test_lmdb_lisa,
                'label_map_file': label_map_file_lisa,
                'name_size_file': name_size_file_lisa,
                'output_result_dir': os.path.join(job_dir, 'output'),
                'test_batch_size': batch_size,
                'batch_size_per_device': batch_size,
                'num_classes': 8,
                'background_label_id': 7,
                'num_test_image': num_test_image_lisa,
                'loss_weight': 0.2,
                },
            'KITTI': {
                'train_lmdb': train_lmdb,
                'test_lmdb': test_lmdb,
                'label_map_file': label_map_file,
                'name_size_file': name_size_file,
                'output_result_dir': os.path.join(job_dir, 'output'),
                'test_batch_size': batch_size,
                'batch_size_per_device': batch_size,
                'num_classes': 4,
                'background_label_id': 0,
                'num_test_image': num_test_image,
                'loss_weight': 1.0,
                }
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

def debug():
    pass

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
        
        if debug_iter != -1 and (i + 1) % debug_iter == 0:
            debug(sgd_solver.net)

if __name__ == '__main__':
    create_jobs()
    training()
