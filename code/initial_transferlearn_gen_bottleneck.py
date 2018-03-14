from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import shutil
import sys
import random
import errno
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from numpy import set_printoptions
import matplotlib.pyplot as plt
import pandas as pd
from save_resnet18_test_scores import predict_via_single_model
from ImageFolderData import ImageFolder
from Resnet101Layer4DinputData import leaf_resnet_eval_layer_tensor
from DataWDistilledLogits import leaf_resnet_train_distill
from get_result_file_paths import get_saved_logits_fn, get_result_file_paths
from ResNet101TransferModel import NewModelFromResNet101BottleNeck
from resnet18_run import share_parser
from initial_transferlearn_models import resnet18Base, resnet34Base, resnet50Base, resnet101Base, resnet152Base, resnet18t, resnet34t, resnet50t, resnet101t, resnet152t
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def main(args):
    #https://stackoverflow.com/questions/41961949/google-oauth-inside-jupyter-notebook
    assert args.train_size >= args.batch_size
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    data_separate = args.dataseparate
    leaf_train_loader = torch.utils.data.DataLoader(
        ImageFolder(args.dataroot
                    , data_type='train', data_separate=data_separate
                    ,transform=transforms.Compose([
                        transforms.Scale(224),
                        #transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                   ), 
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    leaf_test_loader =  torch.utils.data.DataLoader(
        ImageFolder(args.dataroot
                    , data_type='test',data_separate=data_separate
                    ,transform=transforms.Compose([
                        transforms.Scale(224),
                        #transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                   ), 
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    print('traing data size: %d' % leaf_train_loader.dataset.__len__())
    print('test data size: %d' % leaf_test_loader.dataset.__len__())

    model = models.__dict__[args.arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    if args.arch == 'resnet18':
        bottleneck_model = resnet18Base(model)
    elif args.arch == 'resnet34':
        bottleneck_model = resnet34Base(model)
    elif args.arch == 'resnet50':
        bottleneck_model = resnet50Base(model)
    elif args.arch == 'resnet101':
        bottleneck_model = resnet101Base(model)
    elif args.arch == 'resnet152':
        bottleneck_model = resnet152Base(model)

    for param in bottleneck_model.parameters():
        print(param.requires_grad)
        #print(param.data, param.size())
        break
    #for param in bottleneck_model.parameters():
    #    param.requires_grad = False
    if args.cuda:
        bottleneck_model = torch.nn.DataParallel(bottleneck_model).cuda()
    bottleneck_model.eval()

    fo = open(os.path.join(args.dataroot,data_separate,'labels.txt'), 'r')
    classes = []
    for l in fo:
        classes.append(l.split('\n')[0])
    fo.close()

    save_layer_tensor(leaf_train_loader, bottleneck_model, classes, args.dataroot, 'bottleneck_tensors_labels'+'/'+args.arch)
    save_layer_tensor(leaf_test_loader, bottleneck_model, classes, args.dataroot, 'bottleneck_tensors_labels'+'/'+args.arch)

# save training/ validation/ tesing pt in one class dir
def save_layer_tensor(dataloader, model, classes, root_dir, transfer_data_dir):
    file_dir = os.path.expanduser(os.path.join(root_dir, transfer_data_dir))
    try:
        os.makedirs(file_dir, exist_ok=True)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    for label in classes:        
        try:
            os.mkdir(os.path.join(file_dir,label))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
    
    model.eval()
    for i, (batchdata, batchpaths, batchlabels) in enumerate(dataloader):
        tensor = model(Variable(batchdata, volatile=True))
        for j, path in enumerate(batchpaths):
            newsubpathlist = path[:-4].split('/')
            newwannapath = os.path.join(file_dir, newsubpathlist[-2], newsubpathlist[-1])+'.pt'
            if os.path.exists(newwannapath) == False:
                with open(newwannapath , 'wb') as f:
                    torch.save((tensor.data[j].cpu().numpy(), batchlabels[j]), f)
                print('create ' + newwannapath)

if __name__=='__main__':
    parser = share_parser()
    args = parser.parse_args()
    for arch in ['resnet18', 'resnet34','resnet50', 'resnet101','resnet152']: #
        args.arch = arch
        main(args)

