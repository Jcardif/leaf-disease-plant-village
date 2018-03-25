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
from ImageFolderData import ImageFolder
from Resnet101Layer4DinputData import leaf_resnet_eval_layer_tensor
from get_result_file_paths import get_result_file_paths
from ResNet101TransferModel import NewModelFromResNet101BottleNeck
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from ResNet18Model import ResNet18FC
from resnet18_run import share_parser

def data_class_num_dict(datasettxtfile):
    labels_map = {}
    with open(datasettxtfile, 'r') as f:
        dataset = f.read().splitlines()
    for _entry in dataset:
        l = int(_entry[-2:])# maximum 38
        try:
            labels_map[l] += 1
        except:
            labels_map[l] = 1
    return labels_map


def predict_via_single_model(criterion, bottleneck_dataset_loader, retrained_model=None, preds_labels_file=None,all_logits=None, save_model=True, cuda=True, base_model=None):
    """
    If the there isn't a logits file of the model, this will create one
    Else, it will read the logits file. Then, it will calculate f1 related statistics.
    Also, we can use pure logits(batchsize, num_classes), to calculate the statistics.
    """
    running_corrects = 0
    running_loss = 0.0
    dataset_size = 0

    
    #wrong_list = []
    preds_list = []
    labels_list = []
    #if has_weights:
    #    criterion = nn.CrossEntropyLoss(weight=train_class_weights_tensor) # avg loss
    #else:
    #    criterion = nn.CrossEntropyLoss()
    if save_model==False or (save_model and preds_labels_file is not None and os.path.exists(preds_labels_file) == False):
        retrained_model.eval()
        logits_list = []
        
        if cuda:
            criterion = criterion.cuda()
            #retrained_model = torch.nn.DataParallel(retrained_model).cuda()
        for (bottleneck,_,targets) in bottleneck_dataset_loader:
            dataset_size += len(targets)
            inputv = Variable(bottleneck, volatile=True)
            if cuda:
                targets = targets.cuda(async=True)
            targetsv = Variable(targets, volatile=True)
            if base_model is None:
                logits = retrained_model(inputv)
            else:
                logits = retrained_model(base_model(inputv))
            loss = criterion(Variable(logits.data, volatile=True), targetsv)
            running_loss += loss.data[0]*len(targets)
            logits_list.append(logits.data.cpu())
            _, preds = torch.max(logits.data, 1)
            compare_list = (preds == targets)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(targets.cpu().numpy().tolist())
            running_corrects += torch.sum(compare_list)
            #temp_wrong_list = []
            #for i, out in enumerate(compare_list):
            #    if out == 0:
            #        temp_wrong_list.append(i)
            #wrong_list.append(temp_wrong_list)
        if save_model:
            print('creating %s' % preds_labels_file)
            torch.save(logits_list, preds_labels_file)
            print('create finished')
            return
    else:
        targets = torch.LongTensor([x[1] for x in bottleneck_dataset_loader.dataset.tensors_labels])
        dataset_size = len(targets)
        print('datasize: %d' % dataset_size)
        if all_logits is None:
            logits_list = torch.load(preds_labels_file)
            #print(logits_list)
            all_logits = torch.cat(logits_list, dim=0)
            print('reading %s' % preds_labels_file)
        print(all_logits.shape)
        print(targets.shape)
        loss = criterion(Variable(all_logits, volatile=True), Variable(targets, volatile=True))
        print(loss.data[0])
        running_loss += loss.data[0]*dataset_size
        _, preds = torch.max(all_logits, 1)
        compare_list = (preds == targets)
        #preds_list.extend(preds.cpu().numpy().tolist())
        #labels_list.extend(targets.cpu().numpy().tolist())
        preds_list = preds.cpu().numpy().tolist()
        labels_list = targets.cpu().numpy().tolist()
        running_corrects += torch.sum(compare_list)
    
    dataset_size = bottleneck_dataset_loader.dataset.__len__()
    acc = accuracy_score(labels_list, preds_list)#running_corrects / dataset_size
    f1 = f1_score(labels_list, preds_list,average='weighted')
    prec = precision_score(labels_list, preds_list,average='weighted')
    recall = recall_score(labels_list, preds_list,average='weighted')
    loss_per_img = running_loss/dataset_size
    print('Acc: {:.5f} [{:.0f}/{:.0f}], f1: {:.5f}, precision: {:.5f}, recall: {:.5f}, loss: {:.5f}'
          .format(acc, running_corrects, dataset_size, f1, prec, recall, loss_per_img))
    #print('Wrong labels')
    #print(' '.join('%3s' % classes[wrong_label] for wrong_label in wrong_labels))
    #print('Real labels')
    #print(' '.join('%3s' % classes[real_label] for real_label in real_labels))
    #return wrong_list, len(wrong_labels)
    return acc, f1, prec, recall, loss_per_img, all_logits

def main(args):
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

    # original model
    largemodel = models.__dict__[args.arch](pretrained=True)
    largemodel.eval()
    if args.arch == 'resnet18':
        # transfer learning model
        themodel = ResNet18FC(largemodel)
    else:
        themodel = NewModelFromResNet101BottleNeck(largemodel)
    if args.cuda:
        themodel = torch.nn.DataParallel(themodel).cuda()
    themodel.eval()
    # training begin
    #for is_soft_target in ['False']:#, 'TruefromRandom']:'TruefromMean'] :
    #for data_sep in ['10-90']:#,'20-80','40-60','50-50', '60-40','80-20']:
    is_soft_target = args.softlogits
    data_sep = args.dataseparate
    if os.path.exists(args.saveroot)==False:
        os.makedirs(args.saveroot, exist_ok=True)

    CE_loss = nn.CrossEntropyLoss()
    if args.cuda:
        CE_loss = CE_loss.cuda()
    if args.arch == 'resnet18':
        leaf_loader = torch.utils.data.DataLoader(
            ImageFolder(args.dataroot
                        , data_type=args.datasettype, data_separate=data_sep
                        ,transform=transforms.Compose([
                            transforms.Scale(224),
                            #transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                       ), 
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        leaf_loader = torch.utils.data.DataLoader(
            leaf_resnet_eval_layer_tensor(data_type=args.datasettype
                     , root_dir = args.dataroot
                     , data_separate = data_sep
                     , transf_in_dir = args.transferinputdir)
            , batch_size=args.test_batch_size
            , shuffle=False, **kwargs)
    print(leaf_loader.dataset.__len__())
    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        # resume from a checkpoint
        model_path, preds_labels_file = get_result_file_paths(epoch, args.lr, args.wd, args.dataset, data_sep, args.arch, args.datasettype, args.has_weights, logits_dir=args.saveroot, model_dir=args.saveroot, is_soft_target=is_soft_target, is_distill=args.distill)
        if os.path.exists(preds_labels_file) == False:
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
                checkpoint = torch.load(model_path)
                themodel.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(model_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(model_path))

        themodel.eval()

        predict_via_single_model( CE_loss, bottleneck_dataset_loader=leaf_loader, retrained_model= themodel, preds_labels_file=preds_labels_file, cuda=args.cuda) #, base_model=base_model

        # single stats to dataframe for drawing and disllation; todo

if __name__=='__main__':
    parser = share_parser()
    args = parser.parse_args()
    assert args.train_size >= args.batch_size
    main(args)
        
