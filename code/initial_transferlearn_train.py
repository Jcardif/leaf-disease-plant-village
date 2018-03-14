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
from resnet18_run import share_parser, train_distilled_model
from initial_transferlearn_models import resnet18Base, resnet34Base, resnet50Base, resnet101Base, resnet152Base, resnet18t, resnet34t, resnet50t, resnet101t, resnet152t
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
def save_checkpoint(state, model_path, best_path, is_best):
    torch.save(state, model_path)
    if is_best:
        shutil.copy2(model_path, best_path)

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
                leaf_resnet_train_distill(logits_path=''
                         , root_dir = args.dataroot
                         , data_separate = args.dataseparate
                         , transf_in_dir = 'bottleneck_tensors_labels/'+args.arch
			 , distill_for_model='resnetfamily')
                , batch_size=args.batch_size
                , shuffle=True, **kwargs)

    print('traing data size: %d' % leaf_train_loader.dataset.__len__())

    model = models.__dict__[args.arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    if args.arch == 'resnet18':
        transfer_model = resnet18t()
    elif args.arch == 'resnet34':
        transfer_model = resnet34t()
    elif args.arch == 'resnet50':
        transfer_model = resnet50t()
    elif args.arch == 'resnet101':
        transfer_model = resnet101t()
    elif args.arch == 'resnet152':
        transfer_model = resnet152t()

    if args.cuda:
        transfer_model = torch.nn.DataParallel(transfer_model).cuda()
    transfer_model.eval()
    optimizer = optim.Adam(transfer_model.parameters(), lr=args.rlr, betas=(args.beta1, 0.999), weight_decay=args.wd)
    CE_loss = nn.CrossEntropyLoss()

    if args.cuda:
        CE_loss = CE_loss.cuda()
    best_loss = 10000.0
    model_dir = os.path.join(args.saveroot, 'initial_tl', args.arch)
    os.makedirs(model_dir, exist_ok=True)
    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        transfer_model.train(True)
        model_path = os.path.join(model_dir, data_separate+'_'+str(epoch)+'.pth.tar')
        best_path = os.path.join(model_dir, data_separate+'_best.pth.tar')
        epoch_loss = train_distilled_model(leaf_train_loader, transfer_model , optimizer, CE_loss, None, None, None, None, epoch, args.epochs, T=1.0, batch_size=args.batch_size, cuda=args.cuda, is_soft_target=args.softlogits, is_distill=args.distill)
        
        transfer_model.eval()
        
        is_best = epoch_loss <= best_loss
        best_loss = min(epoch_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': transfer_model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, model_path, best_path, is_best)

if __name__=='__main__':
    parser = share_parser()
    args = parser.parse_args()
    for arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']: #
        args.arch = arch
        main(args)

