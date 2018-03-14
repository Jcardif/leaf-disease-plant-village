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
from PIL import Image
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from ResNet18Model import ResNet18FC
from DataWDistilledLogits import leaf_resnet_train_distill
from get_result_file_paths import get_saved_logits_fn, get_result_file_paths
from ProbTargetNLL import NLLProbTarget

def share_parser():
    
    model_names = sorted(name for name in models.__dict__ 
                         if name.islower() and not name.startswith("__") 
                         and callable(models.__dict__[name]))

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--dataset', default='leaf',help='dataset name')
    parser.add_argument('--dataroot',
                        default='/home/h/Downloads/plantvillage_deeplearning_paper_dataset'
                        ,help='path to dataset')
    parser.add_argument('--transferinputdir',
                        default='bottleneck_tensors_labels/5306seed_layer3'
                        ,help='sub dir of transfer model input dataset')
    parser.add_argument('--dataseparate',
                        default='10-90'
                        ,help='the sub directory which contains txt file of generated train, validate and test data filenames')
    parser.add_argument('--datasettype'
                        , default='train'
                        , choices=['train','valid', 'test']
                        , help='chose the data set type (train/test)')
    parser.add_argument('--saveroot',
                        default='/media/h/15210519917/resnet18'
                        ,help='path to checkpoint')
    parser.add_argument('--train-size', type=int, default=100, metavar='N',
                       help='input batch size for training (default: 100)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                       help='input batch size for training, should be smaller than --train-size (default: 50)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                       help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                       help='number of epochs to train (default: 30)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--num_channels', type=int, default=3
                        , help='number of input image channels')
    parser.add_argument('--has_weights', type=str, default='True'
                        , help='whether the loss has weights for every class ("True"/"False")')
    parser.add_argument('--num_hidden_neuron', type=int, default=64
                        , help='number of hidden neurons')
    parser.add_argument('--dr', type=float, default=0.5
                        , help='dropout rate, default=0.5')
    parser.add_argument('--wd', type=float, default=0.0005
                        , help='weight decay, default=0.0005')
    parser.add_argument('--eta', type=float, default=20.0
                        , help='coordinate the weights, default=20.0')
    parser.add_argument('--T', type=float, default=3.5
                        , help='temperature for softmax, for distillation of dark knowledge, default=3')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                       help='base model learning rate (default: 0.0002)')
    parser.add_argument('--rlr', type=float, default=0.003, metavar='LR',
                       help='last classify layer learning rate (default: 0.003)')
    parser.add_argument('--beta1', type=float, default=0.5
                        , help='beta1 for adam. default=0.5')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=5306, metavar='S',
                       help='random seed (default: 5306)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--distill', default='False', type=str, metavar='("True"/"False")',
                        help='whether using distillation ("True"/"False")')
    parser.add_argument('--softlogits', default='False', type=str, metavar='("TruefromRandom"/"TruefromMean"/"False")',
                        help='whether using soft logits as classification score ("TruefromRandom"/"TruefromMean"/"False")')
    return parser

# training
def train_distilled_model(datasetloader, stumodel, optimizer, CE_loss, Log_softmax, Softmax, NLLL_loss, Prob_target_NLL, epoch, tepochs, T, batch_size, cuda=True, is_soft_target='False', is_distill='False'): #eval_datasetloader,
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data
    for i, (real_data, _, targets, target_logits) in enumerate(datasetloader):
        if i == datasetloader.dataset.__len__() // batch_size:
            break
        if cuda:
            targets = targets.cuda(async=True)
            target_logits = target_logits.cuda(async=True)
        batch_size = len(targets)
        inputv = Variable(real_data)
        targetsv = Variable(targets)
        target_logitsv = Variable(target_logits)
        # forward
        # https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/13
        logits = stumodel(inputv)
        _, preds = torch.max(logits.data, 1) # return (max, max_indices)
        loss = CE_loss(logits, targetsv)
        if is_soft_target.find('True')!=-1:
            softlabel_loss = NLLL_loss(Log_softmax(logits/T),targetsv) #first try 
            loss += softlabel_loss*T*T
        elif is_distill == 'True':
            loss += Prob_target_NLL(Softmax(target_logitsv/T), Log_softmax(logits/T))*T*T #second try
        # backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.data[0]*batch_size # total loss
        running_corrects += torch.sum(preds == targets)

        if (i % 100) == 0:
            print('[%d/%d][%d/%d] Loss: %.4f'
                  % (epoch, tepochs-1, i, datasetloader.dataset.__len__() // batch_size, loss))  
    # do checkpointing every epoch
    print('[%d/%d][%d/%d] Loss: %.4f'
          % (epoch, tepochs-1, i, datasetloader.dataset.__len__() // batch_size, loss))

    train_dataset_size = len(datasetloader.dataset) // batch_size * batch_size
    epoch_loss = running_loss / train_dataset_size
    epoch_acc = running_corrects / train_dataset_size
    print('[%d/%d] Average statistics on the whole training set; Loss: %.4f, Acc: %.4f (%d/%d)'
          % (epoch, tepochs-1, epoch_loss, epoch_acc, running_corrects, train_dataset_size))
    return epoch_loss

def save_checkpoint(state, model_path):
    torch.save(state, model_path)

def adjust_learning_rate_resnet18(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.3 ** (epoch // 7))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_resnet101(optimizer, epoch, lr, rlr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.3 ** (epoch // 7))
    rlr = rlr * (0.3 ** (epoch // 11))
    for i,param_group in enumerate(optimizer.param_groups):
        if i == 1 and rlr >= 0.00001:
            param_group['lr'] = rlr
        elif i != 1 and lr >= 0.00001:
            param_group['lr'] = lr

def choose_train_hyperparams_acc_arch(model_arch, data_separate, is_distill, saved_logits_filepath, batch_size, kwargs):
    """
    For different model architecture (ResNet18, ResNet101):
    return:
    dataset, model, optimizer
    """
    # original model
    largemodel = models.__dict__[args.arch](pretrained=True)
    largemodel.eval()
    if model_arch == 'resnet18':
        leaf_loader = torch.utils.data.DataLoader(
            leaf_resnet_train_distill(args.dataroot
                        , logits_path=saved_logits_filepath, data_separate=data_separate
                        ,transform=transforms.Compose([
                            transforms.Scale(224),
                            #transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        ,distill_for_model=model_arch
                       ), 
            batch_size=batch_size, shuffle=True, **kwargs)
        themodel = ResNet18FC(largemodel)
        # optimizer
        optimizer = optim.Adam(themodel.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.wd)
        
    elif model_arch == 'resnet101':
        leaf_loader = torch.utils.data.DataLoader(
            leaf_resnet_train_distill(args.dataroot
                        , logits_path=saved_logits_filepath, data_separate=data_separate
                        ,distill_for_model=model_arch
                       ), 
            batch_size=batch_size, shuffle=True, **kwargs)
        themodel = NewModelFromResNet101BottleNeck(largemodel)
        # optimizer
        optimizer = optim.Adam(
            [
                {'params': themodel.module.layer4.parameters()},
                #{'params': themodel.module.avgpool.parameters()}, # no parameters
                {'params': themodel.module.fc.parameters(), 'lr': args.rlr}
            ],lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.wd)
    if args.cuda:
        themodel = torch.nn.DataParallel(themodel).cuda()
    themodel.eval()
    return themodel, leaf_loader, optimizer

    
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

def main(args,best_loss):
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
    
    # original model
    saved_logits_filepath = os.path.join(args.dataroot, 'train_results_avg_models_logits_for_distill_'+data_separate+'.pt')
    themodel, leaf_loader, optimizer = choose_train_hyperparams_acc_arch(args.arch, data_separate, args.distill, saved_logits_filepath, args.batch_size, kwargs)
    print('traing data size: %d' % leaf_loader.dataset.__len__())

    # get the weight for the loss on each class
    train_class_nums = data_class_num_dict(os.path.join(args.dataroot, data_separate, 'train.txt'))
    class_num_max = max(train_class_nums.values())
    train_class_weights = {}
    for train_class_idx, train_class_num in train_class_nums.items():
        train_class_weights[train_class_idx] = 1 + (class_num_max - train_class_num)/(args.eta * class_num_max)
    print(train_class_weights)
    train_class_weights_tensor = []
    for train_class_num in train_class_weights.values():
        train_class_weights_tensor.append(train_class_num)
    train_class_weights_tensor = torch.FloatTensor(train_class_weights_tensor)
   
    # loss
    Log_softmax = nn.LogSoftmax(dim=1)
    Prob_target_NLL = NLLProbTarget()
    Softmax = nn.Softmax(dim=1)
    if args.has_weights == 'True':
        CE_loss = nn.CrossEntropyLoss(weight=train_class_weights_tensor)
        NLLL_loss = nn.NLLLoss(weight=train_class_weights_tensor)
    else:
        CE_loss = nn.CrossEntropyLoss()
        NLLL_loss = nn.NLLLoss()
    if args.cuda:
        Prob_target_NLL = Prob_target_NLL.cuda()
        CE_loss = CE_loss.cuda()
        Log_softmax = Log_softmax.cuda()
        NLLL_loss = NLLL_loss.cuda()
        Softmax = Softmax.cuda()
    
    
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            themodel.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            best_loss = checkpoint['best_loss']
            assert best_loss is not None
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # training begin
    since = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model_path, preds_labels_file = get_result_file_paths(epoch, args.lr, args.wd, args.dataset, data_separate, args.arch, args.datasettype, args.has_weights, logits_dir=args.saveroot, model_dir=args.saveroot, is_soft_target=args.softlogits, is_distill=args.distill)
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        epoch_start_time = time.time()
        if args.arch == 'resnet18':
            adjust_learning_rate_resnet18(optimizer, epoch, args.lr)
        elif args.arch == 'resnet101':
            adjust_learning_rate_resnet101(optimizer, epoch, args.lr, args.rlr)
        themodel.train(True)
        
        epoch_loss = train_distilled_model(leaf_loader, themodel , optimizer, CE_loss, Log_softmax, Softmax, NLLL_loss, Prob_target_NLL, epoch, args.epochs, T=args.T, batch_size=args.batch_size, cuda=args.cuda, is_soft_target=args.softlogits, is_distill=args.distill)
        
        themodel.eval()
        
        #is_best = epoch_loss <= best_loss
        best_loss = min(epoch_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': themodel.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, model_path)
        
        epoch_time = time.time() - epoch_start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
if __name__=='__main__':
    best_loss = 10000.0
    parser = share_parser()
    args = parser.parse_args()
    assert args.train_size >= args.batch_size
    main(args, best_loss)
