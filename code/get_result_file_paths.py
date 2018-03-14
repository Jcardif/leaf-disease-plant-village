import os
def get_result_file_paths(epoch, lr, wd, datasetname, data_separate, archname, data_type, has_weights, logits_dir, model_dir, is_soft_target, is_distill='False'):
    """
    return the saved files' path
    args:
      epoch: epoch number [0,29]
      lr: learning rate
      wd: weight decay
      datasetname: 'leaf'
      archname: architecture name, e.g. 'resnet101'
      datatye: 'train' or 'test'
      has_weights: whether the loss function has weights, "True" or "False", string
      logits_dir: directory which contains logits file
      model_dir: direcotry which contains model parameter file
      is_soft_target: bool, whether the loss function uses the newly defined soft logits
      is_distill: bool, whether the loss function uses distilled logits
    """
    saved_logits_filepath = None
    if has_weights=='True':
        save_logits_per_epoch_suffix = '_lr_'+str(lr)+'_wd_'+str(wd)+'_with_weights'
        save_logits_per_epoch_suffix2 = '_lr_'+str(2e-05)+'_wd_'+str(wd)+'_with_weights'
    else:
        save_logits_per_epoch_suffix = '_lr_'+str(lr)+'_wd_'+str(wd)+'_no_weights'
        save_logits_per_epoch_suffix2 = '_lr_'+str(2e-05)+'_wd_'+str(wd)+'_no_weights'
    if archname == 'resnet18':
        if has_weights=='True':
            save_model_per_epoch_suffix = data_separate+'_checkpoint'+'_lr_'+str(lr)+'_wd_'+str(wd)+'_with_weights'+'.pth.tar' 
        else:
            save_model_per_epoch_suffix = data_separate+'_checkpoint'+'_lr_'+str(lr)+'_wd_'+str(wd)+'_no_weights'+'.pth.tar' 
        if is_soft_target == 'True' and is_distill=='False':
            model_path = os.path.join(model_dir, datasetname+'_'+archname+'_'+str(epoch)+'_distilled_'+save_model_per_epoch_suffix)
            if data_type == 'train':#todo change name according to this
                preds_labels_file = os.path.join(logits_dir, 'train_from_soft_logits_'+datasetname+'_'+archname+'_'+str(epoch) + '_' + data_separate+save_logits_per_epoch_suffix + '.pt')    
            elif data_type == 'test':
                preds_labels_file = os.path.join(logits_dir, 'test_from_soft_logits_'+datasetname+'_'+archname+'_'+str(epoch) + '_' + data_separate+save_logits_per_epoch_suffix + '.pt')
            
        elif is_soft_target=='False' and is_distill=='True':
            model_path = os.path.join(model_dir, datasetname+'_'+archname+'_'+str(epoch)+'_realdistilled_'+save_model_per_epoch_suffix)
            if data_type == 'train':#todo change name according to this
                preds_labels_file = os.path.join(logits_dir, 'train_from_realdistill_logits_'+datasetname+'_'+archname+'_'+str(epoch) + '_' + data_separate+save_logits_per_epoch_suffix + '.pt')
            elif data_type == 'test':
                preds_labels_file = os.path.join(logits_dir, 'test_from_realdistill_logits_'+datasetname+'_'+archname+'_'+str(epoch) + '_' + data_separate+save_logits_per_epoch_suffix + '.pt')
        elif is_soft_target=='False' and is_distill=='False':
            model_path = os.path.join(model_dir, datasetname+'_'+archname+'_'+str(epoch)+'_'+save_model_per_epoch_suffix)
            if data_type == 'train':
                preds_labels_file = os.path.join(logits_dir, 'train_from_logits_'+datasetname+'_'+archname+'_'+str(epoch) + '_' + data_separate+save_logits_per_epoch_suffix + '.pt')
            elif data_type == 'test':
                preds_labels_file = os.path.join(logits_dir, 'test_from_logits_'+datasetname+'_'+archname+'_'+str(epoch) + '_' + data_separate+save_logits_per_epoch_suffix + '.pt')
    elif archname == 'resnet101':
        if has_weights=='True':
            save_model_per_epoch_suffix = data_separate+'_cnn_layer4D_checkpoint'+'_lr_'+str(lr)+'_wd_'+str(wd)+'_with_weights'+'.pth.tar' 
            save_model_per_epoch_suffix2 = data_separate+'_cnn_layer4D_checkpoint'+'_lr_'+str(2e-05)+'_wd_'+str(wd)+'_with_weights'+'.pth.tar' 
        else:
            save_model_per_epoch_suffix = data_separate+'_cnn_layer4D_checkpoint'+'_lr_'+str(lr)+'_wd_'+str(wd)+'_no_weights'+'.pth.tar'
            save_model_per_epoch_suffix2 = data_separate+'_cnn_layer4D_checkpoint'+'_lr_'+str(2e-05)+'_wd_'+str(wd)+'_no_weights'+'.pth.tar'                
        if is_soft_target == 'True' and is_distill=='False':
            model_path = os.path.join(model_dir, datasetname+'_'+archname+'_'+str(epoch)+'_distilled_'+save_model_per_epoch_suffix)
            if data_type == 'train':#todo change name according to this
                preds_labels_file = os.path.join(logits_dir, 'train_from_bottleneck_soft_logits_'+datasetname+'_'+archname+'_'+str(epoch)+'_'+data_separate+'_cnn_layer4D_checkpoint'+save_logits_per_epoch_suffix + '.pt')
            elif data_type == 'test':
                preds_labels_file = os.path.join(logits_dir, 'test_from_bottleneck_soft_logits_'+datasetname+'_'+archname+'_'+str(epoch)+'_'+data_separate+'_cnn_layer4D_checkpoint'+save_logits_per_epoch_suffix + '.pt')
        elif is_soft_target == 'TruefromMean' and is_distill=='False':
            model_path = os.path.join(model_dir, datasetname+'_'+archname+'_'+str(epoch)+'_distilled_'+save_model_per_epoch_suffix2)
            if data_type == 'train':#todo change name according to this
                preds_labels_file = os.path.join(logits_dir, 'train_from_bottleneck_soft_logits'+'_'+datasetname+'_'+archname+'_'+str(epoch)+'_'+data_separate+'_cnn_layer4D_checkpoint'+save_logits_per_epoch_suffix2+'.pt')
            elif data_type == 'test':
                preds_labels_file = os.path.join(logits_dir, 'test_from_bottleneck_soft_logits'+'_'+datasetname+'_'+archname+'_'+str(epoch)+'_'+data_separate+'_cnn_layer4D_checkpoint'+save_logits_per_epoch_suffix2+'.pt')
        elif is_soft_target=='False' and is_distill=='True':
            model_path = os.path.join(model_dir, datasetname+'_'+archname+'_'+str(epoch)+'_realdistilled_'+save_model_per_epoch_suffix)
            if data_type == 'train':#todo change name according to this
                preds_labels_file = os.path.join(logits_dir, 'train_from_bottleneck_realdistill_logits_'+datasetname+'_'+archname+'_'+str(epoch) + '_' + data_separate+save_logits_per_epoch_suffix + '.pt')
            elif data_type == 'test':
                preds_labels_file = os.path.join(logits_dir, 'test_from_bottleneck_realdistill_logits_'+datasetname+'_'+archname+'_'+str(epoch) + '_' + data_separate+save_logits_per_epoch_suffix + '.pt')
        elif is_soft_target=='False' and is_distill=='False':
            model_path = os.path.join(model_dir, datasetname+'_'+archname+'_'+str(epoch)+'_'+save_model_per_epoch_suffix)
            if data_type == 'train':
                preds_labels_file = os.path.join(logits_dir, 'train_from_bottleneck_logits_'+datasetname+'_'+archname+'_'+str(epoch)+'_'+data_separate+'_cnn_layer4D_checkpoint'+save_logits_per_epoch_suffix + '.pt')
            elif data_type == 'test':
                preds_labels_file = os.path.join(logits_dir, 'test_from_bottleneck_logits_'+datasetname+'_'+archname+'_'+str(epoch)+'_'+data_separate+'_cnn_layer4D_checkpoint'+save_logits_per_epoch_suffix + '.pt')
    return model_path, preds_labels_file

def get_saved_logits_fn(logits_dir, datasetname, archname, data_sep, has_weights, lr, wd):
    if has_weights=='True':
        save_logits_per_epoch_suffix = '_lr_'+str(lr)+'_wd_'+str(wd)+'_with_weights'
    else:
        save_logits_per_epoch_suffix = '_lr_'+str(lr)+'_wd_'+str(wd)+'_no_weights'
    if archname=='resnet18':
        saved_logits_filepath = os.path.join(logits_dir, 'train_from_distilled_logits_'+datasetname+'_'+archname+'_'+ data_sep+save_logits_per_epoch_suffix + '.pt')
    elif archname=='resnet101':
        saved_logits_filepath = os.path.join(logits_dir, 'train_from_bottleneck_realdistill_logits_'+datasetname+'_'+archname+'_'+ data_sep+save_logits_per_epoch_suffix + '.pt')
    return saved_logits_filepath
