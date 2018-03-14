import torch
import os
def make_dataset(text_file_path):
    images = []
    with open(text_file_path, 'r') as f:
        for l in f:
            line = l.split('\t')
            images.append((line[0],int(line[1].split('\n')[0])))
    return images

class leaf_resnet_eval_layer_tensor(torch.utils.data.Dataset):
    
    def __init__(self, data_type='', data_separate=''
                 , root_dir = '', transf_in_dir=''):
        assert(data_type != '')
        self.root_dir = root_dir
        self.data_sep = data_separate
        self.transf_in_dir = transf_in_dir
        file_dir = os.path.expanduser(root_dir)
        text_file_dir = os.path.expanduser(os.path.join(root_dir, self.data_sep))
        if data_type=='train': 
            self.tensors_labels = make_dataset(os.path.join(text_file_dir, 'train.txt'))
        elif data_type =='valid':
            self.tensors_labels = make_dataset(os.path.join(text_file_dir, 'valid.txt'))
        else:
            valid_tensors_labels = make_dataset(os.path.join(text_file_dir, 'valid.txt'))
            tensors_labels = make_dataset(os.path.join(text_file_dir, 'test.txt'))
            self.tensors_labels = tensors_labels+valid_tensors_labels
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.tensors_labels[index]
        imgsubpathlist = img_path[:-4].split('/')
        path = os.path.join(self.root_dir, self.transf_in_dir, imgsubpathlist[-2], imgsubpathlist[-1])+'.pt'
        layer_tensors, target = torch.load(path)
        
        return layer_tensors, path, target

    def __len__(self):
        return len(self.tensors_labels)
