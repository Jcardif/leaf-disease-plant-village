import torch
import os
from ImageFolderData import is_image_file, make_dataset, pil_loader, accimage_loader, default_loader
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

# datasetloader for distilling
class leaf_resnet_train_distill(torch.utils.data.Dataset):
    
    def __init__(self, root_dir = '', logits_path='', data_separate='', transf_in_dir=''
                 , transform=None, target_transform=None, distill_for_model='resnet18'):
        self.root_dir = root_dir
        self.distill_for_model = distill_for_model
        self.transform = transform
        self.target_transform = target_transform
        self.logits_path = logits_path
        self.transf_in_dir = transf_in_dir
        if distill_for_model=='resnet18':
            self.loader = default_loader
        else:
            self.loader = torch.load
        self.data_separate = data_separate
        file_dir = os.path.expanduser(root_dir)
        text_file_dir = os.path.expanduser(os.path.join(root_dir, self.data_separate))
        
        self.tensors_labels = make_dataset(os.path.join(text_file_dir, 'train.txt'))
        if os.path.exists(self.logits_path):
            self.logits = torch.load(self.logits_path)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.tensors_labels[index]
        if self.distill_for_model != 'resnet18':
            imgsubpathlist = img_path[:-4].split('/')
            img_path = os.path.join(self.root_dir, self.transf_in_dir, imgsubpathlist[-2], imgsubpathlist[-1])+'.pt'
        if os.path.exists(self.logits_path):
            logit = self.logits[index]
        else:
            logit = 0 #fake
        img = self.loader(img_path)
        if self.distill_for_model != 'resnet18':#(tensor, target)
            img = img[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, img_path, target, logit

    def __len__(self):
        return len(self.tensors_labels)
