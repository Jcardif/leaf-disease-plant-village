import torch

from PIL import Image
import os

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def make_dataset(text_file_path):
    images = []
    with open(text_file_path, 'r') as f:
        for l in f:
            line = l.split('\t')
            images.append((line[0],int(line[1].split('\n')[0])))
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(torch.utils.data.Dataset):

    def __init__(self, root, data_type='', transform=None, target_transform=None
                 , data_separate=''
                 #, valid_perc=0.1, test_perc=0.7
                 , loader=default_loader):
        assert(data_type != '')
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        if data_type=='train':
            self.tensors_labels = make_dataset(os.path.join(root, data_separate, 'train.txt'))
        elif data_type=='valid':
            self.tensors_labels = make_dataset(os.path.join(root, data_separate, 'valid.txt'))
        else:
            valid_imgs = make_dataset(os.path.join(root, data_separate, 'valid.txt'))
            imgs = make_dataset(os.path.join(root, data_separate, 'test.txt'))
            self.tensors_labels = imgs + valid_imgs

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.tensors_labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, path, target


    def __len__(self):
        return len(self.tensors_labels)
