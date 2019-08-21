from io import BytesIO
import os

import numpy as np
from PIL import Image
import torch

class CIFAR10C(torch.utils.data.Dataset):
    """`CIFAR10-C`_ Dataset.
    Args:
        data_dir (string): Directory containing .npy files
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        corruption_name (str): Name of the corruption
        corruption_level (int): Level of the corruption (1, 2, 3, 4, 5)
        apply_compress (bool): Whether to apply JPEG compression
   """
    def __init__(self, data_dir, transform=None, corruption_name=None, corruption_level=None,
                 apply_compress=False):
        super(CIFAR10C, self).__init__()
        self.transform = transform
        self.corruption_name = corruption_name
        self.corruption_level = int(corruption_level)
        self.apply_compress = apply_compress
        
        self.data = np.load(os.path.join(data_dir, corruption_name + '.npy'))
        self.targets = np.load(os.path.join(data_dir, 'labels.npy')).astype(int)

        self.data = self.data[10000 * (self.corruption_level - 1): 10000 * self.corruption_level]
        self.targets = self.targets[10000 * (self.corruption_level - 1): 10000 * self.corruption_level]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.apply_compress:
            f = BytesIO()
            img.save(f, format='JPEG', quality=85, optimize=True)
            f.seek(0)
            img = Image.open(f)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
