import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

@dataclass
class Hyperparameters:
    LEARNING_RATE : float = 1e-4
    BATCH_SIZE : int = 64
    # Do not change without messing with the models
    IMG_SIZE : int = 64
    # 1: monochrome
    # 3: RGB
    CHANNELS_IMG : int = 3
    Z_DIM : int = 100
    NUM_EPOCHS : int = 5
    FEATURES_DISC : int = 64
    FEATURES_GEN : int = 64
    CRITIC_ITERATIONS : int = 5
    LAMBDA_GP : int = 10
    NOISE_DIM : int = 100
    NUM_CLASSES : int = 40
    GEN_EMBEDDING : int = 100
    # No momentum
    ADAM_BETAS : tuple = (0.0, 0.9)

def gradient_penalty(critic, labels, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    eps = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_img = real* eps + fake * (1 - eps)

    # calculate critic scores
    scores = critic(interpolated_img, labels)
    gradient = torch.autograd.grad(
        inputs=interpolated_img,
        outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)

    return gradient_penalty

class CelebA(Dataset):
    def __init__(self,
                 root,
                 image_dir='dataset/img/img_align_celeba',
                 anno_file='dataset/labels/Anno/list_attr_celeba.txt',
                 transform=None):
        super().__init__()
        assert os.path.isdir(root), 'Dataset dir does not exist'
        self.root = root
        self.image_dir = image_dir
        self.anno_file = anno_file
        self.transform = transform

        self.data, self.labels = self.parse_anno_file()

    def parse_anno_file(self):
        if os.path.exists(self.anno_file):
            anno_path = self.anno_file
        elif os.path.exists(os.path.join(self.root, self.anno_file)):
            anno_path = os.path.join(self.root, self.anno_file)
        else:
            raise FileNotFoundError('Annotation file of dataset not exists: {}'.format(self.anno_file))
        data=[]
        attrs = None
        num_img = 0
        with open(anno_path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                #first line contains data number
                if idx == 0:
                    num_img = int(line)
                #second line contains class labels
                elif idx == 1:
                    attrs  = line.split(' ')
                else:
                    elements = [e for e in line.split(' ') if e]
                    img_path = os.path.join(self.root, self.image_dir, elements[0])
                    image_attr = elements[1:]
                    if not os.path.exists(img_path) or not self.__is_image(img_path):
                        continue
                    # 0 for -1 and 1 for 1
                    image_onehot = [0 if int(attr) == -1 else 1 for attr in image_attr]
                    data.append({
                        'path': img_path,
                        'attr': image_onehot
                    })
        print('[Dataset] CelebA: Expect {} images with {} attributes.'.format(num_img, len(attrs)))
        print('[Dataset] CelebA: Find {} images with {} attributes.'.format(len(data), len(data[-1]['attr'])))

        return data, attrs

    def __getitem__(self, index):
        data = self.data[index]
        image_path = data['path']
        image_attr = data['attr']

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        image_attr = torch.tensor(image_attr)
        return image, image_attr

    def __len__(self):
        return len(self.data)

    def __is_image(self, file_path):
        image_extensions = ['.png', '.jpg', '.jpeg']
        basename = os.path.basename(file_path)
        _, extension = os.path.splitext(basename)
        return extension.lower() in image_extensions
