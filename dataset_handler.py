from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch

class DatasetHandler(Dataset):
    def __init__(self):
        self.ds = ImageFolder('dataset/img_align_celeba')

file = open('dataset/Anno/list_attr_celeba.txt').read().split('\n')
attr_names = file[1].split()
