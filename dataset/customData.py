'''
For CelebA-Spoof
'''
from torch.utils.data import Dataset
from imutils import paths
from PIL import Image 
import numpy as np
import torch
import random
import os

def pil_loader(path):    # 一般採用pil_loader函式。
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

def load_mat(path):
    mat = np.loadtxt(path)
    return torch.tensor(mat,dtype=torch.float32)

class customData(Dataset):
    def __init__(self, img_path, txt_path, phase = '',data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            random.Random(4).shuffle(lines)
            # self.img_name = [os.path.join(img_path, (line.strip().split(' ')[0][11:])) for line in lines]
            self.img_name = []
            self.img_label = []
            lengthOfTrain = int(len(lines)*0.8)
            if phase=='train':
                for line in lines:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    label = int(line.strip().split(' ')[-1])
                    if os.path.isfile(path):
                        self.img_name.append(path)
                        self.img_label.append(label)
                    else:
                        print(path)
            elif phase=='test':
                for line in lines:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][10:]))
                    if os.path.isfile(path):
                        if "live" in path:# and getreal:
                            label = 0
                        else:#if not getreal:
                            label = 1
                        self.img_name.append(path)
                        self.img_label.append(label)

            elif phase=='val':
                for line in lines[lengthOfTrain:]:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    label = int(line.strip().split(' ')[-1])
                    if os.path.isfile(path):
                        self.img_name.append(path)
                        self.img_label.append(label)
                    else:
                        print(path)
            elif phase=="ssl":
                for line in lines:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    label = int(line.strip().split(' ')[-1])
                    if os.path.isfile(path):
                        self.img_name.append(path)
                        self.img_label.append(label)
                    else:
                        print(path)

            # self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.phase = phase
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        try:
            img_name = self.img_name[item]
            label = self.img_label[item]
            img = self.loader(img_name)

            if self.data_transforms is not None:
                try:
                    img = self.data_transforms(img)
                except:
                    print("Cannot transform image: {}".format(img_name))
            return img, label
        except:
            print(item)
