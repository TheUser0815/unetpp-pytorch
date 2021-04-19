import os
from PIL import Image, ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import os


def get_loader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



class FileCollector:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.images = []

    def add_data(self, img_dir, mask_dir, subdirs = True):
        self.__collect_image_names(img_dir, mask_dir, subdirs = subdirs)
        self.size = len(self.images)

    def __collect_image_names(self, img_path, mask_path, current = "", subdirs = True):
        path, folders, files=next(os.walk(os.path.join(self.base_dir, img_path, current)))

        #collect images in style (img_path, mask_path, file_path)
        for image in files:
            self.images.append((img_path, mask_path, os.path.join(current, image)))

        #traverse subfolders if requested
        if subdirs:
            for folder in folders:
                self.__collect_image_names(img_path, mask_path, os.path.join(current, folder), subdirs)



class ImageLoader:
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')



class DSet(data.Dataset, FileCollector, ImageLoader):
    def __init__(self, base_dir, img_size, class_channels, channel_wise_mask):
        FileCollector.__init__(self, base_dir)
        ImageLoader.__init__(self)
        self.img_size = img_size
        self.channel_wise_mask = channel_wise_mask
        self.class_channels = class_channels

        if class_channels > 2:
            self.mask_loader = self.__get_multichannel_mask()
        elif class_channels == 2:
            self.mask_loader = self.__get_dualchannel_mask()
        else:
            self.mask_loader = self.__get_monochannel_mask()

        self.re_init()

    def __get_multichannel_mask(self):
        if self.channel_wise_mask:
            def get_mask(img_data, seed):
                gt = []
                for i in range(self.class_channels):
                    gti = self.binary_loader(os.path.join(self.base_dir, img_data[1], str(i), img_data[2]))
                    gti = self.mask_transformer(gti, seed)
                    gt.append(gti)
                gt = torch.stack(gt, 1)[0]
                return gt
        else:
            def get_mask(img_data, seed):
                gt = self.binary_loader(os.path.join(self.base_dir, img_data[1], str(0), img_data[2]))
                gt = self.mask_transformer(gt, seed)[0]
                for i in range(1, self.class_channels):
                    gti = self.binary_loader(os.path.join(self.base_dir, img_data[1], str(i), img_data[2]))
                    gti = self.mask_transformer(gti, seed)[0]
                    gt = torch.round(gt)
                    gti *= i
                    gt += gti
                return gt

        return get_mask

    def __get_dualchannel_mask(self):
        get_mono_mask = self.__get_monochannel_mask()
        if self.channel_wise_mask:
            def get_mask(img_data, seed):
                mono = get_mono_mask(img_data, seed)
                second_ch = 1. - mono
                return torch.stack([mono, second_ch], 1)[0]
        else:
            get_mask = get_mono_mask

        return get_mask

    def __get_monochannel_mask(self):
        if self.channel_wise_mask:
            def get_mask(img_data, seed):
                gt = self.binary_loader(os.path.join(self.base_dir, img_data[1], img_data[2]))
                gt = self.mask_transformer(gt, seed)
                return gt
        else:
            def get_mask(img_data, seed):
                gt = self.binary_loader(os.path.join(self.base_dir, img_data[1], img_data[2]))
                gt = self.mask_transformer(gt, seed)[0]
                return gt
            
        return get_mask

    def getitem(self, index):
        img_data = self.images[index]

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        
        image = self.image_loader(os.path.join(self.base_dir, img_data[0], img_data[2]))
        image = self.transformer(image, seed)

        mask = self.mask_loader(img_data, seed)

        name = os.path.join(img_data[0], img_data[2]).replace("/", "_")

        return image, mask, name

    def next(self):
        image, mask, name = self.getitem(self.index)

        self.index += 1

        return image, mask, name

    def __next__(self):
        return self.next()

    def __iter__(self):
        self.re_init()
        return self

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return self.size

    def re_init(self):
        self.size = len(self.images)
        self.index = 0
        np.random.shuffle(self.images)

    def copy(self):
        cpy = DSet(self.base_dir, self.img_size, self.class_channels, self.channel_wise_mask)
        cpy.image_loader = self.image_loader
        cpy.images = self.images
        cpy.size = self.size
        cpy.mask_transformer = self.mask_transformer
        cpy.transformer = self.transformer
        return cpy

    def split_dset(self, val_percentage):
        val_count = int(len(self.images) * val_percentage)
        np.random.shuffle(self.images)

        val_set = self.copy()
        val_set.images = self.images[:val_count]
        val_set.size = len(val_set.images)

        self.images = self.images[val_count:]
        self.size = len(self.images)

        return self, val_set



class TrainDataset(DSet):
    def __init__(self, base_dir, img_size, augmentations, class_channels, channel_wise_mask, mask_size):
        super().__init__(base_dir, img_size, class_channels, channel_wise_mask)

        transformations = []
        if augmentations:
            print('Using RandomRotation, RandomFlip')
            transformations = [
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5)
            ]
            def transformer_creator(transform_actions):
                def transformer_function(x, seed):
                    random.seed(seed) # apply this seed to img tranfsorms
                    torch.manual_seed(seed) # needed for torchvision 0.7
                    return transform_actions(x)
                return transformer_function
        else:
            print('no augmentation')
            transformer_creator = lambda transform_actions: lambda x, seed: transform_actions(x)


        self.transform = transforms.Compose(
            transformations + 
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ]
        )
        self.transformer = transformer_creator(self.transform)
        
        if mask_size is None:
            self.mask_transformer = self.transformer
        else:
            self.mask_transform = transforms.Compose(
                transformations + 
                [
                    transforms.Resize((mask_size, mask_size)),
                    transforms.ToTensor()
                ]
            )
            self.mask_transformer = transformer_creator(self.mask_transform)
        


class MonochromeDataset(TrainDataset):
    def __init__(self, base_dir, img_size, augmentations=False, class_channels=1, channel_wise_mask=True, mask_size=None):
        super().__init__(base_dir, img_size, augmentations, class_channels, channel_wise_mask, mask_size)
        self.image_loader = self.binary_loader

    def __next__(self):
        return self.next()

    def __iter__(self):
        self.re_init()
        return self

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return self.size
    

class PolychromeDataset(TrainDataset):
    def __init__(self, base_dir, img_size, augmentations=False, class_channels=1, channel_wise_mask=True, mask_size=None):
        super().__init__(base_dir, img_size, augmentations, class_channels, channel_wise_mask, mask_size)
        self.image_loader = self.rgb_loader

    def __next__(self):
        return self.next()

    def __iter__(self):
        self.re_init()
        return self

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return self.size