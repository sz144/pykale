"""
Digits dataset (source and target domain) loading for MNIST, SVHN, MNIST-M (modified MNIST), and USPS. The code is based on 
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

import os
import sys
# from abc import ABC
from .dataset_access import DatasetAccess
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch


transform_default = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


domain_info = {'pacs': (['art_painting', 'cartoon', 'photo', 'sketch'], 7),
               'vlcs': (['CALTECH', 'LABELME', 'PASSCAL', 'SUN'], 5),
               'officehome': (['Art', 'Clipart', 'Product', 'Real_World'], 65)}


class SingleDomainSet(DatasetAccess):

    def __init__(self, data_path, use_data='pacs', domain='art_painting',
                 transform='default', test_size=0.2, random_state=144):

        if not os.path.exists(data_path):
            print('Data path \'%s\' does not' % data_path)
            sys.exit()
        domain_list, self.n_class = domain_info[use_data]
        self.use_data = use_data

        if domain not in domain_list:
            print('Invalid domain')
            sys.exit()
        super().__init__(n_classes=self.n_class)

        # self.domain = domain.lower()
        self.domain = domain
        self.data_folder = os.path.join(data_path, self.domain)
        if transform == 'default':
            # self._transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(
            #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            #     ),
            # ])
            self.transform = transform_default
        else:
            self.transform = transform

        self.random_state = random_state
        self.test_size = test_size

        # random split train test partition for PACS and OfficeHome dataset
        if self.use_data != 'vlcs':
            self._dataset = ImageFolder(self.data_folder, transform=self.transform)

            torch.manual_seed(random_state)
            n_sample = len(self._dataset.imgs)
            n_test = int(n_sample * test_size)
            n_train = n_sample - n_test
            self.train, self.test = random_split(self._dataset, [n_train, n_test])

    def get_train(self):
        if self.use_data == 'vlcs':
            train_folder = os.path.join(self.data_folder, 'full')
            self.train = ImageFolder(train_folder, transform=self.transform)

        return self.train

    def get_test(self):
        if self.use_data == 'vlcs':
            test_folder = os.path.join(self.data_folder, 'test')
            self.test = ImageFolder(test_folder, transform=self.transform)

        return self.test


class MultiAccess(DatasetAccess):

    def __init__(self, data_path, use_data, domains, transform='default', **kwargs):
        # super().__init__(n_classes=7)
        # domain_info = {'pacs': (['art_painting', 'cartoon', 'photo', 'sketch'],
        #                         PACSSet, 7),
        #                'vlcs': (['CALTECH', 'LABELME', 'PASSCAL', 'SUN'],
        #                         VLCSSet, 5),
        #                'OfficeHome': (['Art', 'Clipart', 'Product', 'Real_World'],
        #                               OfficeHomeSet, 65)}
        domain_list, n_class = domain_info[use_data]
        super().__init__(n_classes=n_class)
        self.data_ = dict()
        for d in domains:
            if d not in domain_list:
                print('Invalid target domain')
                sys.exit()
            self.data_[d] = SingleDomainSet(data_path, use_data=use_data, domain=d,
                                            transform=transform, **kwargs)

    def get_train(self):
        train_list = []
        for key in self.data_:
            train_list.append(self.data_[key].get_train())
        self.train = ConcatDataset(train_list)

        return self.train
        
    def get_test(self):
        test_list = []
        for key in self.data_:
            test_list.append(self.data_[key].get_test())
        self.test = ConcatDataset(test_list)
        
        return self.test


# class PACSSet(DatasetAccess):
#
#     def __init__(self, data_path, domain='art_painting', transform='default',
#                  test_size=0.2, random_state=144):
#         super().__init__(n_classes=7)
#         if not os.path.exists(data_path):
#             print('Data path \'%s\' does not' % data_path)
#             sys.exit()
#         if domain.lower() not in ['art_painting', 'cartoon', 'photo', 'sketch']:
#             print('Invalid domain')
#             sys.exit()
#         self._domain = domain.lower()
#         self._data_path = os.path.join(data_path, self._domain)
#         if transform == 'default':
#             # self._transform = transforms.Compose([
#             #     transforms.ToTensor(),
#             #     transforms.Normalize(
#             #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#             #     ),
#             # ])
#             self._transform = transform_default
#         else:
#             self._transform = transform
#         self._dataset = ImageFolder(self._data_path, transform=self._transform)
#
#         torch.manual_seed(random_state)
#         n_sample = len(self._dataset.imgs)
#         n_test = int(n_sample * test_size)
#         n_train = n_sample - n_test
#         self.train, self.test = random_split(self._dataset, [n_train, n_test])
#
#     def get_train(self):
#         return self.train
#
#     def get_test(self):
#         return self.test
#
#
# class VLCSSet(DatasetAccess):
#
#     def __init__(self, data_path, domain='CALTECH', transform='default'):
#         super().__init__(n_classes=5)
#         if not os.path.exists(data_path):
#             print('Data path \'%s\' does not' % data_path)
#             sys.exit()
#         self.data_path = data_path
#         if domain.upper() not in ['CALTECH', 'LABELME', 'PASSCAL', 'SUN']:
#             print('Invalid domain')
#             sys.exit()
#         self._domain = domain.upper()
#         self._data_path = os.path.join(data_path, self._domain)
#         if transform == 'default':
#             self._transform = transform_default
#         else:
#             self._transform = transform
#         # self._dataset = ImageFolder(data_path, transform=self._transform)
#
#     def get_train(self):
#         train_path = os.path.join(self._data_path, 'full')
#         self.train = ImageFolder(train_path, transform=self._transform)
#         return self.train
#
#     def get_test(self):
#
#         test_path = os.path.join(self._data_path, 'test')
#         self.test = ImageFolder(test_path, transform=self._transform)
#
#         return self.test
