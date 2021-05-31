"""
Dataset (source and target domain) loading object for the following object classification datasets:
PACS, VLCS, Office_Home, and Office_Caltech
The code is based on
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

import os

import torch
from torch.utils.data import ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ..prepdata.image_transform import get_transform
from .dataset_access import DatasetAccess

# default image transform
TF_DEFAULT = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
)


DOMAIN_DICT = {
    "pacs": (["art_painting", "cartoon", "photo", "sketch"], 7),
    "vlcs": (["CALTECH", "LABELME", "PASSCAL", "SUN"], 5),
    "officehome": (["Art", "Clipart", "Product", "Real_World"], 65),
    "office_caltech": (["amazon", "caltech", "dslr", "webcam"], 10),
}


class SingleDomainSet(DatasetAccess):
    def __init__(
        self, data_path, use_data="pacs", domain="art_painting", transform="default", test_size=0.2, random_state=144
    ):

        if not os.path.exists(data_path):
            raise ValueError("Data path '%s' does not" % data_path)
        domain_list, n_class = DOMAIN_DICT[use_data]
        self.use_data = use_data

        if domain not in domain_list:
            raise ValueError("Invalid domain")
        super().__init__(n_classes=n_class)

        # self.domain = domain.lower()
        self.domain = domain
        self.data_folder = os.path.join(data_path, self.domain)
        if use_data in ["officehome", "office_caltech"]:
            self.transform = get_transform(kind="office")
        elif transform == "default":
            self.transform = TF_DEFAULT
        else:
            self.transform = TF_DEFAULT

        self.random_state = random_state
        self.test_size = test_size

        # random split train test partition for PACS and OfficeHome dataset
        if self.use_data != "vlcs":
            self._dataset = ImageFolder(self.data_folder, transform=self.transform)

            torch.manual_seed(random_state)
            n_sample = len(self._dataset.imgs)
            n_test = int(n_sample * test_size)
            n_train = n_sample - n_test
            self.train, self.test = random_split(self._dataset, [n_train, n_test])

    def get_train(self):
        if self.use_data == "vlcs":
            train_folder = os.path.join(self.data_folder, "full")
            self.train = ImageFolder(train_folder, transform=self.transform)

        return self.train

    def get_test(self):
        if self.use_data == "vlcs":
            test_folder = os.path.join(self.data_folder, "test")
            self.test = ImageFolder(test_folder, transform=self.transform)

        return self.test


class MultiAccess(DatasetAccess):
    def __init__(self, data_path, use_data, domains, transform="default", **kwargs):
        domain_list, n_class = DOMAIN_DICT[use_data]
        super().__init__(n_classes=n_class)
        self.data_ = dict()
        for d in domains:
            if d not in domain_list:
                raise ValueError("Invalid domain")
            self.data_[d] = SingleDomainSet(data_path, use_data=use_data, domain=d, transform=transform, **kwargs)

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
