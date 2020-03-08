#this is where we mess with the datasets and dataloaders

##fix the class cropper

import logging #learn about logging
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None, features4tree=None):
        self.train_dl,self.valid_dl,self.c,self.features4tree = train_dl,valid_dl,c,features4tree
        
    @property
    def train_ds(self): return self.train_dl.dataset
        
    @property
    def valid_ds(self): return self.valid_dl.dataset

class Dataset():
    def __init__(self, x, y): self.x,self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i],self.y[i]

class DataLoader():
    def __init__(self, ds, bs): self.ds,self.bs = ds,bs
    def __iter__(self):
        for i in range(0, len(self.ds), self.bs): yield self.ds[i:i+self.bs]
    def __len__(self): return (len(self.ds)//self.bs)+1

# def data_shape(conf, datasets=None, n_classes=None):
#     if datasets == None:
#         datasets = set_dataset(conf)

#     datasets = class_crop(conf, datasets=datasets, n_classes=n_classes)
    
#     n_samples = datasets['train'].data.shape[0]
#     features = torch.tensor(datasets['train'].data.shape[1:])
#     n_classes = class_counter(conf, datasets=datasets)

#     return n_samples,features,n_classes

# def class_counter(conf, datasets=None, n_classes=None):
    
#     if datasets == None:
#         datasets = set_dataset(conf)

#     max_classes = datasets['train'].targets.max()+1

#     if n_classes == None:
#         n_classes = max_classes.item()

#     elif (n_classes > max_classes):
#         raise Exception("the number of classes chosen cannot be larger than the number of classes in the dataset.")

#     return n_classes
        
# def class_crop(conf, datasets=None, n_classes=None):
#     if datasets == None:
#         datasets = set_dataset(conf)

#     n_classes = class_counter(conf, datasets=datasets)

#     datasets['train'].data = datasets['train'].data[datasets['train'].targets < n_classes]
#     datasets['train'].tragets = datasets['train'].targets[datasets['train'].targets < n_classes]

#     datasets['val'].data = datasets['val'].data[datasets['val'].targets < n_classes]
#     datasets['val'].tragets = datasets['val'].targets[datasets['val'].targets < n_classes]

#     return datasets



# def get_dls(conf, datasets=None, n_classes=None, **kwargs):
    
#     if datasets == None:
#         datasets = set_dataset(conf)

#     if n_classes == None:
#         n_classes = class_counter(conf, datasets=datasets)

#     datasets = class_crop(conf, datasets=datasets, n_classes=n_classes)

#     train_dl = DataLoader(datasets['train'], batch_size=conf.batch_size, shuffle=True, **kwargs)
#     val_dl = DataLoader(datasets['val'], batch_size=conf.batch_size*2, **kwargs)
    
#     return train_dl, val_dl



# def set_dataset(conf):

#     """
#     configure the dataset
#     """
#     logging.info(f"Using {conf.dataset} dataset")
    
#     #MNIST
#     if conf.dataset == 'mnist':
#         # training set
#         train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, 
#                                                    download=True,
#                                                    transform=transforms.Compose([
#                                                        transforms.ToTensor(),
#                                                        transforms.Normalize((0.1307,), 
#                                                                             (0.3081,))
#                                                    ]))

#         # validation set
#         val_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, 
#                                                   download=True,
#                                                    transform=transforms.Compose([
#                                                        transforms.ToTensor(),
#                                                        transforms.Normalize((0.1307,), 
#                                                                             (0.3081,))
#                                                    ]))
#         return {'train':train_dataset,'val':val_dataset}

#     # prepare Fashion-MNIST dataset
#     if conf.dataset == 'fashionmnist':
#         # training set
#         train_dataset = torchvision.datasets.FashionMNIST('./data/fashionmnist', train=True, 
#                                                    download=True,
#                                                    transform=transforms.Compose([
#                                                        transforms.ToTensor(),
#                                                        transforms.Normalize((0.1307,), 
#                                                                             (0.3081,))
#                                                    ])) #?check into normalize paramets for fminst

#         # validation set
#         val_dataset = torchvision.datasets.FashionMNIST('./data/fashionmnist', train=False, 
#                                                   download=True,
#                                                    transform=transforms.Compose([
#                                                        transforms.ToTensor(),
#                                                        transforms.Normalize((0.1307,), 
#                                                                             (0.3081,))
#                                                    ])) #?check into normalize paramets for fminst
#         return {'train':train_dataset,'val':val_dataset}

#     # prepare CIFAR-10 dataset
#     elif conf.dataset == 'cifar10':
#         # define the image transformation operators
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), 
#                                  (0.2023, 0.1994, 0.2010)),
#         ])
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), 
#                                  (0.2023, 0.1994, 0.2010)),
#         ])        
#         train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', 
#                                                      train=True, 
#                                                      download=True, 
#                                                      transform=transform_train)
#         val_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', 
#                                                     train=False, 
#                                                     download=True, 
#                                                     transform=transform_test)
#         return {'train':train_dataset,'val':val_dataset}

#     else:
#         raise NotImplementedError #?why?