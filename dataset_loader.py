#this is where we mess with the datasets and dataloaders

#implemet class counter (n_classes)

import logging #learn about logging
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dls(conf, **kwargs):
    
    datasets = set_dataset(conf,n_classes=10)

    train_dl = DataLoader(datasets['train'], batch_size=conf.batch_size, shuffle=True, **kwargs)
    val_dl = DataLoader(datasets['val'], batch_size=conf.batch_size*2, **kwargs)
    
    return {'train':train_dl,'val':val_dl}



def set_dataset(args,n_classes):

    """
    configure the dataset
    """
    logging.info(f"Using {args.dataset} dataset")
    
    #MNIST
    if args.dataset == 'mnist':
        # training set
        train_dataset = torchvision.datasets.MNIST('../data/mnist', train=True, 
                                                   download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), 
                                                                            (0.3081,))
                                                   ]))

        # validation set
        val_dataset = torchvision.datasets.MNIST('../data/mnist', train=False, 
                                                  download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), 
                                                                            (0.3081,))
                                                   ]))
        return {'train':train_dataset,'val':val_dataset}

    # prepare Fashion-MNIST dataset
    if opt.dataset == 'fashionmnist':
        # training set
        train_dataset = torchvision.datasets.FashionMNIST('../data/fashionmnist', train=True, 
                                                   download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), 
                                                                            (0.3081,))
                                                   ])) #?check into normalize paramets for fminst

        # validation set
        val_dataset = torchvision.datasets.FashionMNIST('../data/fashionmnist', train=False, 
                                                  download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), 
                                                                            (0.3081,))
                                                   ])) #?check into normalize paramets for fminst
        return {'train':train_dataset,'val':val_dataset}

    # prepare CIFAR-10 dataset
    elif args.dataset == 'cifar10':
        # define the image transformation operators
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])        
        train_dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', 
                                                     train=True, 
                                                     download=True, 
                                                     transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', 
                                                    train=False, 
                                                    download=True, 
                                                    transform=transform_test)
        return {'train':train_dataset,'val':val_dataset}

    else:
        raise NotImplementedError #?why?