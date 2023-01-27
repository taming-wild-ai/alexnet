from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Dataset(object):
    def init_CIFAR10(self):
        self.train_loader = DataLoader(
            CIFAR10(
                './data/cifar10',
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor()])),
            batch_size=128,
            shuffle=True,
            num_workers=8)
        data_test = CIFAR10(
            './data/cifar10',
            train=False,
            download=True,
                transform=transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor()]))
        self.test_len = len(data_test)
        self.test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    def init_MNIST(self):
        self.train_loader = DataLoader(
            MNIST(
                './data/mnist',
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor()])),
            batch_size=256,
            shuffle=True,
            num_workers=8)

        data_test = MNIST('./data/mnist',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize((128, 128)),
                            transforms.ToTensor()]))
        self.test_len = len(data_test)
        self.test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    def __init__(self, name):
        {
            'mnist': self.init_MNIST,
            'cifar10': self.init_CIFAR10
        }[name]()