import unittest
from alexnet import AlexNet
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset

class TestSingleBatch(unittest.TestCase):
    def test_CIFAR10(self):
        dataset = Dataset('cifar10')
        net = AlexNet(10, 3)
        criterion = nn.CrossEntropyLoss()
        for epoch, batch, loss in net.train_loss_gen(
            epochs=90,
            criterion=criterion,
            optimizer=optim.SGD(net.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
            data_loader=dataset.train_loader):
            self.assertLess(loss, 2.5)
            break

    def test_MNIST(self):
        dataset = Dataset('mnist')
        net = AlexNet(10, 1)
        criterion = nn.CrossEntropyLoss()
        for epoch, batch, loss in net.train_loss_gen(
            epochs=15,
            criterion=criterion,
            optimizer=optim.Adam(net.model.parameters(), lr=2e-4),
            data_loader=dataset.train_loader):
            self.assertLess(loss, 2.5)
            break

if __name__ == "__main__":
    unittest.main()
