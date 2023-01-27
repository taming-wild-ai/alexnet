import unittest
from alexnet import AlexNet
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
import torchvision.models as tvm

class TestFullTraining(unittest.TestCase):
    def test_CIFAR10_transfer_MEDIOCRE_RESULTS2(self):
        dataset = Dataset('cifar10')
        net = AlexNet(10, 3, transfer_from=tvm.AlexNet_Weights.DEFAULT)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.model.classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        lr_lambda = lambda epoch: 0.1 if epoch in [20, 40, 60, 80] else 1.0
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lr_lambda)
        for epoch, batch, loss in net.train_loss_gen(
            epochs=90,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=dataset.train_loader):
            if batch == len(dataset.train_loader) - 1: # end of epoch
                scheduler.step()
        total_correct = 0
        for _batch, loss, correct in net.test(criterion, dataset.test_loader):
            total_correct += correct
        self.assertAlmostEqual(float(total_correct) / dataset.test_len, 0.81, 2)

    def test_CIFAR10_transfer_MEDIOCRE_RESULTS1(self):
        dataset = Dataset('cifar10')
        net = AlexNet(10, 3, transfer_from=tvm.AlexNet_Weights.DEFAULT)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.model.classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        lr_lambda = lambda epoch: 0.1 if epoch in [22, 45, 67] else 1.0
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lr_lambda)
        for epoch, batch, loss in net.train_loss_gen(
            epochs=90,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=dataset.train_loader):
            if batch == len(dataset.train_loader) - 1: # end of epoch
                scheduler.step()
        total_correct = 0
        for _batch, loss, correct in net.test(criterion, dataset.test_loader):
            total_correct += correct
        self.assertAlmostEqual(float(total_correct) / dataset.test_len, 0.8, 1)

    def test_CIFAR10_MEDIOCRE_RESULTS(self):
        dataset = Dataset('cifar10')
        net = AlexNet(10, 3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        lr_lambda = lambda epoch: 0.1 if epoch in [22, 45, 67] else 1.0
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lr_lambda)
        for epoch, batch, loss in net.train_loss_gen(
            epochs=90,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=dataset.train_loader):
            if batch == len(dataset.train_loader) - 1: # end of epoch
                scheduler.step()
        total_correct = 0
        for _batch, loss, correct in net.test(criterion, dataset.test_loader):
            total_correct += correct
        self.assertAlmostEqual(float(total_correct) / dataset.test_len, 0.81, 2)

    def test_MNIST(self):
        dataset = Dataset('mnist')
        net = AlexNet(10, 1)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.model.parameters(), lr=2e-4)
        for epoch, batch, loss in net.train_loss_gen(
            epochs=15,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=dataset.train_loader):
            pass
        total_correct = 0
        for _batch, loss, correct in net.test(criterion, dataset.test_loader):
            total_correct += correct
        self.assertAlmostEqual(float(total_correct) / dataset.test_len, 0.99, 2)

if __name__ == "__main__":
    unittest.main()
