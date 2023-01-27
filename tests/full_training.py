import unittest

class TestFullTraining(unittest.TestCase):
    def test_MNIST(self):
        from alexnet import AlexNet
        import torch.nn as nn
        import torch.optim as optim
        from dataset import Dataset

        dataset = Dataset()
        net = AlexNet(10, 1)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=2e-4)
        for epoch, batch, loss in net.train_loss_gen(
            epochs=15,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=dataset.train_loader):
            print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss}")
        total_correct = 0
        for _batch, loss, correct in net.test(criterion, dataset.test_loader):
            total_correct += correct
        self.assertAlmostEqual(float(total_correct) / dataset.test_len, 0.99, 2)

if __name__ == "__main__":
    unittest.main()
