import torchvision.models as tvm
import torch.nn as nn

class AlexNet(tvm.AlexNet):
    def __init__(self, num_classes, num_channels):
        super(AlexNet, self).__init__(num_classes)
        self.features[0] = nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2)

    def train_loss_gen(self, epochs, criterion, optimizer, data_loader):
        for epoch in range(1, epochs+1):
            self.train()
            for batch, (images, labels) in enumerate(data_loader):
                optimizer.zero_grad()

                output = self(images)

                loss = criterion(output, labels)

                yield epoch, batch, loss

                loss.backward()
                optimizer.step()

    def test(self, criterion, data_loader):
        self.eval()
        for batch, (images, labels) in enumerate(data_loader):
            output = self(images)
            loss = criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            correct = pred.eq(labels.view_as(pred)).sum()
            yield batch, loss, correct
