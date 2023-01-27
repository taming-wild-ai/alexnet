import torchvision.models as tvm
import torch.nn as nn

class AlexNet(object):
    def __init__(self, num_classes, num_channels, transfer_from=None):
        super(AlexNet, self).__init__()
        if transfer_from is None:
            self.model = tvm.AlexNet(num_classes)
            self.model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2)
        else:
            self.model = tvm.alexnet(weights=transfer_from)
            if num_channels != self.model.features[0].in_channels:
                raise Exception(f"num_channels parameter must equal {self.model.features[0].num_channels} for these pretrained weights")
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes))

    def train_loss_gen(self, epochs, criterion, optimizer, data_loader):
        for epoch in range(1, epochs+1):
            self.model.train()
            for batch, (images, labels) in enumerate(data_loader):
                optimizer.zero_grad()

                output = self.model(images)

                loss = criterion(output, labels)

                yield epoch, batch, loss

                loss.backward()
                optimizer.step()

    def test(self, criterion, data_loader):
        self.model.eval()
        for batch, (images, labels) in enumerate(data_loader):
            output = self.model(images)
            loss = criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            correct = pred.eq(labels.view_as(pred)).sum()
            yield batch, loss, correct
