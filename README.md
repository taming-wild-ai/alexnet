# Introduction

The code in this repository illustrates training from scratch and transfer
learning using the [Torchvision implementation of the AlexNet
model](https://pytorch.org/vision/stable/models/alexnet.html).

# Installing Dependencies

`pip install -r requirements.txt`

# Running Tests

Full training on MNIST to 99% accuracy:

```shell
python -m unittest tests.full_training.TestFullTraining.test_MNIST
```

Full training on CIFAR 10 to 81% accuracy:

```shell
python -m unittest tests.full_training.TestFullTraining.test_CIFAR10_MEDIOCRE_RESULTS
```

First experiment with transfer learning on CIFAR 10 to 80% accuracy:

```shell
python -m unittest tests.full_training.TestFullTraining.test_CIFAR10_transfer_MEDIOCRE_RESULTS1
```

Second experiment with transfer learning on CIFAR 10 to 81% accuracy:

```shell
python -m unittest tests.full_training.TestFullTraining.test_CIFAR10_transfer_MEDIOCRE_RESULTS2
```
