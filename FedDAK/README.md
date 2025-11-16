# FedDAK


## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy

## Datasets
We conduct experiments on six datasets:
- MNIST
- CIFAR-10
- CIFAR-100
- FashionMNIST
- DomainNet

## Training
```
python main.py --dataset Cifar100 --num_clients 20 --global_epochs 201 --join_ratio 1.0 --partition dir --alpha 0.1
