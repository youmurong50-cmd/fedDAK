import torch
import torch.nn as nn


class GFE(nn.Module):
    """Global Feature Extractor"""
    def __init__(self, in_channels=1, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        z = self.convs(x)
        z = z.view(x.size(0), -1)
        z = self.fc(z)
        return z


class CSFE(nn.Module):
    """Client-Specific Feature Extractor"""
    def __init__(self, in_channels=1, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, embed_dim * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        z = self.convs(x)
        z = z.view(x.size(0), -1)
        z = self.fc(z)
        return z


class FedDAKModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.gfe = GFE(in_channels, hidden_size, embed_dim)
        self.csfe = CSFE(in_channels, hidden_size, embed_dim)
        self.phead = nn.Linear(embed_dim * 2, num_classes)

    def classification(self, x):
        # Client-specific feature: frozen, use mean (deterministic)
        with torch.no_grad():
            z = self.csfe(x)
            mean, logvar = torch.split(z, self.embed_dim, dim=1)
            csf = mean  # no reparameterization during inference

        # Global feature (trainable)
        gf = self.gfe(x)

        # Concatenate and classify
        pf = torch.cat([gf, csf], dim=1)
        logits = self.phead(pf)
        return logits, gf, csf


def create_model(dataset):
    if dataset == "MNIST":
        return FedDAKModel(in_channels=1, num_classes=10, hidden_size=1024, embed_dim=512)
    elif dataset == "FashionMNIST":
        return FedDAKModel(in_channels=1, num_classes=10, hidden_size=1024, embed_dim=512)
    elif dataset == "Cifar10":
        return FedDAKModel(in_channels=3, num_classes=10, hidden_size=1600, embed_dim=512)
    elif dataset == "Cifar100":
        return FedDAKModel(in_channels=3, num_classes=100, hidden_size=1600, embed_dim=512)
    elif dataset == "DomainNet":
        return FedDAKModel(in_channels=3, num_classes=10, hidden_size=10816, embed_dim=512)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")