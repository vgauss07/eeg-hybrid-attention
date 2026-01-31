import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.25):
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(F1, F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            self.feature_dim = self._features(dummy).shape[1]

        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def _features(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        return self.classifier(self._features(x))
