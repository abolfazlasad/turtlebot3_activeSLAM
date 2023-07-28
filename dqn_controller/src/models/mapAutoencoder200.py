import torch
import numpy
import torch.nn as nn
import torchvision.transforms as transforms

def transform_map(m):
    assert(m.min() == -1)
    assert(m.max() <= 100)

    m = m.squeeze()
    assert(len(m.shape) == 2)

    m[m > 0] = 1
    m += 1
    m = torch.from_numpy(m).to(torch.float32)
    m /= 2.

    assert(0 <= m.min())
    assert(m.max() <= 1)

    idx0 = numpy.arange(m.shape[0])[m.sum(axis=1) != 0][[0, -1]] + [-5, 5]
    idx1 = numpy.arange(m.shape[1])[m.sum(axis=0) != 0][[0, -1]] + [-5, 5]
    len0 = idx0[1] - idx0[0]
    len1 = idx1[1] - idx1[0]
    idx0[1] += len0 % 2
    idx1[1] += len1 % 2
    len0 = idx0[1] - idx0[0]
    len1 = idx1[1] - idx1[0]
    if len0 > len1:
        diff = (len0 - len1) // 2
        idx1 += [-diff, diff]
    elif len1 > len0:
        diff = (len1 - len0) // 2
        idx0 += [-diff, diff]

    mm = m[idx0[0]:idx0[1], idx1[0]:idx1[1]]
    return transforms.Resize(size=(200, 200), antialias=True)(mm[None, :, :])


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(4, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=3),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(4, 4, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(-1, 64, 2, 2)
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x