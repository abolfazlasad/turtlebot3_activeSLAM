import torch
import numpy
import torch.nn as nn

MAX_SCAN_DISTANCE = 3.5

def transform_scan(s):
    s = s.reshape(-1, 360)
    s = numpy.nan_to_num(s, posinf=MAX_SCAN_DISTANCE)
    s = numpy.roll(s, 180, axis=-1)
    s = torch.from_numpy(s).to(torch.float32)
    s /= MAX_SCAN_DISTANCE
    return s

class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(4, 4, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(4, 8, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(8, 16, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5),
            nn.ReLU(),

            nn.Flatten()
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 360)
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),

            nn.ConvTranspose1d(16, 16, kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, kernel_size=4),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),

            nn.ConvTranspose1d(8, 8, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 4, kernel_size=5),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose1d(4, 4, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(-1, 32, 1)
        x = self.decoder(x)
        x = x.reshape(-1, 360)
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
