import torch.nn as nn

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

            nn.Conv2d(16, 32, kernel_size=3, stride=2),
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
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),
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
            nn.ReLU(),
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
