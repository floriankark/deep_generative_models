import torch
import torch.nn as nn
from torchinfo import summary


class VAE(nn.Module):
    def __init__(
        self,
        input_dim=128,
        last_hidden_dim=128,
        encoder_channels=(1, 32, 64, 128),
        decoder_channels=(128, 64, 32, 1),
        device=torch.device("cpu"),
    ):

        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.last_hidden_dim = last_hidden_dim
        self.device = device
        self.name = "VAE_MASHOOD"

        print(self.input_dim, last_hidden_dim, decoder_channels, encoder_channels)

        # Dynamic encoder based on provided channels
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        encoder_channels[i],
                        encoder_channels[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2),
                )
                for i in range(len(encoder_channels) - 1)
            ],
            nn.Flatten(),
        ).to(device)

        self.flattened_dim = (
            input_dim // 2 ** (len(encoder_channels) - 1)
        ) ** 2 * encoder_channels[-1]

        self.mean = nn.Linear(self.flattened_dim, last_hidden_dim).to(device)
        self.var = nn.Linear(self.flattened_dim, last_hidden_dim).to(device)

        self.decoder_input = nn.Linear(last_hidden_dim, self.flattened_dim).to(device)

        # Dynamic decoder based on provided channels
        self.decoder = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    decoder_channels[0],
                    input_dim // 2 ** (len(decoder_channels) - 1),
                    input_dim // 2 ** (len(decoder_channels) - 1),
                ),
            ),
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_channels[i],
                        decoder_channels[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2),
                )
                for i in range(len(decoder_channels) - 1)
            ],
        ).to(device)

    def encode(self, x):
        x = self.encoder(x.to(self.device))
        var = torch.clamp(self.var(x), min=1e-6)
        mean = self.mean(x)
        return var, mean

    def reparameterization(self, mean, var):
        self.norm = torch.distributions.Normal(0, 1)
        epsilon = self.norm.sample(var.shape).to(self.device)
        z = mean + var * epsilon
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        var, mean = self.encode(x)
        z = self.reparameterization(mean, var)
        x = self.decode(z)
        return x, mean, var


if __name__ == "__main__":
    # Test the extended model
    model = VAE(
        input_dim=128,
        last_hidden_dim=128,
        encoder_channels=(1, 32, 64, 128),
        decoder_channels=(128, 64, 32, 1),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    summary(model, input_size=(16, 1, 128, 128), device=model.device)
