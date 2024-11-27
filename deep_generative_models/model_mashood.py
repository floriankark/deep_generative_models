import torch.nn as nn
import torch
from torchinfo import summary


class VAE(nn.Module):
    def __init__(self, input_dim=128, last_hidden_dim=128):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.name = "VAE_MASHOOD"

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.flattened_dim = (input_dim // 8) * (input_dim // 8) * 128

        self.mean = nn.Linear(self.flattened_dim, last_hidden_dim)
        self.var = nn.Linear(self.flattened_dim, last_hidden_dim)

        self.decoder_input = nn.Linear(last_hidden_dim, self.flattened_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, input_dim // 8, input_dim // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        x = self.encoder(x)
        var = torch.clamp(self.var(x), min=1e-6)
        mean = self.mean(x)
        return var, mean

    def reparameterization(self, mean, var):
        self.norm = torch.distributions.Normal(0, 1)
        epsilon = self.norm.sample(var.shape)
        z = mean + var * epsilon
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        var, mean = self.encode(x)
        z = self.reparameterization(mean, var)
        x_hat = self.decode(z)
        return x_hat, mean, var

if __name__ == "__main__":
    # Check the model architecture
    model = VAE(input_dim=128, last_hidden_dim=128)
    summary(model, input_size=(16, 1, 128, 128), device="cpu")