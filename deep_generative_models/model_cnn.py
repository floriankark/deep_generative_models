import tomli
import torch
from torch import nn
from torchinfo import summary

from config.paths import CNN_MODEL_CONFIG

with open(CNN_MODEL_CONFIG, "rb") as f:
    config = tomli.load(f)


class EncoderBlock(nn.Module):
    def __init__(self, channel_in: int, channel_out: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_out,
            **config["block"],
        )
        self.bn = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class DecoderBlock(nn.Module):
    def __init__(self, channel_in: int, channel_out: int) -> None:
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels=channel_in,
            out_channels=channel_out,
            **config["block"],
        )
        self.bn = nn.BatchNorm2d(channel_out)
        self.relu = nn.LeakyReLU(**config["leaky_relu"])

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, channels: list[int], latent_dim: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Sequential(
            *[EncoderBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        self.fc_mu = nn.Linear(in_features=config["fc_input"] * config["fc_input"] * channels[-1], 
                               out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=config["fc_input"] * config["fc_input"] * channels[-1],
                                   out_features=latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, config["fc_input"] * config["fc_input"] * self.channels[-1])
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self, channels: list[int], latent_dim: int):
        super().__init__()
        self.channels = channels # TODO: just reverse the encoder channels
        self.fc = nn.Linear(in_features=latent_dim, 
                            out_features=config["fc_input"] * config["fc_input"] * channels[0], 
                            bias=config["bias"])
        self.conv = nn.Sequential(
            *[DecoderBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)] 
        )
        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels[-1], out_channels=channels[-1], **config["head"]),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(**config["leaky_relu"]),
            nn.Conv2d(in_channels=channels[-1], out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()) # Normalize the output to [-1, 1] for the MSE loss

    def forward(self, x):
        x = self.fc(x).reshape(
            -1, self.channels[0], config["fc_input"], config["fc_input"]
        )
        x = self.conv(x)
        x = self.head(x)
        return x


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_channels: list[int] = config["enc"]["channels"],
        decoder_channels: list[int] = config["dec"]["channels"],
        device: torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.name = "VAE_FLO"
        self.encoder = Encoder(encoder_channels, latent_dim).to(device)
        self.decoder = Decoder(decoder_channels, latent_dim).to(device)
        self.device = device

    def reparametrize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = torch.randn_like(mu).to(self.device)
        return mu + logvar * eps

    def loss(self, x, x_hat, mu, logvar):
        var = torch.clip(logvar.exp(), min=1e-5) # Clip the variance to avoid numerical instability
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kldivergence = -0.5 * torch.sum(logvar - mu.pow(2) - var + 1)
        return recon_loss + kldivergence
    
    def init_params(self, mean=0.0, std=0.02):
        # standard is uniform initialization
        # see https://github.com/pytorch/pytorch/blob/07906f2f2b6848e3fe1c1c45e98f0f7acb54116b/torch/nn/modules/linear.py#L114
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


if __name__ == "__main__":
    # Check the model architecture
    model = VAE(input_dim=128, latent_dim=512)
    summary(model, input_size=(128, 1, 128, 128), device="cpu")
