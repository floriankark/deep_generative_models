import tomli
import torch
from torch import nn
from torchinfo import summary

from config.paths import CNN_MODEL_CONFIG

# Load the TOML config
with open(CNN_MODEL_CONFIG, "rb") as f:
    config = tomli.load(f)


class Block(nn.Module):
    def __init__(self, channel_in: int, channel_out: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_out,
            **config["block"],
        )
        self.bn = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel_out,
            out_channels=channel_out,
            **config["block"],
        )

    def forward(self, x):
        return self.conv2(self.relu(self.bn(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, channels: list[int], latent_dim: int):
        super().__init__()
        self.channels = channels
        self.encoder_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(**config["pool"])
        
        self.fc = nn.Linear(
            in_features=config["fc_input"] * config["fc_input"] * channels[-1],
            out_features=config["fc_output"],
            bias=config["bias"],
        )

        self.fc_mu = nn.Linear(in_features=config["fc_output"], out_features=latent_dim)
        self.fc_logvar = nn.Linear(
            in_features=config["fc_output"], out_features=latent_dim
        )

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
            x = self.pool(x)
        x = x.reshape(-1, self.channels[-1] * config["fc_input"] * config["fc_input"])
        x = self.fc(x)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self, channels: list[int], latent_dim: int):
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(
            in_features=latent_dim,
            out_features=config["fc_input"] * config["fc_input"] * channels[0],
            bias=config["bias"],
        )
        self.up_conv = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i], **config["up_conv"])
                for i in range(len(channels) - 1)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc(x)).reshape(
            -1, self.channels[0], config["fc_input"], config["fc_input"]
        )
        for i in range(len(self.channels) - 1):
            x = self.up_conv[i](x)
            x = self.decoder_blocks[i](x)
        return x


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = config[
            "latent_dim"
        ],  # dont need to specify input_dim, already in config TODO: change that so that config value is passed at init here
        encoder_channels: list[int] = config["enc"]["channels"],
        decoder_channels: list[int] = config["dec"]["channels"],
        device: torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.name = "VAE_FLO"
        self.encoder = Encoder(encoder_channels, latent_dim).to(device)
        self.decoder = Decoder(decoder_channels, latent_dim).to(device)
        self.head = nn.Conv2d(
            in_channels=decoder_channels[-1], out_channels=1, **config["head"]
        ).to(device)
        self.tanh = nn.Tanh()
        self.device = device

    def reparametrize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = torch.randn_like(mu).to(self.device)
        return mu + logvar * eps

    def loss(self, x, x_hat, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kldivergence = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1)
        l2_reg = sum(torch.sum(param**2) for param in self.parameters())
        return recon_loss + kldivergence + l2_reg * 1e-6

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.tanh(
            self.head(self.decoder(z))
        )  # Normalize the output to [-1, 1] for the MSE loss
        return x_hat, mu, logvar

if __name__ == "__main__":
    """batch_size = 1
    image_size = 28
    input_channels = 1
    x = torch.randn(batch_size, input_channels, image_size, image_size) 
    print(x.shape)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(encoder_channels=config["enc"]["channels"], 
              decoder_channels=config["dec"]["channels"], 
              latent_dim=config["latent_dim"], 
              device=device
              )
    x_reconstructed = vae(x)
    print(x_reconstructed.shape)
    print(vae.loss(x, x_reconstructed, vae.encoder(x)[0], vae.encoder(x)[1]))"""
    #model = VAE(input_dim=256, latent_dim=128)
    #summary(model, input_size=(4, 1, 256, 256), device="cpu")
