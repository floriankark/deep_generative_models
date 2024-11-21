import torch.nn as nn
import torch


import torch
from torch import nn
import numpy as np

""" 
Input Dimension:
- Mnist: 28x28 -> 784


Input -> hidden dim -> mean,std -> Parameterisation trick -> Decoder -> Output img

"""


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        ## Input -> hidden dim -> mean,std -> Parameterisation trick
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(
            h_dim, z_dim
        )  # KL divergence Pushes this layer to learn Normal Dist
        self.hid_2sigma = nn.Linear(
            h_dim, z_dim
        )  # KL divergence Pushes this layer to learn Normal Dist

        # decoder
        ## -> Decoder -> Output img
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))  # normalize for mnist

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return (
            x_reconstructed,
            mu,
            sigma,
        )  # need all 3 for the loss function KL divergence and Reconstruction loss


if __name__ == "__main__":
    input_dim = 28 * 28  # flattend images
    x = torch.randn(4, input_dim)  # batch_size x image_size
    vae = VariationalAutoEncoder(input_dim=input_dim)
    x_reconstructed, mu, sigma = vae(x)

    print(x_reconstructed.shape)
    print(sigma.shape)
    print(mu.shape)


class VAE_MASHOOD(nn.Module):
    def __init__(self, input_dim=128, last_hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        # wenn zeit ist mit cnns statt linear
        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim**2, last_hidden_dim * 10
            ),  # anzahl der layers und dimension variabel machen
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim * 10, last_hidden_dim * 5),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim * 5, last_hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(last_hidden_dim * 2, last_hidden_dim),
            nn.LeakyReLU(0.2),
        )
        # man kann nicht den mean und varianz berechnen sonst stimmen die dimensionen nicht evtll später mit mean und var in einem vektor probieren
        self.mean = nn.Linear(last_hidden_dim, last_hidden_dim)
        self.var = nn.Linear(last_hidden_dim, last_hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(last_hidden_dim, last_hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(last_hidden_dim * 2, last_hidden_dim * 5),
            nn.LeakyReLU(0.2),
            nn.Linear(last_hidden_dim * 5, last_hidden_dim * 10),
            nn.LeakyReLU(0.2),
            nn.Linear(last_hidden_dim * 10, input_dim**2),
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        var = self.var(x)
        var = torch.clamp(
            var, min=1e-6
        )  # sont kann KL nan ausgeben wegen negativem log
        mean = self.mean(x)
        return var, mean

    def reparameterization(self, mean, var):
        self.norm = torch.distributions.Normal(0, 1)
        epsilon = self.norm.sample(var.shape)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        var, mean = self.encode(x)
        z = self.reparameterization(mean, var)
        x_hat = self.decode(z)
        x_hat = x_hat.reshape((-1, 1, self.input_dim, self.input_dim))
        return x_hat, mean, var
