import datetime
import json
import torch
import tomli
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from deep_generative_models.dataset import create_dataloader
from deep_generative_models.model import VariationalAutoEncoder
from config.paths import CELL_DATA, IMAGES, STORAGE, TRAIN_CONFIG
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# from deep_generative_models.model_mashood_enhanced import VAE

from deep_generative_models.model_mashood import VAE

# from deep_generative_models.model_cnn import VAE

with open(TRAIN_CONFIG, "rb") as f:
    config = tomli.load(f)


class VAETrainer:
    def __init__(self, model, device, num_epochs):
        self.device = device
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), **config["optimizer"])
        self.train_loader, self.val_loader = self.load_data()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",  # Minimize the monitored metric (e.g., validation loss)
            factor=0.1,  # Reduce LR by a factor of 0.1
            patience=1,  # Number of epochs with no improvement to wait
            verbose=True,  # Print a message when the LR is reduced
            threshold=0.08,
        )

    def load_data(self):
        hdf5_file_path = CELL_DATA
        train_loader = create_dataloader(hdf5_file_path, **config["train_loader"])
        val_loader = create_dataloader(hdf5_file_path, **config["val_loader"])
        return train_loader, val_loader

    def train_epoch(self):
        self.model.train()
        train_losses = []
        loop = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Training",
        )
        for i, x in loop:
            x = x.to(self.device)
            x_reconstructed, mu, sigma = self.model(x)
            loss = self.compute_loss(x, x_reconstructed, mu, sigma, **config["loss"])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
        return train_losses

    def validate_epoch(self):
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            loop = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc="Validation",
            )
            for i, x in loop:
                x = x.to(self.device)
                x_reconstructed, mu, sigma = self.model(x)
                loss = self.compute_loss(
                    x, x_reconstructed, mu, sigma, **config["loss"]
                )
                val_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())
        return val_losses

    def train(self):
        train_losses = []
        val_losses = []
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            train_losses.extend(train_loss)
            val_losses.extend(val_loss)
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {sum(train_loss)/len(train_loss):.4f}, Validation Loss: {sum(val_loss)/len(val_loss):.4f}"
            )
            val_loss_avg = sum(val_loss) / len(val_loss)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss_avg}")
            self.scheduler.step(val_loss_avg)
            print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        self.save_model(
            STORAGE / f"trained_model_{self.model.name}_{self.timestamp}.pth"
        )

        config_dir = STORAGE / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        self.plot_losses(train_losses, val_losses)

    def interp1d(self, array: np.ndarray, new_len: int) -> np.ndarray:
        la = len(array)
        return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)

    def plot_losses(self, train_losses, val_losses):
        # Interpolate validation losses to match the length of training losses
        val_losses_stretched = self.interp1d(val_losses, len(train_losses))
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            label="Training Loss",
        )
        plt.plot(
            range(1, len(val_losses_stretched) + 1),
            val_losses_stretched,
            label="Validation Loss",
        )
        plt.title("Training and Validation Loss per Batch")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(
            IMAGES / f"train_val_loss_plot_{self.model.name}_{self.timestamp}.png"
        )
        plt.show()

    def compute_loss(self, x, x_reconstructed, mu, logvar, beta: float) -> torch.Tensor:
        reconstruction_loss = nn.functional.mse_loss(
            x_reconstructed, x, reduction="sum"
        )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + beta * kl_div

    def logcosh_loss(
        self, x, x_reconstructed, mu, log_var, alpha: float, beta: float
    ) -> torch.Tensor:
        # ß_norm =  ßM/N from the paper where M is the number of samples in minibatch and N is the total number of samples in the data
        kld_weight = (
            config["train_loader"]["batch_size"]
            / config["train_loader"]["tiles_per_epoch"]
        )
        beta_norm = beta * kld_weight

        t = x_reconstructed - x
        recons_loss = (
            alpha * t
            + torch.log(1.0 + torch.exp(-2 * alpha * t))
            - torch.log(torch.tensor(2.0))
        )
        recons_loss = (1.0 / alpha) * recons_loss.mean()

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        return recons_loss + beta_norm * kld_loss

    def save_model(self, path):
        print("saving model")
        torch.save(self.model.state_dict(), path)


def main():
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps:0" if torch.backends.mps.is_available() else "cpu"
    )

    print(DEVICE)
    MODEL = VAE(**config["model_mashood"], device=DEVICE)
    # MODEL.init_params(0.0, 0.02)

    trainer = VAETrainer(MODEL, DEVICE, **config["trainer"])
    trainer.train()


if __name__ == "__main__":
    main()
