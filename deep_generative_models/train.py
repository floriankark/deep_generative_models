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

# from deep_generative_models.model_mashood import VAE
from deep_generative_models.model_cnn import VAE

with open(TRAIN_CONFIG, "rb") as f:
    config = tomli.load(f)


class VAETrainer:
    def __init__(self, model, device, num_epochs):
        self.device = device
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), **config["optimizer"])
        self.train_loader, self.val_loader = self.load_data()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_data(self):
        hdf5_file_path = CELL_DATA
        train_loader = create_dataloader(hdf5_file_path, **config["train_loader"])
        val_loader = create_dataloader(hdf5_file_path, **config["val_loader"])
        return train_loader, val_loader

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        loop = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Training",
        )
        for i, x in loop:
            x = x.to(self.device)
            x_reconstructed, mu, sigma = self.model(x)
            loss = self.compute_loss(x, x_reconstructed, mu, sigma)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return train_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            loop = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc="Validation",
            )
            for i, x in loop:
                x = x.to(self.device)
                x_reconstructed, mu, sigma = self.model(x)
                loss = self.compute_loss(x, x_reconstructed, mu, sigma)
                val_loss += loss.item()
                loop.set_postfix(loss=loss.item())
        return val_loss / len(self.val_loader)

    """def compute_loss(self, x, x_reconstructed, mu, sigma):
        reconstruction_loss = nn.functional.mse_loss(
            x_reconstructed, x, reduction="sum"
        )
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        return reconstruction_loss + kl_div"""

    def compute_loss(self, x, x_reconstructed, mu, logvar):
        reconstruction_loss = nn.functional.mse_loss(
            x_reconstructed, x, reduction="sum"
        )
        kl_div = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1)
        return reconstruction_loss + kl_div

    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, self.num_epochs + 1),
            train_losses,
            label="Training Loss",
            marker="o",
        )
        plt.plot(
            range(1, self.num_epochs + 1),
            val_losses,
            label="Validation Loss",
            marker="o",
        )
        plt.title("Training and Validation Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(
            IMAGES / f"train_val_loss_plot_{self.model.name}_{self.timestamp}.png"
        )
        plt.show()

    def save_model(self, path):
        print("saving model")
        torch.save(self.model.state_dict(), path)

    def train(self):
        train_losses = []
        val_losses = []
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )
        self.save_model(
            STORAGE / f"trained_model_{self.model.name}_{self.timestamp}.pth"
        )

        config_dir = STORAGE / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / f"{self.model.name}_{self.timestamp}.json", "w") as f:
            json.dump(config["model"], f)

        self.plot_losses(train_losses, val_losses)


def main():
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps:0" if torch.backends.mps.is_available() else "cpu"
    )

    print(DEVICE)
    MODEL = VAE(**config["model"], device=DEVICE)

    trainer = VAETrainer(MODEL, DEVICE, **config["trainer"])
    trainer.train()


if __name__ == "__main__":
    main()
