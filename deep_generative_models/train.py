import datetime
import torch
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from deep_generative_models.dataset import create_dataloader
from deep_generative_models.model import VariationalAutoEncoder
from config.paths import CELL_DATA, IMAGES, STORAGE
from deep_generative_models.model_mashood import VAE


class VAETrainer:
    def __init__(self, model, input_dim, batch_size, lr, num_epochs, device):
        self.device = device´
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # self.model = VariationalAutoEncoder(input_dim, h_dim, z_dim).to(device)
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss(reduction="sum")
        # self.loss_fn = nn.functional.mse_loss(x_hat, x, reduction='sum')
        self.train_loader, self.val_loader = self.load_data()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_data(self):
        train_brains = ["B01", "B02", "B05"]  # Training brains
        val_brains = ["B07"]
        # brains = ["B20"]  # Test
        hdf5_file_path = CELL_DATA
        tile_size = 128
        batch_size = 4
        train_dataloader = create_dataloader(
            hdf5_file_path,
            train_brains,
            tile_size,
            batch_size,
            num_workers=0,
        )

        val_dataloader = create_dataloader(
            hdf5_file_path,
            val_brains,
            tile_size,
            batch_size,
            tiles_per_epoch=100,
            num_workers=0,
        )

        return train_dataloader, val_dataloader

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        loop = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Training",
        )
        for i, x in loop:
            # deprecated FFN: x = x.to(self.device).view(x.shape[0], self.input_dim)
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
                x_reconstructed, mu, sigma = self.model(x)
                loss = self.compute_loss(x, x_reconstructed, mu, sigma)
                val_loss += loss.item()
                loop.set_postfix(loss=loss.item())
        return val_loss / len(self.val_loader)

    def compute_loss(self, x, x_reconstructed, mu, sigma):
        reconstruction_loss = nn.functional.mse_loss(
            x_reconstructed, x, reduction="sum"
        )
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
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
        plt.savefig(IMAGES / f"train_val_loss_plot_{self.model.name}_{self.timestamp}.png")
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
        self.save_model(STORAGE / f"trained_model_{self.model.name}_{self.timestamp}.pth")
        self.plot_losses(train_losses, val_losses)


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 128
    Z_DIM = 128
    # H_DIM = 200 deprecated
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    LR = 3e-4

    MODEL = VAE(INPUT_DIM, Z_DIM)

    trainer = VAETrainer(MODEL, INPUT_DIM, BATCH_SIZE, LR, NUM_EPOCHS, DEVICE)
    trainer.train()


if __name__ == "__main__":
    main()
