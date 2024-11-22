import torch
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from dataset import create_dataloader
from deep_generative_models.model import VariationalAutoEncoder
from config.paths import IMAGES, STORAGE


class VAETrainer:
    def __init__(self, input_dim, h_dim, z_dim, batch_size, lr, num_epochs, device):
        self.device = device
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = VariationalAutoEncoder(input_dim, h_dim, z_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss(reduction="sum")
        # self.loss_fn = nn.functional.mse_loss(x_hat, x, reduction='sum')
        self.train_loader = self.load_data()

    def load_data(self):
        brains = ["B01", "B02", "B05"]  # Training brains
        brains = ["B20"]
        hdf5_file_path = "data/cell_data.h5"
        tile_size = 64
        batch_size = 4
        tiles_per_epoch = 100
        dataloader = create_dataloader(
            hdf5_file_path,
            brains,
            tile_size,
            batch_size,
            tiles_per_epoch,
            num_workers=0,
        )

        return dataloader

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        loop = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Training",
        )
        for i, x in loop:
            x = x.to(self.device).view(x.shape[0], self.input_dim)
            x_reconstructed, mu, sigma = self.model(x)
            loss = self.compute_loss(x, x_reconstructed, mu, sigma)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return train_loss / len(self.train_loader)

    def compute_loss(self, x, x_reconstructed, mu, sigma):
        reconstruction_loss = nn.functional.mse_loss(
            x_reconstructed, x, reduction="sum"
        )
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        return reconstruction_loss + kl_div

    def plot_losses(self, train_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, self.num_epochs + 1),
            train_losses,
            label="Training Loss",
            marker="o",
        )

        plt.title("Training and Validation Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(IMAGES / "train_val_loss_plot.png")
        plt.show()

    def save_model(self, path):
        print("saving model")
        torch.save(self.model.state_dict(), path)

    def train(self):
        train_losses = []
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            # val_loss = self.validate_epoch()
            train_losses.append(train_loss)
            # val_losses.append(val_loss)
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: WIP"
            )
        self.save_model(STORAGE / "trained_model.pth")
        self.plot_losses(train_losses)


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 4096
    H_DIM = 200
    Z_DIM = 20
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LR = 3e-4

    trainer = VAETrainer(INPUT_DIM, H_DIM, Z_DIM, BATCH_SIZE, LR, NUM_EPOCHS, DEVICE)
    trainer.train()


if __name__ == "__main__":
    main()
