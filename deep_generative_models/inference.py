import tomli
import torch
import matplotlib.pyplot as plt
from deep_generative_models.model_cnn import VAE
from config.paths import IMAGES, STORAGE, CELL_DATA, TRAIN_CONFIG
from deep_generative_models.dataset import create_dataloader

with open(TRAIN_CONFIG, "rb") as f:
    config = tomli.load(f)


class VAEInference:
    def __init__(self, model_path, input_dim, z_dim, device):
        self.device = device
        self.input_dim = input_dim
        self.model = VAE(input_dim=input_dim, latent_dim=z_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.z_dim = z_dim

    def load_data(self, batch_size):
        brains = ["B20"]  # Validation brains
        tile_size = self.input_dim
        tiles_per_epoch = 10  # Adjust as needed
        dataloader = create_dataloader(
            CELL_DATA, brains, tile_size, batch_size, tiles_per_epoch, num_workers=0
        )
        return dataloader

    def inference(self, batch_size=10):
        dataloader = self.load_data(batch_size)
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)
                x_reconstructed, _, _ = self.model(x)
                self.plot_reconstructed_images(x, x_reconstructed)
                break

    def sample(self, num_samples=10):
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim).to(self.device)
            samples = self.model.decoder(z)
            self.plot_generated_images(samples)

    def plot_reconstructed_images(self, original, reconstructed):
        original = original.view(-1, self.input_dim, self.input_dim).cpu().numpy()
        reconstructed = (
            reconstructed.view(-1, self.input_dim, self.input_dim).cpu().numpy()
        )
        fig, axes = plt.subplots(2, len(original), figsize=(15, 3))
        for i in range(len(original)):
            axes[0, i].imshow(original[i], cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].imshow(reconstructed[i], cmap="gray")
            axes[1, i].axis("off")
        plt.savefig(IMAGES / "reconstructed_images.png")
        plt.close()

    def plot_generated_images(self, samples):
        samples = samples.view(-1, self.input_dim, self.input_dim).cpu().numpy()
        fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
        for i in range(len(samples)):
            axes[i].imshow(samples[i], cmap="gray")
            axes[i].axis("off")
        plt.savefig(IMAGES / "generated_images.png")
        plt.close()


def main():
    DEVICE = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "cpu" if torch.backends.mps.is_available() else "cpu"
        )  # @TODO fix keep cpu for now needs file changes
    )

    print(DEVICE)
    INPUT_DIM = config["model"]["input_dim"]  # Put into config
    Z_DIM = config["model"]["latent_dim"]  # Put into config
    MODEL_PATH = STORAGE / "trained_model_VAE_FLO_20241126_143813.pth"

    inference = VAEInference(MODEL_PATH, INPUT_DIM, Z_DIM, DEVICE)
    inference.inference()
    inference.sample()


if __name__ == "__main__":
    main()
