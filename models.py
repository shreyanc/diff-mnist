import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusers import UNet2DModel
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseModel_DiffusersUnet(nn.Module):
    def __init__(self, class_emb_size=129):
        super().__init__()
        self.class_emb = nn.Embedding(10, class_emb_size)
        self.nmodel = UNet2DModel(
            sample_size=(28, 28),
            in_channels=1+class_emb_size,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x_t, t, c):
        # Shape of x:
        bs, ch, w, h = x_t.shape

        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(c)  # Map to embedding dimension
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x_t, class_cond), 1)  # (bs, 5, 28, 28)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.nmodel(net_input, t).sample


class DiffusionModelTrainer:
    def __init__(self, beta1, beta2, n_T, noise_model_name, lr=2e-04, bs=128):
        self.betas = torch.linspace(beta1, beta2, n_T).to(device)

        self.sqrt_betas = torch.sqrt(self.betas)

        self.alphas = 1 - self.betas
        self.log_alphas = torch.log(self.alphas)
        self.alphabars = torch.cumsum(self.log_alphas, dim=0).exp()

        self.sqrt_alphabars = torch.sqrt(self.alphabars)
        self.inv_sqrt_alphas = 1 / torch.sqrt(self.alphas)

        self.sqrt_oneminus_alphabars = torch.sqrt(1 - self.alphabars)

        if noise_model_name == "diffusers_unet":
            self.noise_model = NoiseModel_DiffusersUnet().to(device)
        else:
            raise ValueError(
                f"noise model name must be in ['diffusers_unet'], is {noise_model_name}"
            )

        self.criterion = nn.MSELoss()
        self.n_T = n_T

        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
        )

        dataset = MNIST(
            "./data",
            train=True,
            download=True,
            transform=tf,
        )
        self.dataloader = DataLoader(
            dataset, batch_size=bs, shuffle=True, num_workers=20
        )
        self.optim = torch.optim.Adam(self.noise_model.parameters(), lr=lr)

    def train_step(self, x, c):
        noise = torch.randn_like(x, device=device)
        t = torch.randint(1, self.n_T, (x.shape[0],), device=device)
        t_norm = t / self.n_T

        x_t = (
            self.sqrt_alphabars[t].reshape(-1, 1, 1, 1) * x
            + self.sqrt_oneminus_alphabars[t].reshape(-1, 1, 1, 1) * noise
        )

        loss = self.criterion(noise, self.noise_model(x_t, t_norm, c))
        loss.backward()

        return loss.item()
    
    def sample(self, num_samples):
        x_i = torch.randn(num_samples, 1, 28, 28).to(device)  # x_T ~ N(0, 1)
        y = torch.randint(0, 10, (num_samples,)).to(device)  # Generate random target labels


        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for t in range(self.n_T-1, 0, -1):
            z = torch.randn(num_samples, 1, 28, 28).to(device) if t > 1 else 0
            eps = self.noise_model(x_i, torch.tensor(t).to(device) / self.n_T, y)
            x_i = (
                self.inv_sqrt_alphas[t] * (x_i - eps * self.betas[t]/self.sqrt_oneminus_alphabars[t])
                + self.sqrt_betas[t] * z
            )

        return x_i, y

    def train(self, n_epochs=100):
        for epoch in range(n_epochs):
            self.noise_model.train()

            pbar = tqdm(self.dataloader)
            loss_ema = None

            for x, labels in pbar:
                self.optim.zero_grad()

                x = x.to(device)
                labels = labels.to(device)

                loss = self.train_step(x, labels)

                if loss_ema is None:
                    loss_ema = loss
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss
                pbar.set_description(f"loss: {loss_ema:.4f}")
                self.optim.step()

            self.noise_model.eval()
            with torch.no_grad():
                xh, y = self.sample(16)

                images = [img.squeeze().cpu().numpy() for img in xh]

                # Plot the images using matplotlib subplots
                fig, axes = plt.subplots(4, 4, figsize=(4, 4))
                axes = axes.flatten()

                for img, label, ax in zip(images, y, axes):
                    ax.imshow(img, cmap='gray')
                    ax.set_title(str(label.item()))
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(f"trainer_code_sample_{epoch}.png", dpi=100)




if __name__ == "__main__":
    trainer = DiffusionModelTrainer(
        beta1=1e-04,
        beta2=0.02,
        n_T=1000,
        noise_model_name="diffusers_unet",
        lr=2e-04,
        bs=128,
    )

    trainer.train(n_epochs=100)
