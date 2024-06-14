import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from diffusers import UNet2DModel
# import matplotlib as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ddpm_schedules(beta1: float, beta2: float, T: int):
    # Generate the beta values linearly from beta_start to beta_end
    beta = torch.linspace(beta1, beta2, T).to(device)
    alpha = 1 - beta
    # Pre-calculate alpha_bar_t for all timesteps
    alpha_bar = torch.cumprod(alpha, dim=0)



def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:

    # ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)

    ddpm = DDPM(eps_model=ContextUnet(1), betas=(1e-4, 0.02), n_T=1000)
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"ContextUnet_ddpm_sample_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ContextUnet_ddpm_mnist.pth")
