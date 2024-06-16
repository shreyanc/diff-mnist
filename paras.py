import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from diffusers import UNet2DModel, UNet2DConditionModel
from torchvision.transforms import ToPILImage

from matplotlib import pyplot as plt

# import matplotlib as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from types import SimpleNamespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ddpm_schedules(beta1: float, beta2: float, T: int):
    beta_t = torch.linspace(beta1, beta2, T).to(device)
    sqrt_beta_t = torch.sqrt(beta_t)

    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return SimpleNamespace(
        alpha_t=alpha_t,
        oneover_sqrta=oneover_sqrta,
        sqrt_beta_t=sqrt_beta_t,
        alphabar_t=alphabar_t,
        sqrtab=sqrtab,
        sqrtmab=sqrtmab,
        mab_over_sqrtmab=mab_over_sqrtmab_inv,
    )



class EpsModelDiffusersConditional(nn.Module):
    """
    Not working
    """
    def __init__(self, label_emb_size=128):
        super().__init__()
        self.model = UNet2DConditionModel(
            sample_size=(28, 28),
            in_channels=1,
            out_channels=1,
        )

        self.class_embed = nn.Embedding(10, 128)  # Assuming 10 classes and embedding size of 128

    def forward(self, x, t, class_labels):
        batch_size, _, height, width = x.shape
        class_embeds = self.class_embed(class_labels)
        class_embeds = class_embeds.unsqueeze(1).unsqueeze(2)  # Shape (batch_size, 1, 1, embedding_dim)
        encoder_hidden_states = class_embeds.expand(batch_size, height, width, -1)
        encoder_hidden_states = encoder_hidden_states.reshape(batch_size, -1, 128)  # Reshape to (batch_size, sequence_length, embedding_dim)
        return self.model(x, t, encoder_hidden_states)



class EpsModelDiffusers(nn.Module):
    def __init__(self, class_emb_size=128):
        super().__init__()
        self.class_emb = nn.Embedding(10, class_emb_size)
        self.model = UNet2DModel(
            sample_size=(28,28),
            in_channels=1+class_emb_size,
            out_channels=1,
            layers_per_block=2,        # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",          # a regular ResNet downsampling block
                "AttnDownBlock2D",      # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",        # a ResNet upsampling block with spatial self-attention
                "AttnUpBlock2D",
                "UpBlock2D",            # a regular ResNet upsampling block
            )
        )

    def forward(self, x, t, class_labels):
        bs, _, h, w = x.shape
        class_cond = self.class_emb(class_labels).view(bs, self.class_emb.embedding_dim, 1, 1)
        class_cond = class_cond.expand(bs, self.class_emb.embedding_dim, h, w)
        net_input = torch.cat((x, class_cond), 1)
        return self.model(net_input, t).sample


class DDPMConditional(nn.Module):
    def __init__(self, eps_model, betas, n_T):
        super().__init__()

        self.img_size = [28, 28]
        self.betas = betas
        self.n_T = n_T

        self.criterion = nn.MSELoss()

        self.eps_model = eps_model

        self.scheds = ddpm_schedules(self.betas[0], self.betas[1], self.n_T)

    def forward(self, x_0, class_label):
        t = torch.randint(0, self.n_T, (x_0.shape[0],), device=device)
        eps = torch.randn_like(x_0, device=device)

        x_t = (
            self.scheds.sqrtab[t, None, None, None] * x_0
            + self.scheds.sqrtmab[t, None, None, None] * eps
        )

        eps_pred = self.eps_model(x_t, t / self.n_T, class_label)

        loss = self.criterion(eps, eps_pred)

        return loss

    def sample(self, n_samples: int) -> torch.Tensor:

        x_t = torch.randn(n_samples, 1, *self.img_size).to(device)  # x_T ~ N(0, 1)
        y = torch.randint(0, 10, (n_samples,)).to(device)  # Generate random target labels

        for t in range(self.n_T-1, 0, -1):
            z = torch.randn(n_samples, 1, *self.img_size).to(device) if t > 1 else 0
            eps_pred = self.eps_model(x_t, torch.tensor(t).to(device) / self.n_T, y)
            x_t = (
                self.scheds.oneover_sqrta[t]
                * (x_t - eps_pred * self.scheds.mab_over_sqrtmab[t])
                + self.scheds.sqrt_beta_t[t] * z
            )
        
        return x_t, y


def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:

    ddpm = DDPMConditional(eps_model=EpsModelDiffusers(), betas=(1e-4, 0.02), n_T=1000)
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
        for x, labels in pbar:
            optim.zero_grad()
            x = x.to(device)
            labels = labels.to(device)

            loss = ddpm(x, labels)

            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh, y = ddpm.sample(16)
            # grid = make_grid(xh, nrow=4)
            # save_image(grid, f"ContextUnet_ddpm_sample_{i}.png")

            images = [img.squeeze().cpu().numpy() for img in xh]

            # Plot the images using matplotlib subplots
            fig, axes = plt.subplots(4, 4, figsize=(4, 4))
            axes = axes.flatten()

            for img, label, ax in zip(images, y, axes):
                ax.imshow(img, cmap='gray')
                ax.set_title(str(label.item()))
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(f"CondDiffusersUnet_ddpm_sample_{i}.png", dpi=100)

            # save model
            torch.save(ddpm.state_dict(), f"./CondDiffusersUnet_ddpm_mnist.pth")


if __name__ == "__main__":
    train_mnist()
