import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from diffusers import UNet2DModel
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
    # Generate the beta values linearly from beta_start to beta_end
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


class DiffusionModel(nn.Module):
    def __init__(self, img_size, num_classes=10, class_emb_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DModel(
            sample_size=img_size,  # the target image resolution
            in_channels=1
            + class_emb_size,  # Additional input channels to accept the conditioning information (the class)
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "AttnUpBlock2D",
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x_t, t, class_label):
        # Shape of x:
        bs, ch, w, h = x_t.shape

        # class conditioning is right shape to add as additional input channel
        class_cond = self.class_emb(class_label).view(
            bs, self.class_emb.embedding_dim, 1, 1
        )
        class_cond = class_cond.expand(bs, self.class_emb.embedding_dim, w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x_t, class_cond), 1)  # (bs, 5, 28, 28)

        # Feed the UNet with net_input, time step t, and return the prediction
        return self.model(net_input, t).sample  # (bs, 1, 28, 28)

    def sample(self, n_sample: int, size, n_T, sched, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        y = torch.randint(0, 10, (n_sample,)).to(device)  # Generate random target labels

        bs, ch, w, h = x_i.shape

        # class conditioning is right shape to add as additional input channel
        class_cond = self.class_emb(y).view(
            bs, self.class_emb.embedding_dim, 1, 1
        )
        class_cond = class_cond.expand(bs, self.class_emb.embedding_dim, w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x_i, class_cond), 1)  

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(n_T-1, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.model(net_input, torch.tensor(i).to(device) / n_T).sample
            x_i = (
                sched.oneover_sqrta[i] * (x_i - eps * sched.mab_over_sqrtmab[i])
                + sched.sqrt_beta_t[i] * z
            )

        return x_i, y


def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:

    # ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)

    # ddpm = DDPM(eps_model=ContextUnet(1), betas=(1e-4, 0.02), n_T=1000)
    # ddpm.to(device)

    n_T = 1000
    criterion = nn.MSELoss()

    ddpm = DiffusionModel(img_size=28, num_classes=10).to(device)
    ddpm_sched = ddpm_schedules(1e-4, 0.02, 1000)

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
            t = torch.randint(0, n_T, (x.shape[0],), device=device)
            eps = torch.randn_like(x, device=device)

            x_t = (
                ddpm_sched.sqrtab[t, None, None, None] * x
                + ddpm_sched.sqrtmab[t, None, None, None] * eps
            )

            eps_pred = ddpm(x_t, t/n_T, labels)

            loss = criterion(eps, eps_pred)

            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            # Assuming `ddpm` is defined and initialized somewhere in your code
            xh, y = ddpm.sample(16, (1, 28, 28), n_T, ddpm_sched, device)

            # Convert xh to PIL images
            to_pil = ToPILImage()
            pil_images = [to_pil(img) for img in xh]

            # Plot the images using matplotlib subplots
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            axes = axes.flatten()

            for img, label, ax in zip(pil_images, y, axes):
                ax.imshow(img, cmap='gray')
                ax.set_title(str(label.item()))
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(f"Paras_ddpm_sample_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./Paras_ddpm_mnist.pth")


if __name__ == "__main__":
    train_mnist()
