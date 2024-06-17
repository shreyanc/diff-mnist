import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from diffusers import UNet2DModel

import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionModel(nn.Module):
    # SHREYAN 2 - change class emb size to 128
    def __init__(self, img_size, num_classes=10, class_emb_size=128):
        super().__init__()

        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DModel(
            sample_size=img_size,
            in_channels=1 + class_emb_size,
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

    def forward(self, x, t, class_label):
        bs, ch, w, h = x.shape
        class_cond = self.class_emb(class_label).view(
            bs, self.class_emb.embedding_dim, 1, 1
        )
        class_cond = class_cond.expand(bs, self.class_emb.embedding_dim, w, h)
        net_input = torch.cat((x, class_cond), 1)  
        return self.model(net_input, t).sample  

num_steps = 1000
beta_start = 0.0001
beta_end = 0.02


# SHREYAN 3 - copy paste alpha beta definitions from working code (and rename following uses)
beta_t = torch.linspace(beta_start, beta_end, num_steps).to(device)
sqrt_beta_t = torch.sqrt(beta_t)

alpha_t = 1 - beta_t
log_alpha_t = torch.log(alpha_t)
alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

sqrtab = torch.sqrt(alphabar_t)
oneover_sqrta = 1 / torch.sqrt(alpha_t)

sqrtmab = torch.sqrt(1 - alphabar_t)
mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab


class RescaleTransform(object):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        return (self.max_val - self.min_val) * tensor + self.min_val


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
)

trainset = datasets.MNIST("data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# SHREYAN 0 - change learning rate to 2e-4
model = DiffusionModel(img_size=(28,28), num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.MSELoss()


def sample_image_multiple(num_samples, epoch):
    with torch.no_grad():
        x = torch.randn(num_samples, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (num_samples,)).to(device)

        for t in range(len(alphabar_t) - 1, -1, -1):
            z = torch.randn(num_samples, 1, 28, 28).to(device) if t > 0 else 0
            alpha = alpha_t[t]
            alphabar = alphabar_t[t]
            beta = beta_t[t]

            predicted_noise = model(
                x, torch.tensor([t]).expand(num_samples).to(device), y
            )

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alphabar)) * predicted_noise
            ) + torch.sqrt(beta) * z

        generated_samples = x.cpu().squeeze().numpy()

    # SHREYAN 1 - change plotting code
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    axes = axes.flatten()

    for img, label, ax in zip(generated_samples, y, axes):
        ax.imshow(img, cmap='gray')
        ax.set_title(str(label.item()))
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"paras_atomic_sample_{epoch}.png", dpi=100)



def train(epochs=100):
    loss_ema = None
    for epoch in range(epochs):
        i = 0
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            t = torch.randint(0, num_steps, (data.shape[0],), device=device)

            alphabar = alphabar_t[t]

            noise = torch.randn_like(data)
            noisy_data = (
                torch.sqrt(alphabar).view(-1, 1, 1, 1) * data
                + torch.sqrt(1 - alphabar).view(-1, 1, 1, 1) * noise
            )

            # SHREYAN 4 - normalize time
            predicted_noise = model(noisy_data, t/num_steps, labels)

            # SHREYAN 5 - fix loss computation
            loss = criterion(noise, predicted_noise)

            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            optimizer.step()

            if (i % 50) == 0:
                print(
                    f"Epoch [{epoch+1}/1000], i: {i}, Loss: {loss.item():.4f}, Running loss: {loss_ema}"
                )

            i += 1

        sample_image_multiple(num_samples=16, epoch=epoch)


if __name__ == "__main__":
    train()
