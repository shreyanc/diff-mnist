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

from basic_unet import ContextUnet, ddpm_schedules

class DiffusionModel(nn.Module):
    # SHREYAN 2 - change class emb size to 128
    def __init__(self, img_size, betas, n_T, num_classes=10, class_emb_size=128):
        super().__init__()

        self.betas = betas
        self.n_T = n_T
        self.criterion = nn.MSELoss()


        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

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

        # self.model = ContextUnet(1)

    def forward(self, x, t, labels):

        eps = torch.randn_like(x)

        x_t = (
            self.sqrtab[t, None, None, None] * x
            + self.sqrtmab[t, None, None, None] * eps
        ) 

        bs, ch, w, h = x_t.shape
        class_cond = self.class_emb(labels).view(
            bs, self.class_emb.embedding_dim, 1, 1
        )
        class_cond = class_cond.expand(bs, self.class_emb.embedding_dim, w, h)
        net_input = torch.cat((x_t, class_cond), 1)  
        # return self.model(net_input, t).sample  

        return self.criterion(eps, self.model(net_input, t / self.n_T).sample)
    
        # return self.model(x, t, class_label)
        # _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
        #     x.device
        # )  # t ~ Uniform(0, n_T)
        # eps = torch.randn_like(x)  # eps ~ N(0, 1)

        # x_t = (
        #     self.sqrtab[_ts, None, None, None] * x
        #     + self.sqrtmab[_ts, None, None, None] * eps
        # )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # # We should predict the "error term" from this x_t. Loss is what we return.

        # return self.criterion(eps, self.model(x_t, _ts / self.n_T, labels))
    
    
    def sample_image_multiple(self, n_samples, size, epoch):
        # with torch.no_grad():
        #     x = torch.randn(num_samples, 1, 28, 28).to(device)
        #     y = torch.randint(0, 10, (num_samples,)).to(device)

        #     for t in range(len(alphabar_t) - 1, -1, -1):
        #         z = torch.randn(num_samples, 1, 28, 28).to(device) if t > 0 else 0
        #         alpha = alpha_t[t]
        #         alphabar = alphabar_t[t]
        #         beta = beta_t[t]

        #         predicted_noise = model(
        #             x, torch.tensor([t/num_steps]).expand(num_samples).to(device), y
        #         )

        #         x = (1 / torch.sqrt(alpha)) * (
        #             x - (beta / torch.sqrt(1 - alphabar)) * predicted_noise
        #         ) + torch.sqrt(beta) * z

        #     generated_samples = x.cpu().squeeze().numpy()

        self.model.eval()
        with torch.no_grad():
            x_i = torch.randn(n_samples, *size).to(device)  # x_T ~ N(0, 1)
            y = torch.randint(0, 10, (n_samples,)).to(device)  # Generate random target labels

            bs, ch, w, h = x_i.shape
            class_cond = self.class_emb(y).view(
                    bs, self.class_emb.embedding_dim, 1, 1
                )
            class_cond = class_cond.expand(bs, self.class_emb.embedding_dim, w, h)

            # This samples accordingly to Algorithm 2. It is exactly the same logic.
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_samples, *size).to(device) if i > 1 else 0
                
                net_input = torch.cat((x_i, class_cond), 1) 
                eps = self.model(net_input, torch.tensor(i).to(device) / self.n_T).sample
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

        # return x_i, y

        # SHREYAN 1 - change plotting code
        fig, axes = plt.subplots(4, 4, figsize=(4, 4))
        axes = axes.flatten()

        for img, label, ax in zip(x_i, y, axes):
            ax.imshow(img.squeeze().cpu().numpy(), cmap='gray')
            ax.set_title(str(label.item()))
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"paras_atomic_sample_{epoch}.png", dpi=100)

# num_steps = 1000
# beta_start = 0.0001
# beta_end = 0.02


# # SHREYAN 3 - copy paste alpha beta definitions from working code (and rename following uses)
# beta_t = torch.linspace(beta_start, beta_end, num_steps).to(device)
# sqrt_beta_t = torch.sqrt(beta_t)

# alpha_t = 1 - beta_t
# log_alpha_t = torch.log(alpha_t)
# alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

# sqrtab = torch.sqrt(alphabar_t)
# oneover_sqrta = 1 / torch.sqrt(alpha_t)

# sqrtmab = torch.sqrt(1 - alphabar_t)
# mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

betas=(1e-4, 0.02)
n_T=1000

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
)

trainset = datasets.MNIST("data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# SHREYAN 0 - change learning rate to 2e-4
model = DiffusionModel(img_size=(28,28), betas=betas, n_T=n_T, num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)



def train(epochs=100):
    loss_ema = None
    for epoch in range(epochs):
        i = 0
        model.train()
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            t = torch.randint(0, n_T, (data.shape[0],), device=device)

            # alphabar = alphabar_t[t]

            # noise = torch.randn_like(data)
            # noisy_data = (
            #     torch.sqrt(alphabar).view(-1, 1, 1, 1) * data
            #     + torch.sqrt(1 - alphabar).view(-1, 1, 1, 1) * noise
            # )

            # # SHREYAN 4 - normalize time
            # predicted_noise = model(noisy_data, t/num_steps, labels)

            # # SHREYAN 5 - fix loss computation order
            # loss = criterion(noise, predicted_noise)

            loss = model(data, t, labels)

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

        model.sample_image_multiple(n_samples=16, size=(1, 28,28), epoch=epoch)


if __name__ == "__main__":
    train()
