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



class DiffusionModel(nn.Module):
    def __init__(self, img_size, num_classes=10, class_emb_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size 
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DModel(
            sample_size=img_size,            # the target image resolution
            in_channels=1 + class_emb_size,  # Additional input channels to accept the conditioning information (the class)
            out_channels=1,            # the number of output channels
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
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_label):
        # Shape of x:
        bs, ch, w, h = x.shape
    
        # class conditioning is right shape to add as additional input channel
        class_cond = self.class_emb(class_label).view(bs, self.class_emb.embedding_dim, 1, 1)
        class_cond = class_cond.expand(bs, self.class_emb.embedding_dim, w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)
    
        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)
    
        # Feed the UNet with net_input, time step t, and return the prediction
        return self.model(net_input, t).sample # (bs, 1, 28, 28)
    

def sample_image_multiple(model, n_samples, size, epoch):
        model.eval()
        with torch.no_grad():
            x_i = torch.randn(n_samples, *size).to(device)  # x_T ~ N(0, 1)
            y = torch.randint(0, 10, (n_samples,)).to(device)  # Generate random target labels

            bs, ch, w, h = x_i.shape
            class_cond = model.class_emb(y).view(
                    bs, model.class_emb.embedding_dim, 1, 1
                )
            class_cond = class_cond.expand(bs, model.class_emb.embedding_dim, w, h)

            for i in range(num_steps, 0, -1):
                z = torch.randn(n_samples, *size).to(device) if i > 1 else 0
                
                net_input = torch.cat((x_i, class_cond), 1) 
                eps = model(net_input, torch.tensor(i).to(device) / num_steps).sample
                x_i = (
                    oneover_sqrta[i] * (x_i - eps * mab_over_sqrtmab_inv[i])
                    + sqrt_beta_t[i] * z
                )

        fig, axes = plt.subplots(4, 4, figsize=(4, 4))
        axes = axes.flatten()

        for img, label, ax in zip(x_i, y, axes):
            ax.imshow(img.squeeze().cpu().numpy(), cmap='gray')
            ax.set_title(str(label.item()))
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"paras_corrected_sample_{epoch}.png", dpi=100)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
)

trainset = datasets.MNIST("data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

model = DiffusionModel(img_size=(28,28), num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.MSELoss()

def train(epochs=100):
    loss_ema = None
    for epoch in range(epochs):
        i = 0
        model.train()
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Randomly generate timesteps for each example in the batch
            t = torch.randint(0, num_steps, (data.shape[0],), device=device)
            
            # Retrieve the corresponding alpha_bar_t values for each example in the batch
            alpha_bar_t = alphabar_t[t]
            
            # Generate random noise (mean 0, sd 1) tensor with the same shape as input data
            noise = torch.randn_like(data)
            
            # Calculate the noisy data using the reparameterization trick
            noisy_data = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1) * data + torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1) * noise
            
            predicted_noise = model(noisy_data, t, labels)
            loss = criterion(predicted_noise, noisy_data - data)
            
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            
            optimizer.step()
            
            if (i % 50) == 0:
                print(f"Epoch [{epoch+1}/1000], i: {i}, Loss: {loss.item():.4f}, Running loss: {loss_ema}")
            
            i += 1
        
        sample_image_multiple(model, n_samples=16, size=(1, 28,28), epoch=epoch)


if __name__=='__main__':
    train()