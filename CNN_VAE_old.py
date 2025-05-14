import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision.utils import save_image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = []
        for fname in os.listdir(self.root_dir):
            if fname.endswith('.png'):
                img_path = os.path.join(self.root_dir, fname)
                try:
                    # Try to open the image to ensure it's valid
                    with Image.open(img_path) as img:
                        img.verify()
                    image_paths.append(img_path)
                except (IOError, SyntaxError) as e:
                    print(f"Skipping invalid image file: {img_path}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label since we don't have labels


class ConvolVariatinalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=32, z_dim=2, output_channels=3, im_size=(512, 512)):
        super().__init__()
        # encoder 
        self.conv1 = nn.Conv2d(input_dim, h_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(h_dim, h_dim * 2, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(h_dim * 2 * 64 * 64, 2 * z_dim)  # Adjusted for 32x32 images
        self.fc_mu = nn.Linear(2 * z_dim, z_dim)
        self.fc_logvar = nn.Linear(2 * z_dim, z_dim)
        
        # decoder
        self.fc_dec = nn.Linear(z_dim, h_dim * 2 * 64 * 64)  # Adjusted for 32x32 images
        self.deconv1 = nn.ConvTranspose2d(h_dim * 2, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(h_dim, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        print("Encoder")
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc_enc(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = torch.relu(self.fc_dec(z))
        x = x.view(x.size(0), -1, 64, 64)  # Adjusted for 128x128 images
        x = torch.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == '__main__':
    BATCH_SIZE = 1024
    NUM_EPOCHS = 1000

    Z_DIMS = [32]
    H_DIMS = [32, 64, 128]

    final_Losses = []

    for Z_DIM in Z_DIMS:
        for H_DIM in H_DIMS:
            if Z_DIM == H_DIM or Z_DIM > H_DIM: 
                pass
            else:   
                print(f"Training model with Z_DIM={Z_DIM}, H_DIM={H_DIM}")
                # Load RGB images from the "Figures" folder
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),  # Resize images to 32x32
                    transforms.ToTensor()
                ])    
                dataset = CustomImageDataset(root_dir=r'C:\Users\jung_\OneDrive\Documentos\Poli\TPN\Python\Tcc\DeepLearning\data\Waves2', transform=transform)
                train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = ConvolVariatinalAutoEncoder(input_dim=3, h_dim=H_DIM, z_dim=Z_DIM, output_channels=3).to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                loss_fn = nn.BCELoss(reduction='sum')
                
                for epoch in range(NUM_EPOCHS):
                    model.train()
                    train_loss = 0
                    for x, _ in train_loader:
                        x = x.to(device)
                        x_reconstructed, mu, logvar = model(x)
                        # compute loss
                        reconstructed_loss = loss_fn(x_reconstructed, x)
                        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = reconstructed_loss + kl_div

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')
                    final_Losses.append(train_loss / len(train_loader.dataset))
                # Save the model
                torch.save(model.state_dict(), 'cnn_vae_model_Z{}_H{}.pth'.format(Z_DIM, H_DIM))
                print('Model saved to cnn_vae_model_Z{}_H{}.pth'.format(Z_DIM, H_DIM))
    