import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import time
from torch.cuda.amp import GradScaler, autocast

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
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label since we don't have labels


class ConvolVariatinalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=32, z_dim=2, output_channels=1, im_size=(512, 512)):
        super().__init__()

        # encoder 
        self.conv1 = nn.Conv2d(input_dim, h_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(h_dim)
        self.conv2 = nn.Conv2d(h_dim, h_dim*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(h_dim*2)
        self.conv3 = nn.Conv2d(h_dim*2, h_dim*4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(h_dim*4)
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(int(h_dim * im_size[0]/4 * im_size[1]/4), z_dim*2)
        self.fc_mu = nn.Linear(z_dim*2, z_dim)
        self.fc_logvar = nn.Linear(z_dim*2, z_dim)
        
        # decoder
        self.fc_dec = nn.Linear(z_dim, int(h_dim * im_size[0]/4 * im_size[1]/4))
        self.deconv1 = nn.ConvTranspose2d(h_dim*4, h_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(h_dim*2)
        self.deconv2 = nn.ConvTranspose2d(h_dim*2, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(h_dim)
        self.deconv3 = nn.ConvTranspose2d(h_dim, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(output_channels)

    def encode(self, x):
            
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
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
        x = x.view(x.size(0), -1, int(im_size[0]/8), int(im_size[1]/8)) 
        x = torch.relu(self.bn4(self.deconv1(x)))
        x = torch.relu(self.bn5(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



if __name__ == '__main__':
    BATCH_SIZE = 1024
    NUM_EPOCHS = 1000

    Z_DIMS = [32]
    H_DIMS = [32, 64, 128]

    final_Losses = []

    final_Losses = []
    im_size = (256, 256)
    for Z_DIM in Z_DIMS:
        for H_DIM in H_DIMS:
            if Z_DIM == H_DIM or Z_DIM > H_DIM: 
                pass
            else:   
                print(f"Training model with Z_DIM={Z_DIM}, H_DIM={H_DIM}")
                # Load RGB images from the "Figures" folder
                transform = transforms.Compose([
                    transforms.Resize(im_size), 
                    transforms.ToTensor()
                ])
                dataset = CustomImageDataset(root_dir='./data/Waves4', transform=transform)
                train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = ConvolVariatinalAutoEncoder(input_dim=1, h_dim=H_DIM, z_dim=Z_DIM, output_channels=1, im_size=im_size).to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
                
                for epoch in range(NUM_EPOCHS):
                    model.train()
                    train_loss = 0

                    # Time calculation
                    start = time.time()

                    for x, _ in train_loader:
                        x = x.to(device)
                        x_reconstructed, mu, logvar = model(x)
                        reconstructed_loss = loss_fn(x_reconstructed, x)
                        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = reconstructed_loss + kl_div

                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                    print("Time: {}".format(time.time() - start))
                    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')
                    final_Losses.append(train_loss / len(train_loader.dataset))
                # Save the model
                torch.save(model.state_dict(), 'cnn_vae_model_Z{}_H{}.pth'.format(Z_DIM, H_DIM))
                print('Model saved to cnn_vae_model_Z{}_H{}.pth'.format(Z_DIM, H_DIM))