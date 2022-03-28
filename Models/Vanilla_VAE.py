import torch
import torch.nn as nn
import torch.nn.functional as F
from libcpab import Cpab
from Constants import tess_size, is_determinstic
from Constants import device_T


"""
This is a vanilla VAE that learns transformations from
tumor to tumor - Once these transformations are learned
we apply them on the whole image by smoothing them out
Image Size (30x30)
"""
class Vanilla_VAE(nn.Module):
    def __init__(self, img_channels = 2, z_dim = 12):
        super(Vanilla_VAE, self).__init__()
 
        
        self.T = Cpab(tess_size=tess_size, backend='pytorch', device=device_T, zero_boundary=True)
        
        # encoder conv layers
        self.convE1 = nn.Conv2d(img_channels, out_channels = 64, kernel_size = 3) #output size (28)
        self.convE2 = nn.Conv2d(64, out_channels = 32, kernel_size = 3) #output size (26)
        self.convE3 = nn.Conv2d(32, out_channels = 16, kernel_size = 3) #output size (24)
        self.convE4 = nn.Conv2d(16, out_channels = 16, kernel_size = 3) #output size (22)
        self.convE5 = nn.Conv2d(16, out_channels = 8, kernel_size = 3) #output size (20)
        
        #Encode fully connected layers
        self.FCE1 = nn.Linear(in_features=20*20*8, out_features=128)
        self.FCE2 = nn.Linear (in_features=128, out_features = z_dim ) #mu 
        self.FCE3 = nn.Linear (in_features=128, out_features = z_dim ) #var

        
        # decoder full connected layers
        self.FCD1 = nn.Linear(in_features=z_dim, out_features=2400)
        self.FCD2 = nn.Linear(in_features=2400, out_features=2000)
        self.FCD3 = nn.Linear (in_features=2000, out_features = 1800)
        self.FCD4 = nn.Linear (in_features=1800, out_features = 1700)
        self.FCD5 = nn.Linear(in_features=1700, out_features=1598)

        

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode (self, x):

        #x is concatenated two images
        x_out = self.convE1 (x)
        x_out = F.leaky_relu(x_out)

        x_out = self.convE2 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE4 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE5 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = x_out.view(x_out.size(0), -1)


        x_out = self.FCE1 (x_out)
        x_out = F.leaky_relu (x_out)
        
        if not is_determinstic:
            mu = self.FCE2 (x_out)
            log_var = self.FCE3 (x_out)

            z = self.reparameterize (mu, log_var)
            return z, mu, log_var

        else:
            
            mu = self.FCE2 (x_out)
            return mu

    def decode (self, x):

        x_out = self.FCD1 (x)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.FCD2 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.FCD3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.FCD4 (x_out)
        x_out = F.leaky_relu (x_out)

        theta = self.FCD5 (x_out)

        return theta
 
    def forward(self, x):

        # encoding
        if not is_determinstic:
            z, mu, log_var = self.encode (x)
        else:
            z = self.encode (x)
        
        theta = self.decode (z)
            
        reconstruction = self.T.transform_data(x[:,0,:,:].unsqueeze(1), theta, outsize=x[:,0,:,:].unsqueeze(1).size()[2:], return_velocities=False)
        
        if not is_determinstic:
            return reconstruction, mu, log_var, z
        return reconstruction