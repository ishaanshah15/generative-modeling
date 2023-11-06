import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        """
        TODO 2.1 : Fill in self.convs following the given architecture
        """

        self.convs = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )

        

        #TODO 2.1: fill in self.fc, such that output dimension is self.latent_dim
        self.input_dim = self.convs(torch.randn((1,*input_shape))).flatten().shape[0]
        self.fc = nn.Linear(self.input_dim,self.latent_dim)

    def forward(self, x):
        #TODO 2.1 : forward pass through the network, output should be of dimension : self.latent_dim
        conv_out = self.convs(x)
        linear_in = conv_out.reshape(conv_out.shape[0],-1)
        return self.fc(linear_in)

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        #TODO 2.4: fill in self.fc, such that output dimension is 2*self.latent_dim

        self.fc = nn.Linear(self.input_dim,2*latent_dim)

    def forward(self, x):
        #TODO 2.4: forward pass through the network.
        # should return a tuple of 2 tensors, mu and log_std
        conv_out = self.convs(x)
        linear_in = conv_out.reshape(conv_out.shape[0],-1)
        
        out = self.fc(linear_in)
        mu = out[:,:self.latent_dim]
        log_std = out[:,self.latent_dim:]
        return mu, log_std


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        #TODO 2.1: fill in self.base_size
        self.base_size = (256,output_shape[1]//8,output_shape[2]//8)

        linear_dim = torch.randn(self.base_size).flatten().shape[0]
        
        self.fc = nn.Linear(latent_dim,linear_dim)

        """
        TODO 2.1 : Fill in self.deconvs following the given architecture
       
        """

        self.deconvs =  nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )

    def forward(self, z):
        #TODO 2.1: forward pass through the network, first through self.fc, then self.deconvs.
        out = self.fc(z)
       
        out = out.reshape((z.shape[0],*self.base_size))

        out = self.deconvs(out)

        return out

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    #NOTE: You don't need to implement a forward function for AEModel. For implementing the loss functions in train.py, call model.encoder and model.decoder directly.
