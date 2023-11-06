import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class UpSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers

        self.scale_factor = upscale_factor
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=1, padding=padding)
        


    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel wise upscale_factor^2 times
        # 2. Use pixel shuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)
        # to form a (batch x channel x height*upscale_factor x width*upscale_factor) output
        # 3. Apply convolution and return output

        #shape = list(x.shape)
        #shape[1] = shape[1]*(self.scale_factor**2)
        #x = nn.Upsample(size=tuple(shape))(x)
        
        x = x.repeat_interleave(int(self.scale_factor**2),dim=1)
        x = self.pixel_shuffle(x)
        return self.conv(x)




class DownSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers

        self.scale_factor = downscale_ratio
        self.pixel_unshuffle = nn.PixelUnshuffle(self.scale_factor)
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=1,padding=padding)
        

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use pixel unshuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle)
        # to form a (batch x channel * downscale_factor^2 x height x width) output
        # 2. Then split channel wise into (downscale_factor^2xbatch x channel x height x width) images
        # 3. Average across dimension 0, apply convolution and return output
        x = self.pixel_unshuffle(x)
        x = x.permute(1,0,2,3)
        x = x.reshape(int(self.scale_factor**2),-1,x.shape[1],x.shape[2],x.shape[3])
        x = x.permute(0,2,1,3,4)
        x = torch.mean(x,dim=0)
        return self.conv(x)


class ResBlockUp(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        # TODO 1.1: Setup the network layers

        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(3,3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            UpSampleConv2D(n_filters,kernel_size=3,n_filters=n_filters,padding=1)
        )

        self.upsample_residual = UpSampleConv2D(input_channels,kernel_size=1,n_filters=n_filters,padding=0)


    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Make sure to upsample the residual before adding it to the layer output.
        layer_out = self.layers(x)
        conv_out = self.upsample_residual(x) #TODO: Check this structure is correct
        return conv_out + layer_out

class ResBlockDown(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        # TODO 1.1: Setup the network layers

        self.layers = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(input_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(),
                    DownSampleConv2D(n_filters,n_filters=n_filters,kernel_size=3,padding=1),
                    nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        

        self.downsample_residual = DownSampleConv2D(input_channels,n_filters=n_filters,kernel_size=1,padding=0)
            

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through self.layers and implement a residual connection.
        # Make sure to downsample the residual before adding it to the layer output.
        layer_out = self.layers(x)
        conv_out = self.downsample_residual(x)
        return layer_out + conv_out #TODO: Check this is the correct structure


class ResBlock(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        # TODO 1.1: Setup the network layers

        self.layers = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(input_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        
        self.conv1x1 = nn.Identity()
        if n_filters != input_channels:
            self.conv1x1 = nn.Conv2d(input_channels, n_filters, kernel_size=1)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        layer_out = self.layers(x)
        conv_out = self.conv1x1(x)
        return layer_out + conv_out


class Generator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        # TODO 1.1: Setup the network layers
        self.starting_image_size = starting_image_size
        self.latent_dim = 128
        self.dense = nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim*(self.starting_image_size**2))
        self.layers = nn.Sequential(ResBlockUp(input_channels=self.latent_dim,n_filters=self.latent_dim),
                                    ResBlockUp(input_channels=self.latent_dim,n_filters=self.latent_dim),
                                    ResBlockUp(input_channels=self.latent_dim,n_filters=self.latent_dim),
                                    nn.BatchNorm2d(self.latent_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(self.latent_dim, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.Tanh())
     
        self.starting_image_size = starting_image_size



    @torch.jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        x = self.dense(z)
        x = x.reshape(-1,self.latent_dim,self.starting_image_size,self.starting_image_size)
        return self.layers(x)

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        #TODO: Ishaan fix cuda arg
        z = torch.normal(torch.zeros(n_samples,self.latent_dim),torch.ones(n_samples,self.latent_dim)).to('cuda') 
        return self.forward_given_samples(z)


class Discriminator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO 1.1: Setup the network layers
        self.latent_dim = 128
        self.layers = nn.Sequential(ResBlockDown(input_channels=3,n_filters=self.latent_dim),
                                    ResBlockDown(input_channels=self.latent_dim,n_filters=self.latent_dim),
                                    ResBlock(input_channels=self.latent_dim,n_filters=self.latent_dim),
                                    ResBlock(input_channels=self.latent_dim,n_filters=self.latent_dim))
        self.relu = nn.ReLU()
        self.dense = nn.Linear(in_features=self.latent_dim*64, out_features=1)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to sum across the image dimensions after passing x through self.layers.
        x = self.layers(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0],-1)
        return self.dense(x)
