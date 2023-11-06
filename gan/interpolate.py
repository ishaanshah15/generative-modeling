import argparse
import torch
import os
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision
import itertools


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # TODO 1.2: Generate and save out latent space interpolations.
    # Generate 100 samples of 128-dim vectors
    # Do so by linearly interpolating for 10 steps across each of the first two dimensions between -1 and 1.
    # Keep the rest of the z vector for the samples to be some fixed value (e.g. 0).
    # Forward the samples through the generator.
    # Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    a = list((torch.arange(10) - 4.5)/4.5)
    interp_10 = torch.tensor(list(itertools.product(a,a)))
    latents = torch.zeros(100,128)
    latents[:,:2 ] = interp_10 
    image = 0.5*gen.forward_given_samples(latents.to('cuda')) + 0.5
    
    torchvision.utils.save_image(image,path,nrow=10)

if __name__ == '__main__':
    gen = torch.load('data_gan/generator.pt')
    path = os.path.join('interpolations/dcgan_interpolations.png')
    interpolate_latent_space(gen,path)

    gen = torch.load('data_ls_gan/generator.pt')
    path = os.path.join('interpolations/lsgan_interpolations.png')
    interpolate_latent_space(gen,path)

    gen = torch.load('data_wgan_gp/generator.pt')
    path = os.path.join('interpolations/wgan_gp_interpolations.png')
    interpolate_latent_space(gen,path)

    import ipdb
    ipdb.set_trace()

    
