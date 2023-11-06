import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas
        self.alphas = alphas
        # TODO 3.1: compute the cumulative products for current and previous timesteps
        
        
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.0]).to(self.device),self.alphas_cumprod[:-1]))

        # TODO 3.1: pre-compute values needed for forward process
        # This is the coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = torch.div(1,torch.pow(self.alphas_cumprod,0.5))
        # This is the coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = self.x_0_pred_coef_1*torch.pow(1- self.alphas_cumprod ,0.5)

        # TODO 3.1: compute the coefficients for the mean
        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = torch.div(torch.pow(self.alphas_cumprod_prev,0.5)*self.betas,1 - self.alphas_cumprod)
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 =  torch.div(torch.pow(self.alphas,0.5)*(1 - self.alphas_cumprod_prev),1 - self.alphas_cumprod)

        # TODO 3.1: compute posterior variance
        # calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = torch.div(1-self.alphas_cumprod_prev,1-self.alphas_cumprod)*self.betas
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # TODO 3.1: Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0
        # hint: can use extract function from utils.py
        
        x_0_term = self.posterior_mean_coef1[t].reshape(t.shape[0],1,1,1)*x_0
        x_t_term = self.posterior_mean_coef2[t].reshape(t.shape[0],1,1,1)*x_t
        posterior_mean = x_0_term + x_t_term

        return posterior_mean, self.posterior_variance[t], self.posterior_log_variance_clipped[t]

    def model_predictions(self, x_t, t):
        # TODO 3.1: given a noised image x_t, predict x_0 and the additive noise
        # to predict the additive noise, use the denoising model.
        # Hint: You can use extract function from utils.py.
        # clamp x_0 to [-1, 1]
       
        pred_noise = self.model(x_t,t)
        
        eps_term = self.x_0_pred_coef_2[t].reshape((t.shape[0],1,1,1))*pred_noise
        x_t_term = self.x_0_pred_coef_1[t].reshape((t.shape[0],1,1,1))*x_t
        
        x_0 = x_t_term - eps_term
        x_0 = torch.clamp(x_0,min=-1,max=1)
        
        return (pred_noise, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # TODO 3.1: given x at timestep t, predict the denoised image at x_{t-1}.
        # also return the predicted starting image.
        # Hint: To do this, you will need a predicted x_0. Which function can do this for you?

        _,x_0 = self.model_predictions(x,t)
        posterior_mean, posterior_variance, posterior_log_variance_clipped = self.get_posterior_parameters(x_0,x,t)
        z = torch.normal(mean=0,std=1,size=x.shape).to(x.device) 
        pred_img = posterior_mean + torch.pow(posterior_variance.reshape(x.shape[0],1,1,1),0.5)*z

        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps-1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)

        img = unnormalize_to_zero_to_one(img)
        
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        # TODO 3.2: Generate a list of times to sample from.
        samples = torch.arange(sampling_timesteps)*((total_timesteps-1)/(sampling_timesteps-1))
        samples = samples.type(torch.int32)

       
        return samples

    def get_time_pairs(self, times):
        # TODO 3.2: Generate a list of adjacent time pairs to sample from.
        samples = times
        pairs = torch.stack((samples[:-1], samples[1:]),dim=1)
        pairs = torch.flip(pairs, [0, 1])
        
        return pairs
        


    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        # TODO 3.2: Compute the output image for a single step of the DDIM sampling process.

        posterior_variance = eta*torch.div(1 - alphas_cumprod[tau_isub1],1 - alphas_cumprod[tau_i])*self.betas[tau_isub1]
        mean_x0_coeff = torch.pow(alphas_cumprod[tau_isub1],0.5)
        mean_noise_coeff = torch.pow(1 - alphas_cumprod[tau_isub1] - posterior_variance,0.5)

        tau_i =  torch.full((img.shape[0],), tau_i, device=self.device, dtype=torch.long)
        tau_isub1 =  torch.full((img.shape[0],), tau_isub1, device=self.device, dtype=torch.long)
        

        pred_noise,x_0 = self.model_predictions(img,tau_i)

        
        posterior_mean = mean_x0_coeff*x_0 + mean_noise_coeff*pred_noise

        img =  posterior_mean + torch.pow(posterior_variance,0.5)*torch.normal(mean=0,std=1,size=(img.shape)).to(device)

        # predict x_0 and the additive noise for tau_i

        # extract \alpha_{\tau_{i - 1}} and \alpha_{\tau_{i}}

        # compute \sigma_{\tau_{i}}

        # compute the coefficient of \epsilon_{\tau_{i}}

        # sample from q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0)
        # HINT: use the reparameterization trick

        return img, x_0

    def sample_ddim(self, shape, z):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)

        
        img = unnormalize_to_zero_to_one(img)

        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device = self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        #TODO 3.3: fill out based on the sample function above

        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim

        z = z.reshape(shape)

        return sample_fn(shape, z)
        
