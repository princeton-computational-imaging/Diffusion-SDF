import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, reduce

from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')


class BetaVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 kl_std=1.0,
                 beta: int = 4,
                 gamma:float = 10., 
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5, # 10000 in default configs
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.in_channels = in_channels

        self.kl_std = kl_std

        #print("kl standard deviation: ", self.kl_std)

        modules = []
        if hidden_dims is None:
            #hidden_dims = [32, 64, 128, 256, 512]
            hidden_dims = [512, 512, 512, 512, 512]

        self.hidden_dims = hidden_dims

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim) 


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4) 

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels, # changed from 3 to in_channels
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())


        #print(self)

    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        result = enc_input
        result = self.encoder(enc_input)  # [B, D, 2, 2]
        result = torch.flatten(result, start_dim=1) # ([32, D*4])

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z: latent vector: B, D (D = latent_dim*3)
        '''
        
        result = self.decoder_input(z) # ([32, D*4])
        result = result.view(-1, int(result.shape[-1]/4), 2, 2)  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        result = self.decoder(result)
        result = self.final_layer(result) # ([32, D, resolution, resolution])
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), data, mu, log_var, z]

    # only using VAE loss
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        data = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        #print("recon, data shape: ", recons.shape, data.shape)
        #recons_loss = F.mse_loss(recons, data)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

      
        if self.kl_std == 'zero_mean':
            latent = self.reparameterize(mu, log_var) 
            #print("latent shape: ", latent.shape) # (B, dim)
            l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
            kl_loss = l2_size_loss / latent.shape[0]

        else:
            std = torch.exp(0.5 * log_var)
            gt_dist = torch.distributions.normal.Normal( torch.zeros_like(mu), torch.ones_like(std)*self.kl_std )
            sampled_dist = torch.distributions.normal.Normal( mu, std )
            #gt_dist = normal_dist.sample(log_var.shape)
            #print("gt dist shape: ", gt_dist.shape)

            kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist) # reversed KL
            kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()

        return kld_weight * kl_loss

    def sample(self,
               num_samples:int,
                **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.cuda()

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_latent(self, x):
        '''
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        '''
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z 