import torch
import torch.nn as nn
import pyro
import pyro.distributions as distributions
from pyro.infer import config_enumerate


# class MDN(torch.jit.ScriptModule):
class MDN(nn.Module):
    """MDN
    
    Mixture Density Network.
    """
    def __init__(self, n_input: int, n_hidden: int, n_gaussians: int):
        super().__init__()
        self.z_h = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh()
        )
        self.n_input = n_input
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.n_gaussians = n_gaussians
        pyro.module("Model", self)
    
    def forward(self, data: torch.Tensor):
        z_h = self.z_h(data.view(-1, self.n_input))
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu

    # For inference, we enumerate over the discrete Gaussian mixtures.
    @config_enumerate
    def model(self, x: torch.Tensor, y: torch.Tensor):
        """
        Generative model for the data.
        """
        pyro.module("Model", self)

        pi, sigma, mu = self.forward(y)
        muT = torch.transpose(mu, 0, 1)
        sigmaT = torch.transpose(sigma, 0, 1)

        n_samples = y.shape[0]
        with pyro.plate("samples", n_samples):
            assign = pyro.sample("assign", distributions.Categorical(pi))
            if len(assign.shape) == 1:
                sample = pyro.sample('obs', distributions.Normal(torch.gather(muT, 0,
                                                                     assign.view(1, -1))[0],
                                                        torch.gather(sigmaT, 0,
                                                                     assign.view(1, -1))[0]),
                                     obs=x)
            else:
                sample = pyro.sample('obs', distributions.Normal(muT[assign][:, 0],
                                                        sigmaT[assign][:, 0]),
                                     obs=x)
        return sample
    
    def sampling(self, y: torch.Tensor, n: int):
        pi, sigma, mu = self.forward(y)
        muT = torch.transpose(mu, 0, 1)
        sigmaT = torch.transpose(sigma, 0, 1)
        
        m1 = torch.distributions.categorical.Categorical(pi)
        assign = m1.sample(sample_shape=(n,))
        m2 = torch.distributions.normal.Normal(
            torch.gather(muT, 0, assign),
            torch.gather(sigmaT, 0, assign),
        )
        sample = m2.sample()
        return sample