import numpy as np
import torch
from torchvision import transforms
from ..util import util

class Feeder(torch.utils.data.Dataset):
    def __init__(self, 
                 data, 
                 index, 
                 y=None, 
                 transform=transforms.Compose([]),
                 n_gaussians: int = 3):
        self.data = data
        self.index = index
        self.n_gaussians = n_gaussians
        if y is not None:
            self.y = torch.from_numpy(y)
        else:
            self.y = None
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        with torch.no_grad():
            center, indexes = self.index[i]
            x = torch.from_numpy(self.data[indexes]).float()
            c_i = np.where(indexes==center)[0][0]
            pi, sigma, mu = x[:,-3*self.n_gaussians:-2*self.n_gaussians], x[:,-2*self.n_gaussians:-self.n_gaussians], x[:,-self.n_gaussians:]
            m1 = torch.distributions.categorical.Categorical(pi)
            assign = m1.sample(sample_shape=(1,))
            m2 = torch.distributions.normal.Normal(
                torch.gather(mu.T, 0, assign),
                torch.gather(sigma.T, 0, assign),
            )
            samp = m2.sample()
            x = np.insert(
                np.broadcast_to(x,(1,*x.shape)),
                [2],
                samp[:,:,np.newaxis],
                axis=2)
            x[:,:,:3] = util.change_origin(x[:,:,:3], x[:,[c_i],:3])  # target galaxy is at the origin
            x = x.reshape(-1, x.shape[2])
            x = self.transform(x)
            trans_x = torch.from_numpy(x).float()
        if self.y is not None:
            return trans_x, self.y[center]
        else:
            return trans_x, None