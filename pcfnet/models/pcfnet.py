from collections import OrderedDict
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init


class MaskedBatchNorm1d(nn.Module):
    """ A masked version of nn.BatchNorm1d. Only tested for 3D inputs.
        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, L)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``True``
        Shape:
            - Input: :math:`(N, C, L)`
            - input_mask: (N, 1, L) tensor of ones and zeros, where the zeros indicate locations not to use.
            - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
            
    https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    """

    def __init__(self,
                 num_features: int, 
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 1))
            self.bias = nn.Parameter(torch.Tensor(num_features, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.ones(1, num_features, 1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: torch.Tensor, input_mask: Optional[torch.Tensor]=None):
        # Calculate the masked mean and variance
        shape = input.shape
        C = shape[-2]
        if input_mask is not None:
            cast_mask = torch.broadcast_to(input_mask, input.shape)
        else:
            cast_mask = torch.ones(input.shape)
        # if input_mask is not None and input_mask.shape != (B, 1, L):
        #     raise ValueError('Mask should have shape (B, 1, L).')
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        masked = input * cast_mask
        n = cast_mask.sum()

        # Sum
        dims = list(np.delete(np.arange(len(shape)),-2))
        masked_sum = masked.sum(dim=dims, keepdims=True)
        # Divide by sum of mask
        current_mean = masked_sum / n
        current_var = ((masked - current_mean * cast_mask) ** 2).sum(dim=dims, keepdims=True) / n
        # Update running stats
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * current_mean)
                self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * current_var)
            self.num_batches_tracked += 1
        # Norm the input
        if self.track_running_stats and not self.training:
            normed = (masked - self.running_mean) / (torch.sqrt(self.running_var + self.eps))
        else:
            normed = (masked - current_mean) / (torch.sqrt(current_var + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight + self.bias
        return normed


class KNN():
    """KNN
    https://github.com/WangYueFt/dgcnn/tree/master/pytorch
    """
    def __init__(self,):
        self.masking = FillMask(-10.**9)
        
    def knn(self, x: torch.Tensor, mask:Optional[torch.Tensor] = None, k: int = 20):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        if mask is not None:
            pairwise_distance = self.masking(pairwise_distance, mask)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx


class GraphFeature(nn.Module):
    def __init__(self,
                 k: int = 20,
                 idx: Optional[torch.Tensor] = None,
                 dim9: bool = False):
        super(GraphFeature, self).__init__()
        self.knn = KNN()
        self.k = k
        self.idx = idx
        self.dim9 = dim9

    def get_graph_feature(self,
                          x: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None, 
                          k: int = 20, 
                          idx=None, 
                          dim9=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if not dim9:
                idx = self.knn.knn(x, mask=mask, k=k)   # (batch_size, num_points, k)
            else:
                idx = self.knn.knn(x[:, 6:], mask=mask, k=k)
        device = torch.device(x.device)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature      # (batch_size, 2*num_dims, num_points, k)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        return self.get_graph_feature(x, mask, k=self.k, idx=self.idx, dim9=self.dim9)

    
class ConvBNReLU1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int, 
                 kernel_size: int = 1, 
                 bias: bool = False, 
                 ac: bool =True):
        super(ConvBNReLU1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        self.bn = MaskedBatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.ac = ac

    def forward(self, x: torch.Tensor, mask:torch.Tensor):
        x = self.bn(self.conv(x), input_mask=mask)
        if self.ac:
            x = self.relu(x)
        return x


class ConvMaxBNReLU1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 neighbor: int,
                 connection: str,
                 kernel_size: int = 1,
                 bias: bool = False,
                 ac: bool = True):
        super(ConvMaxBNReLU1D, self).__init__()
        self.gf = GraphFeature(k=neighbor)
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size, bias=bias)
        self.bn = MaskedBatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.connection = connection
        if self.connection=='residual':
            self.pro = ConvBNReLU1D(in_channels, out_channels,ac=ac)
        elif self.connection=='concat':
            self.pro = ConvBNReLU1D(in_channels*2, out_channels,ac=ac)
        else:
            raise ValueError(f"Expected connection is 'residual' or 'concat' but input is {self.connection}")   

    def forward(self, x, mask):
        pro_x = self.relu(self.bn(self.conv(self.gf(x, mask)).max(dim=-1, keepdim=False)[0], input_mask=mask))
        if self.connection=='residual':
            x = self.pro(x+pro_x, mask)
        elif self.connection=='concat':
            x = self.pro(torch.cat([x,pro_x], 1), mask)
        else:
            raise ValueError(f"Expected connection is 'residual' or 'concat' but input is {self.connection}")
        return x


class PCFNetfeat(nn.Module):
    """PCFNetfeat
    
    PCFNet Feature Extractor.
    """
    def __init__(self,
                 dim: int = 4,
                 hidden_num: list[int] = [16, 32, 64, 128, 256],
                 k: int = 5,
                 connection: str = 'residual'):
        super(PCFNetfeat, self).__init__()
        self.k = k
        self.connection = connection
        self.pre = ConvBNReLU1D(dim, hidden_num[0],)
        self.blocks = nn.ModuleList()
        self.layer_num = len(hidden_num) - 1
        for i in range(self.layer_num):
            if i == (self.layer_num - 1):
                self.blocks.append(ConvMaxBNReLU1D(hidden_num[i], hidden_num[i+1], k, connection=connection, ac=False))
            else:
                self.blocks.append(ConvMaxBNReLU1D(hidden_num[i], hidden_num[i+1], k, connection=connection,))
        self.fill_mask = FillMask()

    def forward(self, x: torch.Tensor, c:torch.Tensor, mask: Optional[torch.Tensor]=None):
        trans = None
        trans_feat = None
        # add color information
        if c is not None:
            x = torch.cat([x, c], 1)
        # pre-linear
        x = self.pre(x, mask)
        # DGCNN
        for i in range(self.layer_num):
            x = self.blocks[i](x, mask)
   
        if mask is not None:
            x = self.fill_mask(x, mask)
        x = torch.max(x, 2, keepdim=False)[0]
        return x, trans, trans_feat


class PCFNet(nn.Module):
    """PCFNet
    
    PCFNet Model.
    Please see Takeda et la. (2024) for more details.
    """
    def __init__(self,
                 k: int = 2,
                 dim: int = 4, 
                 hidden_num: list[int] = [16, 32, 64, 128, 256],
                 neighbor: int = 5,
                 connection: str = 'residual'):
        super(PCFNet, self).__init__()
        self.out_dim = hidden_num[-1]
        self.feat = PCFNetfeat(
            dim=dim,
            hidden_num=hidden_num,
            k=neighbor,
            connection=connection
        )
        self.sq = nn.Sequential(OrderedDict([
            ('cls_fc1', nn.Linear(self.out_dim, 128, bias=False)),
            ('cls_bn1', nn.BatchNorm1d(128)),
            ('relu1', nn.ReLU()),
            ('cls_fc2', nn.Linear(128, 64, bias=False)),
            ('dropout', nn.Dropout(p=0.3)),
            ('cls_bn2', nn.BatchNorm1d(64)),
            ('relu2', nn.ReLU()),
            ('cls_fc3', nn.Linear(64, k)),
            ('soft', nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: Optional[torch.Tensor]=None):
        x1, trans, trans_feat = self.feat(x, c, mask)
        x2 = self.sq(x1)
        return x2, trans, trans_feat


class FillMask(nn.Module):
    def __init__(self, value=-np.inf):
        super(FillMask, self).__init__()
        self.value = value

    def forward(self, x: torch.Tensor, mask: torch.Tensor,):
        device = x.device
        x = torch.where(mask==1, x, torch.as_tensor(self.value, device=device))
        return x