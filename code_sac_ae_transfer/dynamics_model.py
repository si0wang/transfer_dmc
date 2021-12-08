import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import gzip
import itertools

device = torch.device('cuda')

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

class EnsembleFC(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class EnsembleModel(nn.Module):
    def __init__(self, feature_size, ensemble_size, use_decay=False):
        super(EnsembleModel, self).__init__()
        self.nn1 = EnsembleFC(feature_size + feature_size, feature_size, ensemble_size, weight_decay=0.000025)
        self.use_decay = use_decay
        self.apply(weights_init_)
        self.swish = Swish()

    def forward(self, state_latent, action_latent):
        x = torch.cat([state_latent, action_latent], 2)
        # nn1_output = self.swish(self.nn1(x))
        # nn2_output = self.swish(self.nn2(nn1_output))

        nn1_output = self.nn1(x)
        return nn1_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, labels):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(labels.shape) == 3
        mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
        total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss


class DynamicsModel(nn.Module):
    def __init__(self, feature_size, hidden_size=256, use_decay=False):
        super(DynamicsModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = nn.Linear(feature_size + feature_size, feature_size)
        self.use_decay = use_decay
        self.apply(weights_init_)
        self.swish = Swish()

    def forward(self, state_latent, action_latent):
        x = torch.cat([state_latent, action_latent], 1)

        nn1_output = self.nn1(x)
        return nn1_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if type(m) == nn.Linear:
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss


class RewardModel(nn.Module):
    def __init__(self, feature_size, reward_size, hidden_size=256, use_decay=False):
        super(RewardModel, self).__init__()
        self.hidden_size = hidden_size
        # self.nn1 = nn.Linear(feature_size + action_size, hidden_size)
        # self.nn2 = nn.Linear(hidden_size, feature_size + reward_size)
        self.nn1 = nn.Linear(feature_size + feature_size, reward_size)
        self.use_decay = use_decay
        self.apply(weights_init_)
        self.swish = Swish()

    def forward(self, state_latent, action_latent):
        x = torch.cat([state_latent, action_latent], 1)
        # nn1_output = self.swish(self.nn1(x))
        # nn2_output = self.swish(self.nn2(nn1_output))

        nn1_output = self.nn1(x)
        return nn1_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if type(m) == nn.Linear:
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss


class DynamicsActionEncoder(nn.Module):
    def __init__(self, num_actions, hidden_dim, num_feature):
        super(DynamicsActionEncoder, self).__init__()

        self.encode1 = nn.Linear(num_actions, hidden_dim)
        self.encode2 = nn.Linear(hidden_dim, num_feature)

        self.apply(weights_init_)

    def forward(self, action):
        e1 = F.relu(self.encode1(action))
        feature = F.relu(self.encode2(e1))

        return feature

class RewardActionEncoder(nn.Module):
    def __init__(self, num_actions, hidden_dim, num_feature):
        super(RewardActionEncoder, self).__init__()

        self.encode1 = nn.Linear(num_actions, hidden_dim)
        self.encode2 = nn.Linear(hidden_dim, num_feature)

        self.apply(weights_init_)

    def forward(self, action):
        e1 = F.relu(self.encode1(action))
        feature = F.relu(self.encode2(e1))

        return feature