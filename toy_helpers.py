import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


class Data_Sampler(object):
    def __init__(self, state, action, reward, device):

        self.state = state
        self.action = action
        self.reward = reward

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.reward[ind].to(self.device)
        )


def reward_fun(a, std=0.3):
    assert a.shape[-1] == 2
    pos = 0.8
    # std = 0.3
    left_up_conor = Normal(torch.tensor([-pos, pos]), torch.tensor([std, std]))
    left_bottom_conor = Normal(torch.tensor([-pos, -pos]), torch.tensor([std, std]))
    right_up_conor = Normal(torch.tensor([pos, pos]), torch.tensor([std, std]))
    right_bottom_conor = Normal(torch.tensor([pos, -pos]), torch.tensor([std, std]))
    
    left_up_dis = left_up_conor.log_prob(a).sum(dim=-1, keepdim=True).exp()
    left_bottom_dis = left_bottom_conor.log_prob(a).sum(dim=-1, keepdim=True).exp()
    right_up_dis = right_up_conor.log_prob(a).sum(dim=-1, keepdim=True).exp()
    right_bottom_dis = right_bottom_conor.log_prob(a).sum(dim=-1, keepdim=True).exp()
    
    left_up_r = left_up_dis * 1.0
    left_bottom_r = left_bottom_dis * 1.5
    right_up_r = right_up_dis * 2.0
    right_bottom_r = right_bottom_dis * 2.5
                                  
    return left_up_r + left_bottom_r + right_up_r + right_bottom_r


class Ill_Reward(nn.Module):
    def __init__(self, x=0., y=0., std=0.25, ill_std=0.01, device='cpu'):
        super(Ill_Reward, self).__init__()
        pos = 0.8
        # std = 0.3
        self.left_up_conor = Normal(torch.tensor([-pos, pos]).to(device), torch.tensor([std, std]).to(device))
        self.left_bottom_conor = Normal(torch.tensor([-pos, -pos]).to(device), torch.tensor([std, std]).to(device))
        self.right_up_conor = Normal(torch.tensor([pos, pos]).to(device), torch.tensor([std, std]).to(device))
        self.right_bottom_conor = Normal(torch.tensor([pos, -pos]).to(device), torch.tensor([std, std]).to(device))

        self.ill_x, self.ill_y = x, y
        self.ill_center = Normal(torch.tensor([x, y]).to(device), torch.tensor([ill_std, ill_std]).to(device))

    def forward(self, a):
        left_up_dis = self.left_up_conor.log_prob(a).sum(dim=-1, keepdim=True).exp()
        left_bottom_dis = self.left_bottom_conor.log_prob(a).sum(dim=-1, keepdim=True).exp()
        right_up_dis = self.right_up_conor.log_prob(a).sum(dim=-1, keepdim=True).exp()
        right_bottom_dis = self.right_bottom_conor.log_prob(a).sum(dim=-1, keepdim=True).exp()

        ill_center_dis = self.ill_center.log_prob(a).sum(dim=-1, keepdim=True).exp()

        left_up_r = left_up_dis * 1.0
        left_bottom_r = left_bottom_dis * 1.5
        right_up_r = right_up_dis * 2.0
        right_bottom_r = right_bottom_dis * 3.0

        ill_center_r = ill_center_dis * 1.

        return left_up_r + left_bottom_r + right_up_r + right_bottom_r + ill_center_r

