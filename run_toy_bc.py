import os
import torch
import numpy as np
from torch.distributions import Normal
import argparse
import matplotlib.pyplot as plt

from toy_helpers import Data_Sampler, reward_fun

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=2022, type=int)
parser.add_argument('--device', default=0, type=int)
args = parser.parse_args()

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

seed = args.seed

def generate_data(num, device = 'cpu'):
    
    each_num = int(num / 4)
    pos = 0.8
    std = 0.05
    left_up_conor = Normal(torch.tensor([-pos, pos]), torch.tensor([std, std]))
    left_bottom_conor = Normal(torch.tensor([-pos, -pos]), torch.tensor([std, std]))
    right_up_conor = Normal(torch.tensor([pos, pos]), torch.tensor([std, std]))
    right_bottom_conor = Normal(torch.tensor([pos, -pos]), torch.tensor([std, std]))
    
    left_up_samples = left_up_conor.sample((each_num,)).clip(-1.0, 1.0)
    left_bottom_samples = left_bottom_conor.sample((each_num,)).clip(-1.0, 1.0)
    right_up_samples = right_up_conor.sample((each_num,)).clip(-1.0, 1.0)
    right_bottom_samples = right_bottom_conor.sample((each_num,)).clip(-1.0, 1.0)
    
    data = torch.cat([left_up_samples, left_bottom_samples, right_up_samples, right_bottom_samples], dim=0)

    action = data
    state = torch.zeros_like(action)
    reward = reward_fun(action)
    return Data_Sampler(state, action, reward, device)

torch.manual_seed(seed)
np.random.seed(seed)

device = args.device
num_data = int(10000)
data_sampler = generate_data(num_data, device)

state_dim = 2
action_dim = 2
max_action = 1.0

discount = 0.99
tau = 0.005
model_type = 'MLP'

T = 50
beta_schedule = 'vp'
hidden_dim = 128
lr = 3e-4

num_epochs = 200
batch_size = 100
iterations = int(num_data / batch_size)

img_dir = 'toy_imgs/bc'
os.makedirs(img_dir, exist_ok=True)
fig, axs = plt.subplots(1, 4, figsize=(5.5 * 4, 5))
axis_lim = 1.1

# Plot the ground truth
num_eval = 1000
_, action_samples, _ = data_sampler.sample(num_eval)
action_samples = action_samples.cpu().numpy()
axs[0].scatter(action_samples[:, 0], action_samples[:, 1], alpha=0.3)
axs[0].set_xlim(-axis_lim, axis_lim)
axs[0].set_ylim(-axis_lim, axis_lim)
axs[0].set_xlabel('x', fontsize=20)
axs[0].set_ylabel('y', fontsize=20)
axs[0].set_title('Ground Truth', fontsize=25)

from TD3_BC_toy2 import TD3_BC

# Plot TD3+BC_GM with K=2
K = 2
td3_bc_gm = TD3_BC(state_dim=state_dim,
                   action_dim=action_dim,
                   max_action=max_action,
                   device=device,
                   discount=discount,
                   tau=tau,
                   n_components=K)


for i in range(num_epochs):
    for _ in range(iterations):
        td3_bc_gm.train(data_sampler,
                        batch_size=batch_size)

    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = td3_bc_gm.actor(new_state)
new_action = new_action.detach().cpu().numpy()
axs[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[1].set_xlim(-axis_lim, axis_lim)
axs[1].set_ylim(-axis_lim, axis_lim)
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title(f'TD3+BC-GM (K={K})', fontsize=25)

# Plot TD3+BC_GM with K=3
K = 3
td3_bc_gm = TD3_BC(state_dim=state_dim,
                   action_dim=action_dim,
                   max_action=max_action,
                   device=device,
                   discount=discount,
                   tau=tau,
                   n_components=K)

for i in range(num_epochs):
    for _ in range(iterations):
        td3_bc_gm.train(data_sampler,
                        batch_size=batch_size)

    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = td3_bc_gm.actor(new_state)
new_action = new_action.detach().cpu().numpy()
axs[2].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title(f'TD3+BC-GM (K={K})', fontsize=25)

# Plot TD3+BC_GM with K=2
K = 4
td3_bc_gm = TD3_BC(state_dim=state_dim,
                   action_dim=action_dim,
                   max_action=max_action,
                   device=device,
                   discount=discount,
                   tau=tau,
                   n_components=K)

for i in range(num_epochs):
    for _ in range(iterations):
        td3_bc_gm.train(data_sampler,
                        batch_size=batch_size)

    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = td3_bc_gm.actor(new_state)
new_action = new_action.detach().cpu().numpy()
axs[3].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[3].set_xlim(-axis_lim, axis_lim)
axs[3].set_ylim(-axis_lim, axis_lim)
axs[3].set_xlabel('x', fontsize=20)
axs[3].set_ylabel('y', fontsize=20)
axs[3].set_title(f'TD3+BC-GM (K={K})', fontsize=25)


fig.tight_layout()
fig.savefig(os.path.join(img_dir, f'bc_ablation.pdf'))



