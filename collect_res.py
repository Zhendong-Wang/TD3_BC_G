import numpy as np


env_list = ['hopper-medium-expert-v2', 'halfcheetah-medium-expert-v2', 'walker2d-medium-expert-v2']
seed_list = [0, 1, 2]

for env in env_list:
    res = []
    for sd in seed_list:
        res_sd = np.load(f'results_3/TD3_BC_{env}_{sd}.npy')
        res.append(res_sd[-1])

    print(f'{env} mean: {np.mean(res):.2f} std: {np.std(res):.2f}')

