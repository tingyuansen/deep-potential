import numpy as np
import itertools
import time
import os

import torch
import torch.optim as optim

import sys
sys.path.append('../scripts/')
import toy_systems
import flow_torch

#----------------------------------------------------------------------------------------------------
# cpu or gpu
device = 'cpu'
from multiprocessing import Pool

# set number of threads per CPU
os.environ['OMP_NUM_THREADS']='{:d}'.format(1)


#====================================================================================================
# Instantiate Plummer sphere class
plummer_sphere = toy_systems.PlummerSphere()

def sample_df(n_samples, max_dist=None):
    """
    Returns phase-space locations sampled from the Plummer sphere
    distribution function. The shape of the output is
    (n_samples, 6).
    """
    x,v = plummer_sphere.sample_df(n_samples)
    if max_dist is not None:
        r2 = np.sum(x**2, axis=1)
        idx = (r2 < max_dist**2)
        x = x[idx]
        v = v[idx]

    return torch.cat([torch.Tensor(x.astype('f4')), torch.Tensor(v.astype('f4'))], axis=1)

#----------------------------------------------------------------------------------------------------
# make flow
n_flows = 10000

def make_normalizing_flow(i):

    # set random seed
    torch.manual_seed(i*100)

    # make data
    n_samples = 1024 * 128
    data = sample_df(int(1.2 * n_samples), max_dist=10.0)
    data = torch.Tensor(data[:n_samples,:])

#----------------------------------------------------------------------------------------------------
    # flow hyperparameters
    n_dim = 6
    n_units = 4

    n_epochs = 32
    batch_size = 1024

    n_steps = n_samples * n_epochs // batch_size
    print(f'n_steps = {n_steps}')

#----------------------------------------------------------------------------------------------------
    print(f'Training flow {i+1} of {n_flows} ...')
    flow = flow_torch.NormalizingFlow(n_dim, n_units)
    loss_history = flow_torch.train_flow(
        flow, data,
        n_epochs=n_epochs,
        batch_size=batch_size,
        callback=flow_torch.get_training_callback(
            flow,
            plt_fn=None,
            every=1024,
        )
    )

    torch.save(flow.state_dict(), f'plummer_flow_{i:02d}.pth')


#====================================================================================================
# fit spectra in batch
num_CPU = 96
pool = Pool(num_CPU)
pool.map(make_normalizing_flow,range(n_flows));
