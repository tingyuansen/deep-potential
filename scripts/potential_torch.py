#!/usr/bin/env python

from __future__ import print_function, division

import torch
from torch.autograd import Variable
print(f'Pytorch version {torch.__version__}')

import numpy as np
import matplotlib.pyplot as plt



# define asinh
def asinh(x):
    return torch.log(x+(x**2+1)**0.5)


# define placian
def laplacian(xs, f, create_graph=False, keep_graph=None, return_grad=False):
    xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
    xs_flat = torch.stack(xis, dim=1)
    ys = f(xs_flat.view_as(xs))
    (ys_g, *other) = ys if isinstance(ys, tuple) else (ys, ())
    ys_g = torch.squeeze(ys_g)
    ones = torch.ones_like(ys_g)
    (dy_dxs,) = torch.autograd.grad(ys_g, xs_flat, ones, create_graph=True)
    lap_ys = sum(
        torch.autograd.grad(
            dy_dxi, xi, ones, retain_graph=True, create_graph=create_graph
        )[0]
        for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))
    )
    if not (create_graph if keep_graph is None else keep_graph):
        ys = (ys_g.detach(), *other) if isinstance(ys, tuple) else ys.detach()
    result = lap_ys, ys
    if return_grad:
        result += (dy_dxs.detach().view_as(xs),)
    return result



def calc_df_deta(f_func, q, p):
    # Calculate gradients of distribution function
    f = f_func(q, p)

    df_dq = torch.autograd.grad(f, q,\
                            grad_outputs=torch.ones_like(f), retain_graph=True,\
                            create_graph=True, allow_unused=True)[0]
    df_dp = torch.autograd.grad(f, p,\
                            grad_outputs=torch.ones_like(f), retain_graph=True,\
                            create_graph=True, allow_unused=True)[0]
    return f, df_dq, df_dp


def calc_phi_derivatives(phi_func, q):
    phi = phi_func(q)
    dphi_dq = torch.autograd.grad(phi, q, grad_outputs=torch.ones_like(phi),\
                                  retain_graph=True, create_graph=True)[0]
    d2phi_dq2 = laplacian(q, phi_func, create_graph=True, keep_graph=True, return_grad=True)[0]

    return dphi_dq, d2phi_dq2


# @tf.function
def get_phi_loss_gradients(phi, params, q, p,
                           f=None, df_dq=None, df_dp=None,
                           lam=torch.Tensor([1.0]),
                           mu=torch.Tensor([0.01]),
                           sigma_q=torch.Tensor([1.0]),
                           sigma_p=torch.Tensor([1.0]),
                           eps_w=torch.Tensor([0.1]),
                           weight_samples=False):
    """
    Calculates both the loss and the gradients of the loss w.r.t. the
    given parameters.

    In the following, let n be the number of points and d be the
    number of spatial parameters.

    Inputs:
        f (callable): The distribution function. Takes q and p, each
            (n,d) tensors, and returns a (n,) tensor.
        phi (callable): The gravitational potential. Takes q, a
            (n,d) tensor, and returns a (n,) tensor.
        params (torch Variable): The gradients will be taken
            w.r.t these parameters.
        q (torch Tensor): Spatial coordinates at which to compute the
            loss and gradients.
        p (torch Tensor): Momenta at which to compute the loss and
            gradients.
        lam (scalar): Constant that determines how strongly to
            penalize negative matter densities. Larger values
            translate to stronger penalties. Defaults to 1.

    Outputs:
        loss (tf.Tensor): Scalar tensor.
        dloss_dparam (list of tf.Tensor): The gradient of the loss
            w.r.t. each parameter.
    """
    # If df/dq and df/dp are not provided, then calculate them
    # from using the distribution function.
    if f is not None:
        _, df_dq, df_dp = calc_df_deta(f, q, p)
    elif (df_dq is None) or (df_dp is None):
        raise ValueError(
            'If f is not provided, then df_dq and df_dp must be provided.'
        )

    # Calculate derivatives of phi w.r.t. q
    dphi_dq, d2phi_dq2 = calc_phi_derivatives(phi, q)

    # partial f / partial t = {H,f}
    df_dt = torch.sum(df_dp * dphi_dq - df_dq * p, axis=1)

    likelihood = asinh(df_dt.abs())
    prior_neg = asinh(torch.clamp(-d2phi_dq2, 0., np.inf))
    prior_pos = asinh(torch.clamp(d2phi_dq2, 0., np.inf))

    loss = torch.mean(
            likelihood
            + lam*prior_neg
            + mu*prior_pos
    )

    # Gradients of loss w.r.t. NN parameters
    dloss_dparam = torch.autograd.grad(loss, params, grad_outputs=torch.ones_like(loss),\
                                  retain_graph=True, create_graph=True, allow_unused=True)[0]

    return loss, dloss_dparam


class PhiNN(torch.nn.Module):
    """
    Feed-forward neural network to represent the gravitational
    potential.
    """

    def __init__(self, n_dim=3, n_hidden=3, n_features=32, build=True):
        super(PhiNN, self).__init__()
        layers = []

        """
        Constructor for PhiNN.

        Inputs:
            n_dim (int): Number of spatial dimensions in the input.
            n_hidden (int): Number of hidden layers.
            n_features (int): Number of neurons in each hidden layer.
            build (bool): Whether to create the weights and biases.
                        Defaults to True. This option exists so that the
                        loader can set the weights and biases on its
                        own.
        """

        self._n_dim = n_dim
        self._n_hidden = n_hidden
        self._n_features = n_features

        for i in range(self._n_hidden):
            if i == 0:
                layers.append(torch.nn.Linear(self._n_dim, self._n_features))
            else:
                layers.append(torch.nn.Linear(self._n_features, self._n_features))
            layers.append(torch.nn.Sigmoid())

        layers.append(torch.nn.Linear(self._n_features, 1))
        self.model = torch.nn.Sequential(*layers)
        self.add_module("model", self.model)

    def forward(self, x):
        return self.model(x)


def main():
    return 0

if __name__ == '__main__':
    main()
