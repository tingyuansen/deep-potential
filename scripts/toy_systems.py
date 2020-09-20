#!/usr/bin/env python

from __future__ import print_function, division

import torch
print(f'Pytorch version {torch.__version__}')

import numpy as np
import scipy

from scipy.integrate import cumtrapz, solve_ivp
from scipy.interpolate import interp1d


def get_1d_sampler(p, x_min, x_max, n=100):
    x = np.linspace(x_min, x_max, n)
    p_x = p(x)

    P_x = cumtrapz(p_x, x)
    P_x /= P_x[-1]
    P_x = np.hstack([0., P_x])

    x_of_P = interp1d(P_x, x)

    def sample(shape=None):
        u = np.random.uniform(size=shape)
        return x_of_P(u)

    return sample


def draw_from_sphere(n):
    phi = np.random.uniform(0., 2*np.pi, size=n)
    theta = np.arccos(np.random.uniform(-1., 1., size=n))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x,y,z], axis=1)


class PlummerSphere(object):
    def __init__(self):
        self._v_sampler = get_1d_sampler(
            lambda v: v**2 * (1 - v**2 / 2)**(7/2),
            0., np.sqrt(2.)-1.e-8,
            n=1000
        )
        self.df_norm = 24*np.sqrt(2.) / (7*np.pi**3)

    def psi(self, r):
        return 1 / np.sqrt(1 + r**2)

    def phi(self, r):
        return -self.psi(r)

    def rho(self, r):
        return 3/(4*np.pi) * (1+r**2)**(-5/2)

    def sample_r(self, n):
        u = np.random.uniform(size=n)
        r = 1 / np.sqrt(u**(-2/3) - 1)
        return r

    def sample_df(self, n):
        r = self.sample_r(n)
        x = r[:,None] * draw_from_sphere(n)

        psi = self.psi(r)
        v = np.sqrt(psi) * self._v_sampler(n)
        v = v[:,None] * draw_from_sphere(n)

        return x, v

    def df(self, x, v):
        r = np.sqrt(np.sum(x**2, axis=1))
        v2 = np.sum(v**2, axis=1)
        E = self.psi(r) - 0.5*v2
        return self.df_norm * np.clip(E, 0., np.inf)**(7/2)


def integrate_path(force_fn, x0, v0, t_max, **kwargs):
    assert x0.shape == v0.shape

    n_particles, n_dim = x0.shape
    s = (x0.size,)

    def dy_dt(t, y):
        y = np.reshape(y, (2, n_particles, n_dim))
        x, x_t = y
        x_tt = force_fn(t, x)
        y_t = np.concatenate([np.reshape(x_t, s), np.reshape(x_tt, s)])
        return y_t

    y0 = np.concatenate([np.reshape(x0, s), np.reshape(v0, s)])

    sol = solve_ivp(dy_dt, (0.,t_max), y0, **kwargs)

    x,v = np.reshape(sol.y, (2, n_particles, n_dim, sol.y.shape[1]))

    return sol, x, v

def calc_force(phi_func, x):
    phi = phi_func(x)[:,None]
    dphi_dx = torch.autograd.grad(phi, x,\
                                grad_outputs=torch.ones_like(phi), retain_graph=True,\
                                create_graph=True)[0]
    return -dphi_dx

def calc_force_np(phi_func, x):
    x = Variable(torch.from_numpy(x), requires_grad=True)
    return calc_force(phi_func, x).detach().numpy()


def main():
    return 0

if __name__ == '__main__':
    main()
