import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import itertools
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer, required

device = 'cpu'


#==========================================================================================
class NormalizingFlow(nn.Module):
    """
    Represents a normalizing flow, with a unit Gaussian prior
    and a bijector consisting of interleaved invertible 1x1
    convolutions and Rational Quadratic Splines.
    """

    def __init__(self, n_dim, n_units, rqs=None, lu_fact=None):
        """
        Randomly initializes the normalizing flow.

        If rqs and lu_fact are provided, then they will be used
        to create the normalizing flow, instead of randomly
        initializing the bijectors.
        """

        super().__init__()
        self._n_dim = n_dim
        self._n_units = n_units
        self.build(n_dim, n_units, rqs=rqs, lu_fact=lu_fact)

#----------------------------------------------------------------------------------
    def build(self, n_dim, n_units, rqs=None, lu_fact=None):

        # Base distribution: p(x)
        self.dist = Normal(torch.zeros(self._n_dim).to(device),\
                            torch.ones(self._n_dim).to(device))

        # Generate bijectors first, so that they are accessible later
        if rqs is None:
            self.nsf = [NSF_CL(dim=self._n_dim, K=8, B=3, hidden_dim=16)\
                        for _ in range(self._n_units)]
        else:
            self.nsf = rqs

        if lu_fact is None:
            self.conv1x1 = [Invertible1x1Conv(dim=self._n_dim)\
                            for _ in range(self._n_units)]
        else:
            self.conv1x1 = lu_fact

        # Bijection: x -> y
        flows = list(itertools.chain(*zip(self.conv1x1, self.nsf)))
        self.bij = nn.ModuleList(flows)

#----------------------------------------------------------------------------------
    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(device)
        for flow in self.bij:
            x, ld = flow.forward(x)
            log_det += ld
        zs = x
        prior_logprob = self.dist.log_prob(zs).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

#----------------------------------------------------------------------------------
    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(device)
        for flow in self.bij[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
        xs = z
        return xs, log_det

#----------------------------------------------------------------------------------
    def sample(self, sample_shape):
        z = self.dist.sample(sample_shape)
        xs, _ = self.backward(z)
        return xs


#==========================================================================================
def plot_inv1x1conv(bij, ax, label_y_axis=True):
    """
    Plots a representation of an invertible 1x1
    convolution to the provided axes.
    """
    x = np.linspace(-0.8, 0.8, 10).astype('f4')
    y = np.linspace(-0.8, 0.8, 10).astype('f4')
    x,y = np.meshgrid(x,y)
    xy = torch.from_numpy(np.stack([x.flat, y.flat], axis=1))

    xy_p = bij(xy)[0].cpu().detach().numpy()

    ax.scatter(xy[:,0], xy[:,1], c='k', alpha=0.1, s=4)
    ax.scatter(xy_p[:,0], xy_p[:,1], s=4)
    for xy_i,xyp_i in zip(xy, xy_p):
        ax.plot(
            [xy_i[0], xyp_i[0]],
            [xy_i[1], xyp_i[1]],
            c='b',
            alpha=0.1
        )

    if isinstance(bij, Invertible1x1Conv):
        ax.set_title('invertible 1x1 convolution')
    elif isinstance(bij, AffineConstantFlow):
        ax.set_title('activation normalization')

    ax.set_xlabel(r'$x_0$')

    ax.set_ylabel(r'$x_1$', labelpad=-2)
    if not label_y_axis:
        ax.set_yticklabels([])

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

#----------------------------------------------------------------------------------
def plot_nsf(bij, ax, label_y_axis=True):
    """
    Plots a representation of a normalizing spline flow
    to the provided axes.
    """
    cmap = matplotlib.cm.get_cmap('viridis')
    for x0 in torch.linspace(-1., 1., 5):
        c = cmap(0.5*(x0+1))
        x = torch.stack([
            x0*torch.ones([1000]),
            torch.linspace(-1.2, 1.2, 1000)
        ], axis=1)
        y = bij(x)[0].cpu().detach().numpy()
        ax.plot(x[:,1], y[:,1], c=c)
        ax.set_title('RQS')

    ax.grid('on', alpha=0.25)
    ax.set_xlabel(r'$x_1$')
    ax.set_title(r'RQS')

    ax.set_ylabel(r'$x_1^{\prime}$', labelpad=-2)
    if not label_y_axis:
        ax.set_yticklabels([])

#----------------------------------------------------------------------------------
def plot_prob(bij_p, dist, ax_p=None, ax_lnp=None):
    # Get input grid
    x = np.linspace(-2., 2., 300)
    y = np.linspace(-2., 2., 300)
    x,y = np.meshgrid(x, y)
    s = x.shape
    xy_grid = np.stack([x,y], axis=-1)
    xy_grid.shape = (-1, 2)
    xy_grid = torch.from_numpy(xy_grid.astype('f4')).to(device)

    # Image of distribution
    m, _ = xy_grid.shape
    log_det = torch.zeros(m).to(device)
    for flow in bij_p:
        xy_grid, ld = flow.forward(xy_grid)
        log_det += ld
    zs = xy_grid
    prior_logprob = dist.log_prob(zs).view(xy_grid.size(0), -1).sum(1)

    lnp_img = prior_logprob + log_det
    lnp_img = lnp_img.cpu().detach().numpy()
    lnp_img.shape = s

    p_img = np.exp(lnp_img)
    p_img /= np.sum(p_img)

    ax = []

    if ax_p is not None:
        ax_p.imshow(
            p_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmax=np.max(p_img),
            vmin=0.
        )
        ax.append(ax_p)

    if ax_lnp is not None:
        ax_lnp.imshow(
            lnp_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmax=np.max(lnp_img),
            vmin=np.max(lnp_img) - 25.
        )
        ax.append(ax_lnp)

    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])

#----------------------------------------------------------------------------------
def plot_bijections(flow):
    """
    Returns a figure that visualizes the bijections
    in the provided flow.
    """
    n_bij = len(flow.bij)
    fig,ax_arr = plt.subplots(
        3,n_bij,
        figsize=(1+3*n_bij,10),
        subplot_kw=dict(aspect='equal')
    )
    fig.subplots_adjust(
        wspace=0.16, hspace=0.2,
        left=0.03, right=0.99,
        bottom=0.02, top=0.97
    )

    for i,b in enumerate(flow.bij):
        if isinstance(b, NSF_CL):
            plot_nsf(b, ax_arr[0,i], label_y_axis=(i==0))
        elif isinstance(b, Invertible1x1Conv) or isinstance(b, AffineConstantFlow):
            plot_inv1x1conv(b, ax_arr[0,i], label_y_axis=(i==0))
        else:
            ax_arr[0,i].axis('off')

        bij_p = flow.bij[:i+1]
        plot_prob(bij_p, flow.dist, ax_p=ax_arr[1,i], ax_lnp=ax_arr[2,i])

    ax_arr[1,0].set_ylabel(r'$p$', fontsize=18)
    ax_arr[2,0].set_ylabel(r'$\ln p$', fontsize=18)

    return fig


#==========================================================================================
def get_flow_plot_fn(flow, p_true_fn=None):
    """
    Returns a function that produces a visualization of the flow.
    """

    # Get input grid
    x = np.linspace(-2., 2., 300)
    y = np.linspace(-2., 2., 300)
    x,y = np.meshgrid(x, y)
    s = x.shape
    xy_grid = np.stack([x,y], axis=-1)
    xy_grid.shape = (-1, 2)
    xy_grid = torch.from_numpy(xy_grid.astype('f4')).to(device)

    if p_true_fn is None:
        p_true_img = None
    else:
        p_true_img = p_true_fn(xy_grid.cpu().numpy())
        p_true_img /= np.sum(p_true_img)
        p_true_img.shape = s

#----------------------------------------------------------------------------------
    def plot_fn():
        # make sample
        x_sample = flow.dist.sample(sample_shape=[1000])
        y_sample, _ = flow.backward(x_sample)
        _ , prior_logprob, log_det = flow(y_sample)
        c = prior_logprob + log_det

        x_sample = x_sample.cpu().detach().numpy()
        y_sample = y_sample.cpu().detach().numpy()
        c = c.cpu().detach().numpy()

        fig,ax_arr = plt.subplots(
            2,3,
            figsize=(15,10),
            subplot_kw=dict(aspect='equal')
        )
        fig.subplots_adjust(
            left=0.05, right=0.99,
            bottom=0.05, top=0.95
        )

        ax1,ax2,ax3 = ax_arr[0]
        ax1.scatter(x_sample[:,0], x_sample[:,1], alpha=0.3, s=3, c=c)
        ax1.set_title('x')
        ax1.set_xlim(-3., 3.)
        ax1.set_ylim(-3., 3.)
        ax2.scatter(y_sample[:,0], y_sample[:,1], alpha=0.3, s=3, c=c)
        ax2.set_title('y')
        ax2.set_xlim(-3., 3.)
        ax2.set_ylim(-3., 3.)
        for xx,yy in zip(x_sample[::4],y_sample[::4]):
            dxy = yy-xx
            ax3.arrow(
                xx[0], xx[1],
                0.2*dxy[0], 0.2*dxy[1],
                color="black",\
                alpha=0.3
            )
            ax3.set_xlim(-3., 3.)
            ax3.set_ylim(-3., 3.)
        ax3.set_title(r'$0.2 \left( y-x \right)$')

#----------------------------------------------------------------------------------
        # Image of distribution
        _ , prior_logprob, log_det = flow(xy_grid)
        lnp_img = prior_logprob + log_det
        lnp_img = lnp_img.cpu().detach().numpy()
        lnp_img.shape = s

        p_img = np.exp(lnp_img)
        p_img /= np.sum(p_img)

        if p_true_img is None:
            vmax = np.max(p_img)
        else:
            vmax = 1.2 * np.max(p_true_img)

        ax1,ax2,ax3 = ax_arr[1]
        ax1.imshow(
            lnp_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmax=np.max(lnp_img),
            vmin=np.max(lnp_img) - 25.
        )
        ax1.set_title(r'$\ln p \left( y \right)$')
        ax2.imshow(
            p_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmin=0.,
            vmax=vmax
        )
        ax2.set_title(r'$p \left( y \right)$')

        if p_true_img is None:
            ax3.axis('off')
        else:
            ax3.imshow(
                p_true_img,
                extent=(-2., 2., -2., 2.),
                interpolation='nearest',
                vmin=0.,
                vmax=vmax
            )
            ax3.set_title(r'$p_{\mathrm{true}} \left( y \right)$')

        return fig
    return plot_fn


#----------------------------------------------------------------------------------
def get_training_callback(flow, every=500,
                                fname='nvp_{i:05d}.png',
                                p_true_fn=None,
                                **kwargs):
    """
    Returns a standard callback function that can be passed
    to train_flow. Every <every> steps callback prints the
    step number, loss and learning rate, and plots the flow.

    Inputs:
        flow (NormalizingFlow): Normalizing flow to be trained.
        every (int): The callback will run every <every> steps.
            Defaults to 500.
        fname (str): Pattern (using the new Python formatting
            language) used to generate the filename. Can use
            <i>, <n_steps> and <every>.
        p_true_fn (callable): Function that takes coordinates,
            and returns the true probability density. Defaults
            to None.
    """
    plt_fn = kwargs.get(
            'plt_fn',
            get_flow_plot_fn(flow, p_true_fn=p_true_fn)
    )
    #plt_fn = get_flow_plot_fn(flow, p_true_fn=p_true_fn)

    def training_callback(i, n_steps, loss_history, opt):
        if (i % every == 0):
            loss_avg = np.mean(loss_history[-50:])
            lr = opt.param_groups[0]['lr']
            print(
                f'Step {i+1: >5d} of {n_steps}: '
                f'<loss> = {loss_avg: >7.5f} , '
                f'lr = {lr:.4g}'
            )
            if plt_fn is not None:
                fig = plt_fn()
                namespace = dict(i=i, n_steps=n_steps, every=every)
                fig.savefig(fname.format(**namespace), dpi=100)
                plt.close(fig)

    return training_callback


#--------------------------------------------------------------------------------------------
def train_flow(flow, data, optimizer=None, batch_size=32,
               n_epochs=1, callback=None):
    """
    Trains a flow using the given data.

    Inputs:
      flow (NormalizingFlow): Normalizing flow to be trained.
      data (torch.Tensor): Observed points. Shape = (# of points, # of dim).
      optimizer (torch.optim): Optimizer to use.
          Defaults to the Rectified Adam
      batch_size (int): Number of points per training batch.
      n_epochs (int): Number of training epochs.
      callback (callable): Function that will be called
          at the end of each iteration. Function
          signature: f(i, n_steps, loss_history, opt), where i is the
          iteration index (int), n_steps (int) is the total number of
          steps, loss_history is a list of floats, containing the loss
          at each iteration so far, and opt is the optimizer.

    Returns:
      loss_history (list of floats): Loss after each training iteration.
    """

    n_samples = data.shape[0]
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,\
                                               shuffle=True, pin_memory=True)
    n_steps = n_epochs * n_samples // batch_size

    if optimizer is None:
        learning_rate = 2e-2
        opt = RAdam(flow.parameters(), lr=learning_rate)
    else:
        opt = optimizer

    decayRate = 0.998
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=decayRate)

#----------------------------------------------------------------------------------
    loss_history = []
    t1 = time()
    i = 0

    for e in range(int(n_epochs)):
        start_time = time()

        for batch_idx, data_batch in enumerate(train_loader):
            x = data_batch.to(device)
            zs, prior_logprob, log_det = flow(x)
            logprob = prior_logprob + log_det
            loss = -torch.mean(logprob)

            flow.zero_grad()
            loss.backward()
            opt.step()

            loss_history.append(loss.item())

            # save results for video
            #torch.save(flow.state_dict(), f'plummer_flow_video_' + str(i) + '.pth')

            callback(i, n_steps, loss_history, opt)
            i += 1

            if i % 10 == 0:
                my_lr_scheduler.step()

        #print(time()-start_time)

#----------------------------------------------------------------------------------
    t2 = time()
    loss_avg = np.mean(loss_history[-50:])
    n_steps = len(loss_history)
    print(f'<loss> = {loss_avg: >7.5f}')
    print(f'training time: {t2-t1:.1f} s ({(t2-t1)/n_steps:.4f} s/step)')

    return loss_history

#--------------------------------------------------------------------------------------------

def main():
    return 0

if __name__ == '__main__':
    main()



#==========================================================================================
class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x):
        return self.net(x)


#==========================================================================================
class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True).to(device)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True).to(device)) if shift else None

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


#==========================================================================================
class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = nn.init.orthogonal_(torch.randn(dim, dim).to(device))
        P, L, U = torch.lu_unpack(*torch.lu(Q))
        self.register_buffer('P', P)
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim).to(device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


#==========================================================================================
# Neural Spline flows
## Neural Spline Flow ##
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


#--------------------------------------------------------------------------------------
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


#--------------------------------------------------------------------------------------
def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives \
            + input_derivatives_plus_one - 2 * input_delta) \
            + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) \
            * (input_derivatives + input_derivatives_plus_one \
            - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
                      + ((input_derivatives + input_derivatives_plus_one \
                      - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * root.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) \
                    + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives \
                      + input_derivatives_plus_one - 2 * input_delta) \
                      * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * theta.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


#--------------------------------------------------------------------------------------
def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse=False,
                      tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet


#--------------------------------------------------------------------------------------
class NSF_CL(nn.Module):
    """ Neural spline flow, coupling layer, [Durkan et al. 2019] """

    def __init__(self, dim, K=8, B=3, hidden_dim=16, base_network=MLP):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.f1 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)
        self.f2 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0]).to(device)
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det

    def backward(self, z):
        log_det = torch.zeros(z.shape[0]).to(device)
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse = True, tail_bound = self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det


#==========================================================================================
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
